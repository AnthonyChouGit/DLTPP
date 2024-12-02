from DLTPP import DLTPP
import argparse
from train_eval import evaluate
from data import prepare_dataloader, get_max_t
import os
import setproctitle
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from utils import normalize, unnormalize
from torch.utils.tensorboard import SummaryWriter
import time

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--val_steps', type=int, default=10)
parser.add_argument('--cuda', type=str, default='6')
parser.add_argument('--dataname', type=str, default='stack_overflow')
parser.add_argument('--block_num', type=int, default=1)
parser.add_argument('--weight_decay', type=float, default=.01)
parser.add_argument('--embed_size', type=int, default=32)
parser.add_argument('--hist_len', type=int, default=20)
parser.add_argument('--pred_len', type=int, default=10)
parser.add_argument('--stride', type=int, default=10)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--steps', type=int, default=500)
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--title', type=str, default='test')
args = parser.parse_args()
if args.data_dir is None:
    args.data_dir = f'{os.path.dirname(__file__)}/dataset/{args.dataname}'
setproctitle.setproctitle(args.title)


if args.dataname == 'stack_overflow':
    num_types = 22
elif args.dataname == 'wikipedia':
    num_types = 8227
elif args.dataname == 'reddit':
    num_types = 984
    

config = {
    'train_steps': 500,
    'rounding_start':450,
    'block_num': args.block_num,
    'weight_decay': args.weight_decay
}

model = DLTPP(config, num_types, args.embed_size)
max_t = get_max_t(f'{args.data_dir}/train.pt')
ratio = max_t / 100
train_data, stats = prepare_dataloader(f'{args.data_dir}/train.pt', args.batch_size, True, ratio)
eval_data, _ = prepare_dataloader(f'{args.data_dir}/dev.pt', args.eval_batch_size, False, ratio)
test_data, _ = prepare_dataloader(f'{args.data_dir}/test.pt', args.eval_batch_size, False, ratio)

device = f'cuda:{args.cuda}'
model = model.to(device)
model.train()
model.log_mean, model.log_std, model.max_dt = stats
hist_len = args.hist_len
pred_len = args.pred_len
window_size = hist_len + pred_len
stride = args.stride
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
lr_scheduler = ExponentialLR(optimizer, gamma=0.999)
tb_writer = SummaryWriter(f'results/{args.title}')

print('training starts...')
for epoch in range(args.epochs):
    loss_total = 0.
    valid_windows = 0
    start = time.time()
    for batch in train_data:
        marks, times = batch
        batch_size, seq_len = marks.shape
        marks = marks.to(device)
        times = times.to(device)
        temp = torch.cat([torch.zeros(batch_size, 1, device=device), times], dim=1)
        seq_dts = temp[:, 1:] - temp[:, :-1]
        window_num = (seq_len-window_size) // stride + 1
        window_start = torch.arange(start=0, end=window_num*stride, step=stride, device=seq_dts.device)
        window_end = window_start + window_size
        mask = marks.eq(-1)
        for i in range(window_num):
            optimizer.zero_grad()
            hist_dts = seq_dts[:, window_start[i]:window_start[i]+hist_len]
            pred_dts = seq_dts[:, window_start[i]+hist_len:window_end[i]]
            hist_marks = marks[:, window_start[i]:window_start[i]+hist_len]
            pred_marks = marks[:, window_start[i]+hist_len:window_end[i]]
            window_mask = mask[:, window_start[i]:window_end[i]]
            temp = window_mask.sum(1)
            window_mask = (temp>0)
            _, hist_states = model.encode(hist_dts, hist_marks)
            pred_log_tau = torch.log(pred_dts.clamp(1e-8))
            pred_log_tau = normalize(pred_log_tau, model.log_mean, model.log_std)
            t = torch.randint(0, args.steps, (batch_size,), device=hist_dts.device)
            loss = model.diffusion.train_loss(pred_marks, pred_log_tau, t, hist_states)
            loss.masked_fill_(window_mask, 0)
            loss = loss.sum()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss_total += loss
                valid_windows += (~window_mask).sum()
        lr_scheduler.step()
    print(f'Epoch {epoch} finished in {time.time() - start} seconds.')
    avg_loss = loss_total / valid_windows
    tb_writer.add_scalar('train/loss', avg_loss.item(), epoch)
    if epoch % args.val_steps == 0:
        print('Evaluating...')
        start = time.time()
        mae, acc = evaluate(model, test_data, hist_len, pred_len, stride, stats, device)
        tb_writer.add_scalar('val/mae', mae.item(), epoch)
        tb_writer.add_scalar('val/acc', acc.item(), epoch)
        print(f'Evaluation finished in {time.time()-start} seconds.')
tb_writer.close()