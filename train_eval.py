# from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score

# def train(model, train_data_loader, hist_len, pred_len, stride, stats, max_epoch, device):
#     model = model.to(device)
#     model.train()
#     model.log_mean, model.log_std, model.max_dt = stats
#     for epoch in tqdm(range(max_epoch)):
#         for ind, batch in enumerate(train_data_loader):
#             marks, times = batch
#             batch_size = marks.shape[0]
#             marks = marks.to(device)
#             times = times.to(device)
#             temp = torch.cat([torch.zeros(batch_size, 1, device=device), times], dim=1)
#             seq_dts = temp[:, 1:] - temp[:, :-1]
            
#             model.train_self(seq_dts, marks, hist_len, pred_len, stride)
#     torch.save(model.state_dict(), 'model')

@torch.no_grad()
def evaluate(model, data, hist_len, pred_len, stride, stats, device):
    model.log_mean, model.log_std, model.max_dt = stats
    window_size = hist_len + pred_len
    model = model.to(device)
    model.eval()
    all_truth_marks = list()
    all_pred_marks = list()
    all_error = list()
    for batch in data:
        marks, times = batch

        marks = marks.to(device)
        times = times.to(device)
        batch_size, seq_len = times.shape
        window_num = (seq_len-window_size) // stride + 1
        window_start = torch.arange(start=0, end=window_num*stride, step=stride, device=device)
        window_end = window_start + window_size
        mask = marks.eq(-1)
        temp = torch.cat([times[:, 0:1], times], dim=1)
        dts = temp[:, 1:] - temp[:, :-1]
        for i in range(window_num): # We need to guarantee that there is no padding elements in the whole window, 
                                    # if there is, deactivate this window in metrics computation
            hist_marks = marks[:, window_start[i]:window_start[i]+hist_len]
            hist_dts = dts[:, window_start[i]:window_start[i]+hist_len]
            truth_marks = marks[:, window_start[i]+hist_len:window_end[i]] # (batch_size, pred_len)

            truth_dts = dts[:, window_start[i]+hist_len:window_end[i]]
            truth_times = torch.cumsum(truth_dts, dim=1)

            window_mask = mask[:, window_start[i]:window_end[i]]
            temp = window_mask.sum(1)
            window_mask = (temp>0)

            pred_marks, pred_times = model.predict(hist_dts, hist_marks, pred_len) # (batch_size, sample_num, pred_len)
            err = torch.abs(pred_times - truth_times[:, None, :]).sum(-1).mean(-1) # (batch_size, )
            all_error.append(err.masked_select(~window_mask))

            truth_marks = truth_marks[:, None, :].expand(batch_size, 200, pred_len) # (batch_size, sample_num, pred_len)
            all_truth_marks.append(truth_marks.masked_select(~window_mask[:, None, None])) 
            all_pred_marks.append(pred_marks.masked_select(~window_mask[:, None, None]))
    all_error = torch.cat(all_error)
    all_truth_marks = torch.cat(all_truth_marks)
    all_pred_marks = torch.cat(all_pred_marks)
    mae = all_error.mean()
    acc = accuracy_score(all_truth_marks.cpu(), all_pred_marks.cpu())
    # with open(self.output_file, 'a') as f:
    #     f.write(f'mae={mae}, acc={acc}\n')
    model.train()
    return mae, acc

