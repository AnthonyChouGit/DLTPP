import torch
import torch.utils.data

class EventData(torch.utils.data.Dataset):
    def __init__(self, marks, times, ratio=None) -> None:
        '''
            data: torch.tensor (data_num, max_seq_len)
        '''
        super().__init__()
        self.marks = marks # List of int tensors
        times = [seq_ts-seq_ts[0] for seq_ts in times]
        if ratio is not None:
            times = [seq_ts/ratio for seq_ts in times]
        self.times = times
        dts = [seq_ts[1:]-seq_ts[:-1] for seq_ts in times]
        all_dts = torch.cat(dts)
        zero_mask = all_dts.gt(0)
        all_dts = all_dts.masked_select(zero_mask)
        self.log_mean, self.log_std, self.max_dt = (all_dts+1e-6).log().mean().item(), (all_dts+1e-6).log().std().item(), all_dts.max().item()
        # print()
        
    def get_stats(self):
        return self.log_mean, self.log_std, self.max_dt

    def __len__(self):
        return len(self.marks)

    def __getitem__(self, idx):
        marks = self.marks[idx]
        times = self.times[idx]
        return marks, times

def collate_fn(insts):
    marks, times = zip(*insts)
    marks = torch.nn.utils.rnn.pad_sequence(marks, batch_first=True, padding_value=-1)
    times = torch.nn.utils.rnn.pad_sequence(times, batch_first=True)
    return marks, times

def get_dataloader(marks, times, batch_size, shuffle=True, ratio=None):
    ds = EventData(marks, times, ratio)

    dl = torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )

    return dl, ds.get_stats()

def prepare_dataloader(data_dir, batch_size, shuffle=True, ratio=None):
    marks, times = torch.load(data_dir)
    dataloader, stats = get_dataloader(marks, times, batch_size, shuffle, ratio)
    return dataloader, stats

def get_max_t(data_dir):
    _, times = torch.load(data_dir)
    times = [seq_ts-seq_ts[0] for seq_ts in times]
    all_times = torch.cat(times)
    max_t = all_times.max()
    return max_t

def get_stats(data_dir):
    _, times = torch.load(data_dir)
    dts = [seq_ts[1:]-seq_ts[:-1] for seq_ts in times]
    all_dts = torch.cat(dts)
    zero_mask = all_dts.ne(0)
    all_dts = all_dts.masked_select(zero_mask)
    mean, std = all_dts.mean(), all_dts.std()
    return mean, std
