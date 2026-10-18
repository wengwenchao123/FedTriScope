"""跨城市窗口数据；BAY 使用原始 HDF 时间戳修正日内时间。"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CityWindows(Dataset):
    def __init__(self, speed, begin, end, lag, horizon, mean, std, time_ticks=None):
        self.speed = speed
        self.begin, self.end = begin, end
        self.lag, self.horizon = lag, horizon
        self.mean, self.std = mean, std
        self.time_ticks = time_ticks
        self.starts = np.arange(begin, end - lag - horizon + 1, dtype=np.int64)
        original_count = len(self.starts)
        if time_ticks is not None:
            # 一个窗口内的全部输入和标签必须等间隔；不能跨缺失时间段。
            breaks = np.r_[0, np.cumsum(np.diff(time_ticks) != 1)]
            last = self.starts + lag + horizon - 1
            self.starts = self.starts[breaks[last] == breaks[self.starts]]
        self.dropped_windows = original_count - len(self.starts)
        if len(self) <= 0:
            raise ValueError('数据切片不足以构造一个输入/预测窗口')

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        end = start + self.lag
        x = (self.speed[start:end] - self.mean) / self.std
        y = self.speed[end:end + self.horizon].T.copy()
        # 模型接收日历时间步，而速度/标签仍按原始数据行号读取。
        tick = start if self.time_ticks is None else int(self.time_ticks[start])
        return torch.from_numpy(x), torch.tensor(tick, dtype=torch.long), torch.from_numpy(y)

    def sample_batch(self, batch_size):
        samples = [self[int(i)] for i in torch.randperm(len(self))[:batch_size]]
        return tuple(torch.stack(items) for items in zip(*samples))


class CityData:
    def __init__(self, name, args):
        self.name = name
        raw = np.load(Path(args.data_path) / name / 'dataset.npy', mmap_mode='r')
        if raw.ndim != 3 or raw.shape[2] < 2 or not np.isfinite(raw).all():
            raise ValueError(f'{name}: 需要有限的 [T,N,C>=2] 数据，第二通道为日内时间')
        # 从实际日内时间推断采样频率，避免把十分钟数据当作五分钟数据。
        tid = np.asarray(raw[:, 0, 1], dtype=np.float64)
        delta = float((tid[1] - tid[0]) % 1)
        if delta <= 0:
            raise ValueError(f'{name}: 日内时间没有递增')
        self.steps_per_day = int(round(1 / delta))
        expected = np.arange(len(raw)) % self.steps_per_day / self.steps_per_day
        if not np.allclose(raw[:, :, 1], expected[:, None], atol=1e-5):
            raise ValueError(f'{name}: 目前要求连续且从午夜起始的日内时间通道')
        self.speed = np.array(raw[:, :, 0], dtype=np.float32)
        self.num_nodes = self.speed.shape[1]
        self.time_ticks = None
        self.time_source = 'dataset.npy:row_index'
        self.time_slot_corrections = 0
        self.time_gap_count = 0
        if name == 'pems-bay':
            self._load_bay_calendar(Path(args.data_path) / name / 'pems-bay.h5', raw)
        test_start = int(len(raw) * 0.8)
        if name == args.target_city:
            steps = args.target_steps_per_day or self.steps_per_day
            train_end = args.target_days * steps
            if train_end > test_start:
                raise ValueError('目标城市训练区间与测试区间重叠')
        else:
            # 与 pFedCTP 跨城市任务一致，非目标城市可使用完整序列。
            train_end = len(raw)
        fit = self.speed[:train_end] if args.normalization == 'train' else self.speed
        self.mean, self.std = float(fit.mean()), float(fit.std())
        if self.std <= 0:
            raise ValueError(f'{name}: 速度标准差必须大于零')
        self.train = CityWindows(self.speed, 0, train_end, args.lag, args.horizon,
                                 self.mean, self.std, self.time_ticks)
        self.test = CityWindows(self.speed, test_start, len(raw), args.lag, args.horizon,
                                self.mean, self.std, self.time_ticks)
        self.train_loader = DataLoader(self.train, batch_size=args.batch_size, shuffle=True,
                                       num_workers=0, pin_memory=args.device.startswith('cuda'))
        self.test_loader = DataLoader(self.test, batch_size=args.batch_size, shuffle=True,
                                      num_workers=0, pin_memory=args.device.startswith('cuda'))

    def _load_bay_calendar(self, path, raw):
        import pandas as pd

        if not path.is_file():
            raise FileNotFoundError(f'BAY 时间修正需要原始时间戳文件: {path}')
        frame = pd.read_hdf(path, key='speed')
        index = frame.index
        if frame.shape != self.speed.shape:
            raise ValueError('BAY HDF 与 dataset.npy 的行数/节点数不一致')
        if (not isinstance(index, pd.DatetimeIndex) or index.hasnans
                or not index.is_monotonic_increasing or not index.is_unique):
            raise ValueError('BAY HDF 必须包含严格递增且有效的 DatetimeIndex')
        # 允许原有少量缺失值填补，但拒绝行顺序/节点顺序不匹配的 HDF。
        matched = np.isclose(frame.to_numpy(), self.speed, rtol=1e-5, atol=1e-5)
        if np.any(matched.mean(axis=0) < 0.99):
            raise ValueError('BAY HDF 与 dataset.npy 的速度对齐率低于每节点 99%，请检查数据版本/顺序')
        # 日内嵌入表达本地钟表时间；有时区时去掉时区，不转换成 UTC。
        local = index.tz_localize(None) if index.tz is not None else index
        step_ns = 24 * 60 * 60 * 1_000_000_000 // self.steps_per_day
        nanos = local.to_numpy(dtype='datetime64[ns]').astype(np.int64)
        if np.any(nanos % step_ns):
            raise ValueError('BAY 时间戳未落在预期采样时间槽上')
        self.time_ticks = nanos // step_ns
        self.time_source = 'pems-bay.h5:/speed.index'
        stored = np.rint(np.asarray(raw[:, 0, 1]) * self.steps_per_day).astype(np.int64)
        self.time_slot_corrections = int(np.count_nonzero(
            stored != self.time_ticks % self.steps_per_day))
        self.time_gap_count = int(np.count_nonzero(np.diff(self.time_ticks) != 1))
