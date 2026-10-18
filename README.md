# Required local data

Place the existing processed files here (not included in the release):

```
data/
  metr-la/dataset.npy
  pems-bay/dataset.npy
  pems-bay/pems-bay.h5
  shenzhen/dataset.npy
  chengdu/dataset.npy
```

Each `dataset.npy` is a finite numeric array `[time, nodes, channels>=2]`.
Channel 0 contains speed in original units; channel 1 is the legacy daily slot
fraction `row % steps_per_day / steps_per_day`, starting at midnight. The daily
resolution is inferred from channel 1 (288 for LA/BAY, 144 for the DiDi cities).
This channel is metadata only: the model receives speed, not the time scalar.

For BAY the HDF `speed` table must have a strictly increasing DatetimeIndex and
match row/node ordering of the processed array. This calendar supersedes the
legacy synthetic slots. Per-node speed alignment must be at least 99%; windows
crossing timestamp gaps are excluded. Other cities retain the existing processed
row-index time protocol; no unprovided calendar dates are invented.

Windows are constructed wholly within each split (no boundary crossing). Retain
the same processed data version when reproducing historical results. Acquire data
from its authorized provider; this repository does not distribute the datasets.
