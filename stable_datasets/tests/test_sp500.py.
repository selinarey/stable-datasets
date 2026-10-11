import numpy as np

from stable_datasets.timeseries.SP500 import make_windows


def test_make_windows():
    import pandas as pd

    dates = pd.date_range("2020-01-01", periods=35)

    df = pd.DataFrame({
        "date": dates,
        "open": np.arange(100, 135),
        "high": np.arange(101, 136),
        "low": np.arange(99, 134),
        "close": np.arange(100, 135),
        "volume": np.full(35, 1000),
    })

    windows = make_windows(df, sequence_length=30)

    # 35 days produce 6 overlapping 30-day windows.
    assert len(windows) == 6

    # Each window has 30 days and 5 features.
    assert windows[0].shape == (30, 5)

    # The first day's closing price is the normalization reference.
    assert windows[0][0, 3] == 0.0

    # No missing or infinite values.
    assert np.isfinite(windows[0]).all()
