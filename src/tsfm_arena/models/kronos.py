"""Optional Kronos adapter (Shi et al., github.com/shiyu-coder/Kronos).

Kronos is not on PyPI; it is loaded from a source checkout (``kronos.repo_path``) and its published HF
weights. Kronos forecasts K-lines (OHLCV), so the adapter rebuilds a candle series from the returns:
``close`` is the exponentiated cumulative log return, ``open`` is the previous close, ``high``/``low`` are
the max/min of the two and volume is zero. ``KronosPredictor.predict`` samples one path per call, so the
adapter draws ``n_paths`` paths per asset and reports their mean and empirical quantiles of the
cumulative log return ``log(close[t+h] / close[t])``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch

from tsfm_arena.data.panel import FloatArray, ReturnPanel
from tsfm_arena.models.base import Forecast, ModelUnavailableError
from tsfm_arena.settings import KronosSettings

CANDLE_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


class KronosPredictorLike(Protocol):
    def predict(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        T: float,
        top_p: float,
        sample_count: int,
        verbose: bool,
    ) -> pd.DataFrame: ...


def candles_from_returns(returns: FloatArray) -> pd.DataFrame:
    """Candle frame for one asset from its log returns (NaN treated as zero return)."""
    close = np.exp(np.cumsum(np.nan_to_num(returns)))
    open_ = np.concatenate([[1.0], close[:-1]])
    zeros = np.zeros_like(close)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close),
            "low": np.minimum(open_, close),
            "close": close,
            "volume": zeros,
            "amount": zeros,
        }
    )


class KronosForecaster:
    needs_fit = False

    def __init__(
        self,
        name: str,
        predictor: KronosPredictorLike,
        context_length: int,
        cfg: KronosSettings,
        seed: int,
    ) -> None:
        self.name = name
        self.predictor = predictor
        self.context_length = min(context_length, cfg.max_context)
        self.cfg = cfg
        self.bar = pd.Timedelta(cfg.bar)
        self.seed = seed

    def fit(self, history: ReturnPanel, horizon: int) -> None:
        """Pretrained; nothing to fit at evaluation time."""

    def predict(self, context: FloatArray, horizon: int, levels: Sequence[float]) -> Forecast:
        window = context[:, -self.context_length :]
        length = window.shape[1]
        x_ts = pd.Series(pd.Timestamp(0, tz="UTC") + self.bar * np.arange(length))
        y_ts = pd.Series(x_ts.iloc[-1] + self.bar * np.arange(1, horizon + 1))
        means, quants = [], []
        for row in window:
            candles = candles_from_returns(row)
            last_close = float(candles["close"].iloc[-1])
            paths = []
            for k in range(self.cfg.n_paths):
                torch.manual_seed(self.seed + k)
                out = self.predictor.predict(
                    df=candles,
                    x_timestamp=x_ts,
                    y_timestamp=y_ts,
                    pred_len=horizon,
                    T=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    sample_count=1,
                    verbose=False,
                )
                paths.append(np.log(float(out["close"].iloc[horizon - 1]) / last_close))
            arr = np.asarray(paths)
            means.append(arr.mean())
            quants.append(np.quantile(arr, np.asarray(levels)))
        return Forecast(np.asarray(means), np.asarray(quants), tuple(levels))


def load_predictor(
    hf_id: str, tokenizer_id: str, cfg: KronosSettings, cache_dir: Path, device: str, allow_downloads: bool
) -> Any:
    """Import Kronos from its source checkout and load the published weights."""
    if cfg.repo_path is None or not (cfg.repo_path / "model").is_dir():
        raise ModelUnavailableError("Kronos needs a source checkout: set TSFM_KRONOS__REPO_PATH")
    if str(cfg.repo_path) not in sys.path:
        sys.path.insert(0, str(cfg.repo_path))
    try:
        from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore[import-not-found]

        kwargs = {"cache_dir": str(cache_dir), "local_files_only": not allow_downloads}
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id, **kwargs)
        model = Kronos.from_pretrained(hf_id, **kwargs)
    except (ImportError, OSError) as exc:
        raise ModelUnavailableError(f"Kronos {hf_id!r} could not be loaded: {exc}") from exc
    return KronosPredictor(model, tokenizer, device=device, max_context=cfg.max_context)
