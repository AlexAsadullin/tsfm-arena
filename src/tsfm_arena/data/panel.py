"""The cross-sectional return panel every model and metric works on."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class ReturnPanel:
    """Log returns on a common time grid.

    ``returns[t, i]`` is the log return of asset ``i`` over the bar that ends at ``ts[t]`` (int64 ns, UTC,
    strictly increasing). NaN marks a missing observation. The forward ``h``-bar target at row ``t`` is
    ``returns[t + 1 : t + 1 + h].sum(axis=0)``.
    """

    ts: IntArray
    symbols: tuple[str, ...]
    returns: FloatArray

    def __post_init__(self) -> None:
        if self.ts.ndim != 1 or self.returns.ndim != 2:
            raise ValueError("ts must be 1-D and returns 2-D")
        if self.returns.shape != (self.ts.shape[0], len(self.symbols)):
            raise ValueError(
                f"returns shape {self.returns.shape} != ({self.ts.shape[0]}, {len(self.symbols)})"
            )
        if self.ts.shape[0] > 1 and not bool(np.all(np.diff(self.ts) > 0)):
            raise ValueError("ts must be strictly increasing")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be unique")

    @property
    def n_periods(self) -> int:
        return int(self.ts.shape[0])

    @property
    def n_assets(self) -> int:
        return len(self.symbols)

    def upto(self, ts_ns: int) -> ReturnPanel:
        """Rows with ``ts <= ts_ns``: everything knowable at ``ts_ns`` on a revision-free source."""
        end = int(np.searchsorted(self.ts, ts_ns, side="right"))
        return self.rows(0, end)

    def between(self, start_ns: int | None, end_ns: int | None) -> ReturnPanel:
        """Rows with ``start_ns <= ts <= end_ns`` (open bounds when None)."""
        lo = 0 if start_ns is None else int(np.searchsorted(self.ts, start_ns, side="left"))
        hi = self.n_periods if end_ns is None else int(np.searchsorted(self.ts, end_ns, side="right"))
        return self.rows(lo, hi)

    def rows(self, start: int, stop: int) -> ReturnPanel:
        return ReturnPanel(self.ts[start:stop], self.symbols, self.returns[start:stop])

    def select(self, symbols: Sequence[str]) -> ReturnPanel:
        """Columns for ``symbols``, in that order. Unknown symbols raise ``KeyError``."""
        index = {s: i for i, s in enumerate(self.symbols)}
        missing = [s for s in symbols if s not in index]
        if missing:
            raise KeyError(f"unknown symbols: {missing}")
        cols = [index[s] for s in symbols]
        return ReturnPanel(self.ts, tuple(symbols), self.returns[:, cols])

    def forward_returns(self, horizon: int) -> FloatArray:
        """``(T, N)`` cumulative log return over the next ``horizon`` bars; NaN where unavailable."""
        out = np.full_like(self.returns, np.nan)
        if self.n_periods > horizon:
            windows = np.lib.stride_tricks.sliding_window_view(self.returns[1:], horizon, axis=0)
            out[: self.n_periods - horizon] = windows.sum(axis=-1)
        return out

    def fingerprint(self) -> str:
        """Content hash ``sha256:<hex>`` over timestamps, symbols and values."""
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.ts).tobytes())
        digest.update("\x1f".join(self.symbols).encode())
        digest.update(np.ascontiguousarray(self.returns).tobytes())
        return f"sha256:{digest.hexdigest()}"

    def to_long(self) -> pd.DataFrame:
        """Long frame with columns ``ts`` (int64 ns), ``symbol``, ``ret``; NaN rows are dropped."""
        frame = pd.DataFrame(self.returns, columns=list(self.symbols))
        frame.insert(0, "ts", self.ts)
        long = frame.melt(id_vars="ts", var_name="symbol", value_name="ret").dropna()
        return long.sort_values(["ts", "symbol"]).reset_index(drop=True)

    @classmethod
    def from_long(
        cls,
        frame: pd.DataFrame,
        time_col: str,
        symbol_col: str,
        value_col: str,
        value_kind: Literal["return", "price"],
    ) -> ReturnPanel:
        """Build a panel from a long frame of log returns or prices.

        Times may be int64 ns or anything ``pandas.to_datetime`` parses (naive values are read as UTC).
        Prices become log returns ``log(p_t / p_{t-1})`` per symbol; the first row is dropped.
        """
        if frame.empty:
            raise ValueError("empty frame: no rows to build a panel from")
        missing = {time_col, symbol_col, value_col} - set(frame.columns)
        if missing:
            raise KeyError(f"missing columns: {sorted(missing)}")
        ts = _to_ns_series(frame[time_col])
        wide = (
            pd.DataFrame({"ts": ts, "symbol": frame[symbol_col].astype(str), "v": frame[value_col]})
            .pivot_table(index="ts", columns="symbol", values="v", aggfunc="last")
            .sort_index()
        )
        values = wide.to_numpy(dtype=np.float64)
        if value_kind == "price":
            if bool(np.any(values[~np.isnan(values)] <= 0)):
                raise ValueError("prices must be positive")
            values = np.diff(np.log(values), axis=0)
            index = wide.index.to_numpy(dtype=np.int64)[1:]
        else:
            index = wide.index.to_numpy(dtype=np.int64)
        return cls(index, tuple(str(c) for c in wide.columns), values)


def _to_ns_series(column: pd.Series) -> npt.NDArray[np.int64]:
    if pd.api.types.is_integer_dtype(column):
        return np.asarray(column.to_numpy(), dtype=np.int64)
    parsed = pd.to_datetime(column, utc=True)
    return np.asarray(parsed.astype("datetime64[ns, UTC]").astype("int64").to_numpy(), dtype=np.int64)
