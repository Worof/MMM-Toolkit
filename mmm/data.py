"""Loading, validation, auditing and design-matrix construction (no leakage across CV folds)."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

from .config import CFG, MMMConfig


def _schema(cfg: MMMConfig) -> DataFrameSchema:
    cols = {
        cfg.date_column: Column(pa.DateTime, coerce=True, unique=True),
        cfg.target_column: Column(float, Check.ge(0), coerce=True),
    }
    for ch in cfg.spend_columns:
        cols[ch] = Column(float, Check.ge(0), nullable=True, coerce=True)
    for c in cfg.control_columns:
        cols[c] = Column(float, nullable=False, coerce=True)
    return DataFrameSchema(cols, strict=False)


def load_raw(path: Optional[str] = None, cfg: MMMConfig = CFG) -> pd.DataFrame:
    df = pd.read_csv(path or cfg.data_path)
    df[cfg.date_column] = pd.to_datetime(df[cfg.date_column])
    df = _schema(cfg).validate(df).sort_values(cfg.date_column).reset_index(drop=True)
    df[cfg.spend_columns] = df[cfg.spend_columns].fillna(0.0)
    return df


def audit(df: pd.DataFrame, cfg: MMMConfig = CFG) -> dict:
    """Data-quality and identifiability checks an MMM reviewer will ask about first."""
    spend = df[cfg.spend_columns]
    y = df[cfg.target_column]
    gaps = df[cfg.date_column].diff().dt.days.dropna()
    corr = spend.corr()
    cols = list(spend.columns)
    high_corr = [(a, b, round(float(corr.loc[a, b]), 2))
                 for i, a in enumerate(cols) for b in cols[i + 1:] if abs(corr.loc[a, b]) > 0.7]
    try:  # variance inflation factor from the inverse correlation matrix
        vif = dict(zip(cols, np.diag(np.linalg.inv(corr.values)).round(2).tolist()))
    except np.linalg.LinAlgError:
        vif = {c: float("inf") for c in cols}
    mad = np.median(np.abs(y - y.median())) or 1.0
    robust_z = 0.6745 * (y - y.median()) / mad
    n_params = 3 * len(cols) + len(cfg.control_columns) + 8
    return dict(
        rows=len(df),
        start=str(df[cfg.date_column].min().date()),
        end=str(df[cfg.date_column].max().date()),
        weekly_regular=bool((gaps == 7).all()),
        zero_spend_share=(spend == 0).mean().round(3).to_dict(),
        spend_share=(spend.sum() / spend.sum().sum()).round(3).to_dict(),
        high_correlation_pairs=high_corr,
        vif=vif,
        outlier_weeks=df.loc[robust_z.abs() > 3.5, cfg.date_column].dt.strftime("%Y-%m-%d").tolist(),
        obs_per_parameter=round(len(df) / n_params, 1),
    )


def print_audit(a: dict) -> None:
    print(f"Rows: {a['rows']} | {a['start']} -> {a['end']} | regular weekly grid: {a['weekly_regular']}")
    print("Spend share:", a["spend_share"])
    print("Zero-spend share (flighting):", a["zero_spend_share"])
    print("Highly correlated channel pairs (|r|>0.7):", a["high_correlation_pairs"] or "none")
    print("VIF:", a["vif"], "(>5 = weak identifiability)")
    print("Outlier weeks:", a["outlier_weeks"] or "none")
    print("Observations per parameter:", a["obs_per_parameter"], "(aim for >= 10)")


@dataclass
class DesignSpec:
    """Everything learned from the TRAIN window; re-used unchanged on hold-out data."""
    continuous: List[str]
    means: Dict[str, float]
    stds: Dict[str, float]
    event_cols: List[str]
    origin: pd.Timestamp
    trend_mean: float
    trend_std: float
    add_trend: bool

    @property
    def controls(self) -> List[str]:
        return self.continuous + self.event_cols + (["trend"] if self.add_trend else [])


def _event_dummies(df: pd.DataFrame, cfg: MMMConfig, columns: Optional[List[str]] = None) -> pd.DataFrame:
    if not cfg.event_column or cfg.event_column not in df.columns:
        return pd.DataFrame(index=df.index)
    ev = df[cfg.event_column].astype(str)
    ev = ev.where(~ev.isin([cfg.event_none_label, "nan", "None", ""]), other=np.nan)
    d = pd.get_dummies(ev).astype(float)
    d.columns = [f"event_{c}" for c in d.columns]
    if columns is not None:
        d = d.reindex(columns=columns, fill_value=0.0)
    return d


def fit_spec(train: pd.DataFrame, cfg: MMMConfig = CFG) -> DesignSpec:
    weeks = (train[cfg.date_column] - train[cfg.date_column].min()).dt.days / 7.0
    return DesignSpec(
        continuous=list(cfg.control_columns),
        means={c: float(train[c].mean()) for c in cfg.control_columns},
        stds={c: float(train[c].std(ddof=0) or 1.0) for c in cfg.control_columns},
        event_cols=list(_event_dummies(train, cfg).columns),
        origin=train[cfg.date_column].min(),
        trend_mean=float(weeks.mean()),
        trend_std=float(weeks.std(ddof=0) or 1.0),
        add_trend=cfg.add_trend,
    )


def make_design(df: pd.DataFrame, spec: DesignSpec, cfg: MMMConfig = CFG) -> Tuple[pd.DataFrame, pd.Series]:
    """X (date + spend + scaled controls) and y. Controls are z-scored with TRAIN statistics so the
    N(0, sigma) priors on control effects are meaningful (raw values of ~1e7 would defeat them)."""
    X = df[[cfg.date_column] + cfg.spend_columns].copy().reset_index(drop=True)
    for c in spec.continuous:
        X[c] = ((df[c] - spec.means[c]) / spec.stds[c]).values
    ev = _event_dummies(df, cfg, spec.event_cols).reset_index(drop=True)
    for c in spec.event_cols:
        X[c] = ev[c].values
    if spec.add_trend:
        weeks = (df[cfg.date_column] - spec.origin).dt.days / 7.0
        X["trend"] = ((weeks - spec.trend_mean) / spec.trend_std).values
    return X, df[cfg.target_column].reset_index(drop=True)


def time_series_splits(n: int, n_folds: int, test_size: int):
    """Expanding-window splits with consecutive, non-overlapping test windows."""
    if n <= n_folds * test_size:
        raise ValueError("Not enough rows for the requested folds/test_size.")
    first = n - n_folds * test_size
    return [(np.arange(0, first + i * test_size), np.arange(first + i * test_size, first + (i + 1) * test_size))
            for i in range(n_folds)]
