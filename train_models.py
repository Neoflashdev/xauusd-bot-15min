"""
train_models.py
===============
Train the three frozen ensemble models on full historical data and save
them to disk as .pkl files.

Rules:
- This script trains on FULL historical data.
- The live bot (main.py) NEVER trains. It only loads these files.
- Do NOT run this script between paper trades or mid-session.
- Run once, then deploy the bot.

Output:
    models/expansion_model.pkl
    models/fake_breakout_model.pkl
    models/trend_continuation_model.pkl
    models/feature_metadata.pkl
"""

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except ImportError:
    print("[ERROR] xgboost not installed. Run: pip install xgboost")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Import backtest engine (reuse ALL existing logic — no rewrite)
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

# We import the shared functions from the backtest module
import importlib as _importlib
from config import MODELS_DIR, BACKTEST_MODULE
_bt_mod = _importlib.import_module(BACKTEST_MODULE)
CONFIG               = _bt_mod.CONFIG
load_csv             = _bt_mod.load_csv
add_m15_indicators   = _bt_mod.add_m15_indicators
build_4h_trend       = _bt_mod.build_4h_trend
merge_4h_into_m15    = _bt_mod.merge_4h_into_m15
detect_swings_no_repaint = _bt_mod.detect_swings_no_repaint
detect_msb           = _bt_mod.detect_msb
generate_signals     = _bt_mod.generate_signals
simulate_trades      = _bt_mod.simulate_trades
extract_ml_features_v2 = _bt_mod.extract_ml_features_v2
generate_ml_targets  = _bt_mod.generate_ml_targets

MODEL_DIR = str(MODELS_DIR)

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

FEATURE_PREFIXES = ("LIQ_", "MEM_", "VOL_", "GEO_")


def prepare_dataframe(path_m15: str, path_4h: str) -> pd.DataFrame:
    """Load CSVs, compute all indicators, merge 4H trend, detect swings/MSB."""
    df15 = load_csv(path_m15, "M15")
    df4h = load_csv(path_4h, "4H")

    df4h_trend = build_4h_trend(df4h)
    df15 = merge_4h_into_m15(df15, df4h_trend)
    df15 = add_m15_indicators(df15)

    high = df15["High"].values
    low  = df15["Low"].values
    last_sw_high, last_sw_low, sw_high_idx, sw_low_idx = detect_swings_no_repaint(
        high, low, CONFIG["swing_length"]
    )
    df15["last_sw_high"] = last_sw_high
    df15["last_sw_low"]  = last_sw_low
    df15["sw_high_idx"]  = sw_high_idx
    df15["sw_low_idx"]   = sw_low_idx

    bull_msb, bear_msb, msb_size = detect_msb(
        df15["Close"].values, last_sw_high, last_sw_low
    )
    df15["bull_msb"]  = bull_msb
    df15["bear_msb"]  = bear_msb
    df15["msb_size"]  = msb_size

    df15 = generate_signals(df15, allow_longs=True, allow_shorts=True)
    return df15


def build_feature_matrix(df15: pd.DataFrame) -> tuple:
    """Run backtest to get trades, then extract ML features."""
    trades = simulate_trades(df15, rr=CONFIG["rr_ratio"])
    print(f"  Total trades from simulation: {len(trades)}")

    df_ml = extract_ml_features_v2(trades, df15)
    df_ml = generate_ml_targets(trades, df_ml, df15)

    feature_cols = [c for c in df_ml.columns if c.startswith(FEATURE_PREFIXES)]

    X = df_ml[feature_cols].copy()

    # Impute NaN with column medians (computed from training data)
    col_medians = X.median()
    X = X.fillna(col_medians)

    return X, df_ml, col_medians, feature_cols


def train_model(X: pd.DataFrame, y: pd.Series, name: str) -> XGBClassifier:
    """Train a single XGBClassifier on full data."""
    # Drop rows where target is NaN
    valid = y.notna()
    X_valid = X[valid]
    y_valid = y[valid]

    print(f"  Training {name}: {len(X_valid)} samples, {X_valid.shape[1]} features")
    if y_valid.nunique() < 2:
        print(f"  [WARN] {name} has only one class. Skipping.")
        return None

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_valid.values, y_valid.values)
    return model


def save_artifact(obj, filename: str):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print(f"  FROZEN ENSEMBLE MODEL TRAINING ({CONFIG.get('path_m15', 'Data')})")
    print("  Strategy: Direction agnostic | MSB | Ensemble Score")
    print("=" * 60)

    # ── 1. Load and prepare data ──────────────────────────────────────────
    path_m15 = os.path.join(os.path.dirname(__file__), CONFIG["path_m15"])
    path_4h  = os.path.join(os.path.dirname(__file__), CONFIG["path_4h"])

    if not os.path.exists(path_m15):
        print(f"[ERROR] M15 CSV not found: {path_m15}")
        sys.exit(1)
    if not os.path.exists(path_4h):
        print(f"[ERROR] 4H CSV not found: {path_4h}")
        sys.exit(1)

    print("\n[1/4] Loading and preparing data...")
    df15 = prepare_dataframe(path_m15, path_4h)

    # ── 2. Build feature matrix ───────────────────────────────────────────
    print("\n[2/4] Extracting features and targets...")
    X, df_ml, col_medians, feature_cols = build_feature_matrix(df15)

    # ── 3. Train three models ─────────────────────────────────────────────
    print("\n[3/4] Training models on full historical data...")

    model_a = train_model(X, df_ml["TGT_EXP_BIN"], "Model A (ATR Expansion)")
    model_c = train_model(X, df_ml["TGT_FAKE"],    "Model C (Fake Breakout)")
    model_d = train_model(X, df_ml["TGT_CONT"],    "Model D (Trend Continuation)")

    if any(m is None for m in [model_a, model_c, model_d]):
        print("[ERROR] One or more models failed to train. Aborting.")
        sys.exit(1)

    # ── 4. Save models and feature metadata ───────────────────────────────
    print("\n[4/4] Saving frozen models and feature metadata...")

    save_artifact(model_a, "expansion_model.pkl")
    save_artifact(model_c, "fake_breakout_model.pkl")
    save_artifact(model_d, "trend_continuation_model.pkl")

    metadata = {
        "feature_cols": feature_cols,
        "col_medians":  col_medians.to_dict(),
        "n_features":   len(feature_cols),
        "trained_rows": len(X),
        "xgb_params":   XGB_PARAMS,
        "date_range": {
            "start": str(df15["Date"].min()),
            "end":   str(df15["Date"].max()),
        },
    }
    save_artifact(metadata, "feature_metadata.pkl")

    print("\n" + "=" * 60)
    print("  Training complete.")
    print(f"  Features : {len(feature_cols)}")
    print(f"  Rows used: {len(X)}")
    print(f"  Models saved to: {MODEL_DIR}/")
    print("=" * 60)
    print("\nNow run the live bot:")
    print("  python main.py")


if __name__ == "__main__":
    main()
