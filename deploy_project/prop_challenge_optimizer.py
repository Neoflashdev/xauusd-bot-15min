import os
import sys
import argparse
import pandas as pd
import numpy as np
import datetime
import importlib
import warnings
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import random
import csv
import json

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. CONSTANTS & CONFIGURATIONS
# -----------------------------------------------------------------------------

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
RRS = [1.2, 1.5, 1.8, 2.0]
RISKS = [0.5, 0.75, 1.0, 1.25]
CONSEC_LOSSES = [2, 3]
MAX_TRADES_DAY = [1, 2, 3]

# 5 Sets of ML Ensemble Weights (Exp, Cont, Fake)
WEIGHT_SETS = [
    (0.35, 0.35, 0.30),
    (0.25, 0.45, 0.30),
    (0.25, 0.30, 0.45),
    (0.20, 0.40, 0.40),
    (0.40, 0.20, 0.40)
]

# Prop Firm Rules
PROP_FIRMS = {
    "FundedNext": {
        "phase1_target": 8.0, 
        "phase2_target": 5.0, 
        "daily_loss": 5.0, 
        "max_loss": 10.0, 
        "min_days": 5
    },
    "FundingPips": {
        "phase1_target": 10.0, 
        "phase2_target": 6.0, 
        "daily_loss": 4.0, 
        "max_loss": 12.0, 
        "min_days": 0
    },
    "EquityEdge": {
        "phase1_target": 10.0, 
        "phase2_target": 5.0, 
        "daily_loss": 5.0, 
        "max_loss": 10.0, 
        "min_days": 2
    }
}

# -----------------------------------------------------------------------------
# 2. CLI PARSING
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Prop Challenge Optimizer V2")
    parser.add_argument("--symbol", type=str, default="ALL", choices=["ALL", "BTCUSD", "XAUUSD"],
                        help="Which symbol to optimize (default: ALL)")
    parser.add_argument("--mode", type=str, default="search", choices=["search", "full"],
                        help="Mode: 'search' (fast) or 'full' (rigorous validation)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from progress file if it exists")
    return parser.parse_args()

# -----------------------------------------------------------------------------
# 3. OOF CACHE GENERATION
# -----------------------------------------------------------------------------

def build_oof_cache(symbol, direction, rr, xgb_n_estimators, xgb_max_depth):
    """
    Generates Out-Of-Fold probabilities for a specific block and saves to cache/.
    """
    cache_path = f"cache/{symbol}_{direction}_RR{rr}_oof.csv"
    if os.path.exists(cache_path):
        print(f"[CACHE] Found existing cache: {cache_path}")
        return cache_path
        
    print(f"\n[CACHE] Generating OOF cache for {symbol} | {direction} | RR {rr}")
    
    # Load module dynamically
    mod_name = "BTCUSD_Backtest" if symbol == "BTCUSD" else "XAUUSD_Colab_Backtest"
    mod = importlib.import_module(mod_name)
    
    # Load and prep data
    path_m15 = f"{symbol}_15min.csv"
    path_4h  = f"{symbol}_4H.csv"
    
    df15 = mod.load_csv(path_m15, "M15")
    df4h = mod.load_csv(path_4h, "4H")
    df4h_trend = mod.build_4h_trend(df4h)
    df15 = mod.merge_4h_into_m15(df15, df4h_trend)
    df15 = mod.add_m15_indicators(df15)
    
    high, low = df15["High"].values, df15["Low"].values
    last_sw_high, last_sw_low, sw_high_idx, sw_low_idx = mod.detect_swings_no_repaint(high, low, 20)
    df15["last_sw_high"] = last_sw_high
    df15["last_sw_low"]  = last_sw_low
    
    bull_msb, bear_msb, msb_size = mod.detect_msb(df15["Close"].values, last_sw_high, last_sw_low)
    df15["bull_msb"] = bull_msb
    df15["bear_msb"] = bear_msb
    df15["msb_size"] = msb_size
    
    allow_longs  = direction in ["LONG", "BOTH"]
    allow_shorts = direction in ["SHORT", "BOTH"]
    df_sig = mod.generate_signals(df15, allow_longs=allow_longs, allow_shorts=allow_shorts)
    
    trades = mod.simulate_trades(df_sig, rr=rr)
    print(f"  -> Base trades generated: {len(trades)}")
    
    if len(trades) < 50:
        print("  -> Not enough trades to train OOF. Skipping.")
        return None
        
    df_ml = mod.extract_ml_features_v2(trades, df_sig)
    df_ml = mod.generate_ml_targets(trades, df_ml, df_sig)
    
    # Map exit_time natively
    entry_to_exit = {t['entry_time']: t['exit_time'] for t in trades}
    df_ml['exit_time'] = df_ml['entry_time'].map(entry_to_exit)
    df_ml['pnl_R'] = np.where(df_ml['result'] == 'TP', rr, -1.0)
    
    # Extract features
    feat_cols = [c for c in df_ml.columns if c.startswith(("LIQ_", "MEM_", "VOL_", "GEO_"))]
    X_arr = df_ml[feat_cols].values
    y_exp = df_ml['TGT_EXP_BIN'].values
    y_fake = df_ml['TGT_FAKE'].values
    y_cont = df_ml['TGT_CONT'].values
    
    n_folds = 5
    tscv = TimeSeriesSplit(n_splits=n_folds)
    n = len(df_ml)
    
    oof_exp = np.full(n, np.nan)
    oof_fake = np.full(n, np.nan)
    oof_cont = np.full(n, np.nan)
    
    xgb_params = {
        "n_estimators": xgb_n_estimators,
        "max_depth": xgb_max_depth,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1
    }
    
    print(f"  -> Training {n_folds}-Fold TimeSeriesSplit OOF...")
    for train_idx, test_idx in tscv.split(X_arr):
        X_tr, X_te = X_arr[train_idx].copy(), X_arr[test_idx].copy()
        tr_med = np.nanmedian(X_tr, axis=0)
        for j in range(X_tr.shape[1]):
            X_tr[np.isnan(X_tr[:, j]), j] = tr_med[j]
            X_te[np.isnan(X_te[:, j]), j] = tr_med[j]
            
        for y, oof_arr in [(y_exp, oof_exp), (y_fake, oof_fake), (y_cont, oof_cont)]:
            y_tr, y_te = y[train_idx], y[test_idx]
            valid = ~np.isnan(y_tr)
            if valid.sum() > 2 and len(np.unique(y_tr[valid])) > 1:
                m = XGBClassifier(**xgb_params)
                m.fit(X_tr[valid], y_tr[valid])
                oof_arr[test_idx] = m.predict_proba(X_te)[:, 1]
                
    df_ml['oof_exp'] = oof_exp
    df_ml['oof_fake'] = oof_fake
    df_ml['oof_cont'] = oof_cont
    
    # Calculate AUCs
    valid_mask = ~np.isnan(df_ml['oof_exp']) & ~np.isnan(df_ml['TGT_EXP_BIN'])
    auc_exp = roc_auc_score(df_ml.loc[valid_mask, 'TGT_EXP_BIN'], df_ml.loc[valid_mask, 'oof_exp']) if valid_mask.sum() > 0 else 0
    valid_mask = ~np.isnan(df_ml['oof_fake']) & ~np.isnan(df_ml['TGT_FAKE'])
    auc_fake = roc_auc_score(df_ml.loc[valid_mask, 'TGT_FAKE'], df_ml.loc[valid_mask, 'oof_fake']) if valid_mask.sum() > 0 else 0
    valid_mask = ~np.isnan(df_ml['oof_cont']) & ~np.isnan(df_ml['TGT_CONT'])
    auc_cont = roc_auc_score(df_ml.loc[valid_mask, 'TGT_CONT'], df_ml.loc[valid_mask, 'oof_cont']) if valid_mask.sum() > 0 else 0
    
    print(f"  -> Valid ML rows generated: {valid_mask.sum()}")
    print(f"  -> OOF AUCs | Exp: {auc_exp:.3f} | Fake: {auc_fake:.3f} | Cont: {auc_cont:.3f}")
    
    # Save the exact required columns
    export_cols = ["entry_time", "exit_time", "signal_idx", "entry_idx", "direction", "result", "pnl_R", "oof_exp", "oof_fake", "oof_cont"]
    out_df = df_ml[export_cols].copy()
    out_df.to_csv(cache_path, index=False)
    print(f"  -> Cache saved successfully to: {cache_path}")
    
    return cache_path

# -----------------------------------------------------------------------------
# 4. CHRONOLOGICAL EQUITY & PROP SIMULATOR
# -----------------------------------------------------------------------------

def run_rolling_prop_sim(df, rules):
    """
    Simulate Prop Firm Phase 1 and Phase 2 starting from the first trade of every month.
    """
    if len(df) == 0:
        return {
            "runs": 0, "passes": 0, "pass_rate": 0.0,
            "med_days": 0, "p25_days": 0, "p75_days": 0,
            "fail_max_loss": 0, "fail_daily_loss": 0, "fail_time": 0
        }
        
    # Extract to fast numpy arrays / python primitives to avoid pandas .iloc overhead
    entry_months = df['entry_time'].dt.month.values
    entry_years = df['entry_time'].dt.year.values
    entry_dates = df['entry_time'].dt.date.values
    exit_dates = df['exit_time'].dt.date.values
    pnl_pcts = df['pnl_pct'].values / 100.0
    
    n_trades = len(df)
    
    start_indices = []
    last_month = None
    for i in range(n_trades):
        m = entry_months[i]
        y = entry_years[i]
        if (y, m) != last_month:
            start_indices.append(i)
            last_month = (y, m)
            
    def sim_phase(start_idx, target_pct, daily_loss_limit, max_loss_limit, min_days):
        equity = 1.0
        start_balance = 1.0
        daily_start_equity = 1.0
        last_date = None
        
        challenge_start_date = entry_dates[start_idx]
        
        for i in range(start_idx, n_trades):
            t_date = exit_dates[i]
            pnl = pnl_pcts[i]
            
            if t_date != last_date:
                daily_start_equity = equity
                last_date = t_date
                
            equity += pnl
            
            calendar_days = (t_date - challenge_start_date).days + 1
            
            # Check Daily Loss
            daily_dd_pct = (equity - daily_start_equity) / daily_start_equity * 100.0
            if daily_dd_pct <= -daily_loss_limit:
                return False, calendar_days, i, "Daily Loss Limit"
                
            # Check Max Loss
            total_dd_pct = (equity - start_balance) / start_balance * 100.0
            if total_dd_pct <= -max_loss_limit:
                return False, calendar_days, i, "Max Loss Limit"
                
            # Check Target
            gain_pct = (equity - start_balance) / start_balance * 100.0
            if gain_pct >= target_pct and calendar_days >= min_days:
                return True, calendar_days, i, "Passed"
                
        calendar_days = (exit_dates[-1] - challenge_start_date).days + 1
        return False, calendar_days, n_trades, "Out of time"

    passes = 0
    total_days_list = []
    fail_reasons = {'Daily Loss Limit': 0, 'Max Loss Limit': 0, 'Out of time': 0, 'P2: Insufficient Data': 0}
    
    for s_idx in start_indices:
        p1_pass, p1_days, p1_end_idx, p1_reason = sim_phase(s_idx, rules['phase1_target'], rules['daily_loss'], rules['max_loss'], rules['min_days'])
        
        if not p1_pass:
            fail_reasons[p1_reason] = fail_reasons.get(p1_reason, 0) + 1
            continue
            
        if p1_end_idx + 1 >= len(df):
            fail_reasons['P2: Insufficient Data'] += 1
            continue
            
        p2_pass, p2_days, p2_end_idx, p2_reason = sim_phase(p1_end_idx + 1, rules['phase2_target'], rules['daily_loss'], rules['max_loss'], rules['min_days'])
        
        if p2_pass:
            passes += 1
            total_days_list.append(p1_days + p2_days)
        else:
            fail_reasons[p2_reason] = fail_reasons.get(p2_reason, 0) + 1
            
    total_runs = len(start_indices)
    pass_rate = (passes / total_runs * 100.0) if total_runs > 0 else 0.0
    
    med_days = np.median(total_days_list) if passes > 0 else 0
    p25_days = np.percentile(total_days_list, 25) if passes > 0 else 0
    p75_days = np.percentile(total_days_list, 75) if passes > 0 else 0
    
    return {
        "runs": total_runs,
        "passes": passes,
        "pass_rate": pass_rate,
        "med_days": med_days,
        "p25_days": p25_days,
        "p75_days": p75_days,
        "fail_max_loss": fail_reasons.get('Max Loss Limit', 0),
        "fail_daily_loss": fail_reasons.get('Daily Loss Limit', 0),
        "fail_time": fail_reasons.get('Out of time', 0) + fail_reasons.get('P2: Insufficient Data', 0)
    }

def run_monte_carlo(pnl_pct_array, num_runs=100):
    """Vectorized Monte Carlo simulation with replacement."""
    if len(pnl_pct_array) == 0:
        return {}
        
    n_trades = len(pnl_pct_array)
    sims_pct = np.random.choice(pnl_pct_array, size=(num_runs, n_trades), replace=True) / 100.0
    
    ending_returns = sims_pct.sum(axis=1)
    
    equity_curves = np.cumsum(sims_pct, axis=1) + 1.0
    peaks = np.maximum.accumulate(equity_curves, axis=1)
    drawdowns = (equity_curves - peaks) / peaks
    max_dd_pcts = drawdowns.min(axis=1) * 100.0
    
    fail_10 = (max_dd_pcts <= -10.0).sum() / num_runs * 100.0
    fail_12 = (max_dd_pcts <= -12.0).sum() / num_runs * 100.0
    
    streaks = []
    for i in range(num_runs):
        is_loss = sims_pct[i] < 0
        s = pd.Series(is_loss)
        max_s = s.groupby((~s).cumsum()).sum().max() if s.sum() > 0 else 0
        streaks.append(max_s)
        
    return {
        "mc_median_return": np.median(ending_returns) * 100.0,
        "mc_p5_return": np.percentile(ending_returns, 5) * 100.0,
        "mc_p95_return": np.percentile(ending_returns, 95) * 100.0,
        "mc_fail_10_pct": fail_10,
        "mc_fail_12_pct": fail_12,
        "mc_expected_losing_streak": np.mean(streaks),
        "mc_worst_dd": np.percentile(max_dd_pcts, 5)
    }

def evaluate_equity_curve(trades, risk_pct, consec_loss_limit, max_trades_day):
    """Walks chronologically and enforces daily loss/trade limits."""
    df = trades.copy()
    if len(df) == 0:
        return pd.DataFrame(), {}
        
    df = df.sort_values('exit_time').reset_index(drop=True)
    
    accepted = []
    consec_losses = 0
    max_losing_streak = 0
    current_losing_streak = 0
    trades_today = 0
    daily_pnl = 0.0
    last_date = None
    halted_today = False
    
    for _, row in df.iterrows():
        t_date = row['entry_time'].date()
        
        if t_date != last_date:
            trades_today = 0
            daily_pnl = 0.0
            halted_today = False
            last_date = t_date
            
        if halted_today:
            continue
        if trades_today >= max_trades_day:
            continue
            
        pnl_R = row['pnl_R']
        pnl_pct = pnl_R * risk_pct
        
        if pnl_R < 0:
            consec_losses += 1
            current_losing_streak += 1
            if current_losing_streak > max_losing_streak:
                max_losing_streak = current_losing_streak
        else:
            consec_losses = 0
            current_losing_streak = 0
            
        accepted.append({
            'entry_time': row['entry_time'],
            'exit_time': row['exit_time'],
            'pnl_R': pnl_R,
            'pnl_pct': pnl_pct,
        })
        
        trades_today += 1
        daily_pnl += pnl_pct
        
        if consec_losses >= consec_loss_limit:
            halted_today = True
            
    res_df = pd.DataFrame(accepted)
    if len(res_df) == 0:
        return res_df, {}
        
    total_trades = len(res_df)
    win_rate = (res_df['pnl_R'] > 0).mean() * 100
    gross_win = res_df.loc[res_df['pnl_R'] > 0, 'pnl_R'].sum()
    gross_loss = abs(res_df.loc[res_df['pnl_R'] < 0, 'pnl_R'].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else 999.0
    net_R = res_df['pnl_R'].sum()
    avg_R = res_df['pnl_R'].mean()
    
    equity = res_df['pnl_R'].cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd_R = dd.min()
    
    days = (res_df['exit_time'].max() - res_df['exit_time'].min()).days
    months = max(1, days / 30.4)
    trades_per_month = total_trades / months
    
    metrics = {
        'total_trades': total_trades,
        'trades_month': trades_per_month,
        'win_rate': win_rate,
        'profit_factor': pf,
        'net_R': net_R,
        'avg_R': avg_R,
        'max_dd_R': max_dd_R,
        'losing_streak': max_losing_streak
    }
    return res_df, metrics

# -----------------------------------------------------------------------------
# 5. PARAMETER SWEEP
# -----------------------------------------------------------------------------

def sweep_cached_block(symbol, direction, rr, mc_runs):
    cache_path = f"cache/{symbol}_{direction}_RR{rr}_oof.csv"
    if not os.path.exists(cache_path):
        print(f"[ERROR] Cache not found: {cache_path}")
        return []
        
    print(f"\n[SWEEP] Loading {cache_path}")
    df_cache = pd.read_csv(cache_path)
    df_cache['entry_time'] = pd.to_datetime(df_cache['entry_time'])
    df_cache['exit_time'] = pd.to_datetime(df_cache['exit_time'])
    
    # Filter valid OOF predictions
    valid_mask = ~df_cache['oof_exp'].isna() & ~df_cache['oof_fake'].isna() & ~df_cache['oof_cont'].isna()
    df_valid = df_cache[valid_mask].copy()
    
    if len(df_valid) == 0:
        return []
        
    results = []
    
    # Sweep over weight sets
    for (w_exp, w_cont, w_fake) in WEIGHT_SETS:
        df_valid['raw_score'] = (df_valid['oof_exp'] * w_exp) + (df_valid['oof_cont'] * w_cont) - (df_valid['oof_fake'] * w_fake)
        df_valid['score_norm'] = df_valid['raw_score'].rank(pct=True)
        
        for th in THRESHOLDS:
            df_filtered = df_valid[df_valid['score_norm'] >= th]
            if len(df_filtered) < 10:
                continue
                
            for risk in RISKS:
                for closs in CONSEC_LOSSES:
                    for mday in MAX_TRADES_DAY:
                        res_df, m = evaluate_equity_curve(df_filtered, risk, closs, mday)
                        if len(res_df) == 0:
                            continue
                            
                        # Prop Sim
                        fn_res = run_rolling_prop_sim(res_df, PROP_FIRMS['FundedNext'])
                        fp_res = run_rolling_prop_sim(res_df, PROP_FIRMS['FundingPips'])
                        ee_res = run_rolling_prop_sim(res_df, PROP_FIRMS['EquityEdge'])
                        
                        # Monte Carlo
                        mc_res = run_monte_carlo(res_df['pnl_pct'].values, num_runs=mc_runs)
                        
                        results.append({
                            "Symbol": symbol,
                            "Direction": direction,
                            "RR": rr,
                            "Weight_Exp": w_exp,
                            "Weight_Cont": w_cont,
                            "Weight_Fake": w_fake,
                            "Threshold": th,
                            "Risk": risk,
                            "MaxLosses": closs,
                            "MaxTradesDay": mday,
                            "Trades": m['total_trades'],
                            "TradesMo": m['trades_month'],
                            "WinRate": m['win_rate'],
                            "PF": m['profit_factor'],
                            "NetR": m['net_R'],
                            "AvgR": m['avg_R'],
                            "MaxDD_R": m['max_dd_R'],
                            "LosingStreak": m['losing_streak'],
                            
                            "FN_PassRate": fn_res['pass_rate'], "FN_MedDays": fn_res['med_days'], "FN_P25": fn_res['p25_days'], "FN_P75": fn_res['p75_days'],
                            "FN_FailMax": fn_res['fail_max_loss'], "FN_FailDaily": fn_res['fail_daily_loss'], "FN_FailTime": fn_res['fail_time'], "FN_Runs": fn_res['runs'],
                            
                            "FP_PassRate": fp_res['pass_rate'], "FP_MedDays": fp_res['med_days'], "FP_P25": fp_res['p25_days'], "FP_P75": fp_res['p75_days'],
                            "FP_FailMax": fp_res['fail_max_loss'], "FP_FailDaily": fp_res['fail_daily_loss'], "FP_FailTime": fp_res['fail_time'], "FP_Runs": fp_res['runs'],
                            
                            "EE_PassRate": ee_res['pass_rate'], "EE_MedDays": ee_res['med_days'], "EE_P25": ee_res['p25_days'], "EE_P75": ee_res['p75_days'],
                            "EE_FailMax": ee_res['fail_max_loss'], "EE_FailDaily": ee_res['fail_daily_loss'], "EE_FailTime": ee_res['fail_time'], "EE_Runs": ee_res['runs'],
                            
                            "MC_Median": mc_res.get('mc_median_return', 0),
                            "MC_P5": mc_res.get('mc_p5_return', 0),
                            "MC_P95": mc_res.get('mc_p95_return', 0),
                            "MC_Fail10": mc_res.get('mc_fail_10_pct', 0),
                            "MC_Fail12": mc_res.get('mc_fail_12_pct', 0),
                            "MC_ExpStreak": mc_res.get('mc_expected_losing_streak', 0),
                            "MC_WorstDD": mc_res.get('mc_worst_dd', 0)
                        })
                        
    print(f"  -> Generated {len(results)} valid configurations.")
    return results

def get_trades_for_config(row):
    cache_path = f"cache/{row['Symbol']}_{row['Direction']}_RR{row['RR']}_oof.csv"
    if not os.path.exists(cache_path):
        return pd.DataFrame()
        
    df_cache = pd.read_csv(cache_path)
    df_cache['entry_time'] = pd.to_datetime(df_cache['entry_time'])
    df_cache['exit_time'] = pd.to_datetime(df_cache['exit_time'])
    valid_mask = ~df_cache['oof_exp'].isna() & ~df_cache['oof_fake'].isna() & ~df_cache['oof_cont'].isna()
    df_valid = df_cache[valid_mask].copy()
    
    df_valid['raw_score'] = (df_valid['oof_exp'] * row['Weight_Exp']) + (df_valid['oof_cont'] * row['Weight_Cont']) - (df_valid['oof_fake'] * row['Weight_Fake'])
    df_valid['score_norm'] = df_valid['raw_score'].rank(pct=True)
    df_filtered = df_valid[df_valid['score_norm'] >= row['Threshold']]
    
    res_df, _ = evaluate_equity_curve(df_filtered, row['Risk'], row['MaxLosses'], row['MaxTradesDay'])
    return res_df

# -----------------------------------------------------------------------------
# 6. MAIN SKELETON
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs("cache", exist_ok=True)
    
    print("=" * 60)
    print(" V2 PROP CHALLENGE OPTIMIZER (RESEARCH ONLY)")
    print("=" * 60)
    
    xgb_n_estimators = 100 if args.mode == "search" else 300
    xgb_max_depth = 3 if args.mode == "search" else 4
    mc_runs = 100 if args.mode == "search" else 1000
    
    BLOCKS = [
        ("BTCUSD", "BOTH", 1.2), ("BTCUSD", "BOTH", 1.5), ("BTCUSD", "BOTH", 1.8), ("BTCUSD", "BOTH", 2.0),
        ("BTCUSD", "LONG", 1.2), ("BTCUSD", "LONG", 1.5), ("BTCUSD", "LONG", 1.8), ("BTCUSD", "LONG", 2.0),
        ("XAUUSD", "LONG", 1.2), ("XAUUSD", "LONG", 1.5), ("XAUUSD", "LONG", 1.8), ("XAUUSD", "LONG", 2.0),
        ("XAUUSD", "BOTH", 1.2), ("XAUUSD", "BOTH", 1.5), ("XAUUSD", "BOTH", 1.8), ("XAUUSD", "BOTH", 2.0),
    ]
    
    progress_file = "optimizer_progress.json"
    completed_blocks = []
    if args.resume and os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                completed_blocks = json.load(f)
        except Exception:
            pass
            
    all_results = []
    if args.resume and os.path.exists("optimizer_results.csv"):
        try:
            all_results = pd.read_csv("optimizer_results.csv").to_dict('records')
            print(f"[RESUME] Loaded {len(all_results)} existing configurations.")
        except Exception:
            pass
            
    for sym, dir_, rr in BLOCKS:
        block_id = f"{sym}_{dir_}_RR{rr}"
        if block_id in completed_blocks:
            print(f"[SKIP] Block {block_id} already completed.")
            continue
            
        print("\n" + "=" * 60)
        print(f" BLOCK START: {sym} | {dir_} | RR {rr}")
        print("=" * 60)
        
        build_oof_cache(sym, dir_, rr, xgb_n_estimators, xgb_max_depth)
        res = sweep_cached_block(sym, dir_, rr, mc_runs)
        all_results.extend(res)
        
        completed_blocks.append(block_id)
        with open(progress_file, "w") as f:
            json.dump(completed_blocks, f)
            
        if len(all_results) > 0:
            df_res = pd.DataFrame(all_results)
            df_res['AvgPassRate'] = df_res[['FN_PassRate', 'FP_PassRate', 'EE_PassRate']].mean(axis=1)
            df_res['Candidate'] = (df_res['PF'] >= 1.40) & (df_res['MaxDD_R'] >= -10) & (df_res['TradesMo'] >= 5) & (df_res['MC_Fail10'] <= 25.0) & (df_res['AvgPassRate'] >= 30.0)
            df_res.to_csv("optimizer_results.csv", index=False)
            
    if len(all_results) == 0:
        print("\n[CHECKPOINT 6] Sweep complete. No results found.")
        sys.exit(0)
        
    df_res = pd.DataFrame(all_results)
    df_res['AvgPassRate'] = df_res[['FN_PassRate', 'FP_PassRate', 'EE_PassRate']].mean(axis=1)
    df_res['AvgMedDays'] = df_res[['FN_MedDays', 'FP_MedDays', 'EE_MedDays']].mean(axis=1)
    df_res['Candidate'] = (df_res['PF'] >= 1.40) & (df_res['MaxDD_R'] >= -10) & (df_res['TradesMo'] >= 5) & (df_res['MC_Fail10'] <= 25.0) & (df_res['AvgPassRate'] >= 30.0)
    
    # Debug Summary
    print("\n" + "=" * 60)
    print(" [CHECKPOINT 6] FINAL SUMMARY & ROLLING PROP AUDIT")
    print("=" * 60)
    print(f" Total blocks completed : {len(completed_blocks)}")
    print(f" Total configs generated: {len(df_res)}")
    print(f" Total candidate configs: {df_res['Candidate'].sum()}")
    
    sort_cols = ["AvgPassRate", "AvgMedDays", "MC_Fail10", "PF"]
    sort_asc  = [False, True, True, False]
    
    # Best BTC
    df_btc = df_res[(df_res['Symbol'] == 'BTCUSD') & df_res['Candidate']]
    if len(df_btc) > 0:
        b_btc = df_btc.sort_values(by=sort_cols, ascending=sort_asc).iloc[0]
        print(f" Best BTC Config        : {b_btc['Direction']} | RR {b_btc['RR']} | PF {b_btc['PF']:.2f} | DD {b_btc['MaxDD_R']:.2f}R | AvgPass: {b_btc['AvgPassRate']:.1f}% | MC Fail: {b_btc['MC_Fail10']:.1f}%")
    
    # Best XAU
    df_xau = df_res[(df_res['Symbol'] == 'XAUUSD') & df_res['Candidate']]
    if len(df_xau) > 0:
        b_xau = df_xau.sort_values(by=sort_cols, ascending=sort_asc).iloc[0]
        print(f" Best XAU Config        : {b_xau['Direction']} | RR {b_xau['RR']} | PF {b_xau['PF']:.2f} | DD {b_xau['MaxDD_R']:.2f}R | AvgPass: {b_xau['AvgPassRate']:.1f}% | MC Fail: {b_xau['MC_Fail10']:.1f}%")
        
    print("\n" + "=" * 60)
    print(" TOP 20 CANDIDATES OVERALL")
    print("=" * 60)
    df_cands = df_res[df_res['Candidate']]
    if len(df_cands) > 0:
        df_sorted = df_cands.sort_values(by=sort_cols, ascending=sort_asc)
        for i, (_, row) in enumerate(df_sorted.head(20).iterrows(), 1):
            print(f"{i:2d}. {row['Symbol']} {row['Direction']} RR{row['RR']} | Th: {row['Threshold']:.2f} | "
                  f"W({row['Weight_Exp']}, {row['Weight_Cont']}, {row['Weight_Fake']}) | Rsk: {row['Risk']}% | "
                  f"{row['MaxLosses']}L/{row['MaxTradesDay']}T")
            print(f"    PF: {row['PF']:.2f} | MaxDD: {row['MaxDD_R']:.2f}R | Pass: {row['AvgPassRate']:.1f}% (Med {row['AvgMedDays']:.1f}d) | MC Fail10%: {row['MC_Fail10']:.1f}% (WorstDD: {row['MC_WorstDD']:.1f}%)")
    else:
        print(" No candidate configs found overall.")
        
    print("\n" + "=" * 60)
    print(" WORST 10 CONFIGS BY PF (Verify Failures)")
    print("=" * 60)
    df_worst = df_res.sort_values(by="PF", ascending=True).head(10)
    for i, (_, row) in enumerate(df_worst.iterrows(), 1):
        print(f"{i:2d}. {row['Symbol']} {row['Direction']} RR{row['RR']} | Th: {row['Threshold']:.2f} | Rsk: {row['Risk']}%")
        print(f"    PF: {row['PF']:.2f} | AvgPass: {row['AvgPassRate']:.1f}% | FN Time Fails: {row['FN_FailTime']}")
        
    print("\n[CHECKPOINT 6] Sweep complete.")
    
    # -------------------------------------------------------------------------
    # CHECKPOINT 7: PORTFOLIO OPTIMIZATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" [CHECKPOINT 7] PORTFOLIO OPTIMIZATION (50x50)")
    print("=" * 60)
    
    df_btc_all = df_res[df_res['Symbol'] == 'BTCUSD']
    df_xau_all = df_res[df_res['Symbol'] == 'XAUUSD']
    
    if len(df_btc_all) == 0 or len(df_xau_all) == 0:
        print(" Not enough data in both BTC and XAU to form portfolios.")
        sys.exit(0)
        
    df_btc_cands = df_btc_all.sort_values(by=sort_cols, ascending=sort_asc).head(50)
    df_xau_cands = df_xau_all.sort_values(by=sort_cols, ascending=sort_asc).head(50)
    
    btc_trades = []
    print("  -> Extracting BTC trades...")
    for _, row in df_btc_cands.iterrows():
        btc_trades.append((row, get_trades_for_config(row)))
        
    xau_trades = []
    print("  -> Extracting XAU trades...")
    for _, row in df_xau_cands.iterrows():
        xau_trades.append((row, get_trades_for_config(row)))
        
    portfolios = []
    total_pairs = len(btc_trades) * len(xau_trades)
    print(f"  -> Simulating {total_pairs} Portfolio Combinations...")
    
    for (b_row, b_df) in btc_trades:
        for (x_row, x_df) in xau_trades:
            if len(b_df) == 0 or len(x_df) == 0:
                continue
                
            port_df = pd.concat([b_df, x_df]).sort_values('exit_time').reset_index(drop=True)
            total_trades = len(port_df)
            if total_trades < 60:
                continue
                
            gross_win = port_df.loc[port_df['pnl_pct'] > 0, 'pnl_pct'].sum()
            gross_loss = abs(port_df.loc[port_df['pnl_pct'] < 0, 'pnl_pct'].sum())
            pf = gross_win / gross_loss if gross_loss > 0 else 999.0
            
            days = (port_df['exit_time'].max() - port_df['exit_time'].min()).days
            months = max(1, days / 30.4)
            trades_mo = total_trades / months
            
            fn_res = run_rolling_prop_sim(port_df, PROP_FIRMS['FundedNext'])
            fp_res = run_rolling_prop_sim(port_df, PROP_FIRMS['FundingPips'])
            ee_res = run_rolling_prop_sim(port_df, PROP_FIRMS['EquityEdge'])
            
            avg_pass = np.mean([fn_res['pass_rate'], fp_res['pass_rate'], ee_res['pass_rate']])
            avg_med_days = np.mean([fn_res['med_days'], fp_res['med_days'], ee_res['med_days']])
            
            mc_res = run_monte_carlo(port_df['pnl_pct'].values, num_runs=mc_runs)
            
            portfolios.append({
                "BTC_Config": f"{b_row['Direction']} RR{b_row['RR']} Th{b_row['Threshold']:.2f} W({b_row['Weight_Exp']}, {b_row['Weight_Cont']}, {b_row['Weight_Fake']}) Rsk{b_row['Risk']}% Lmt{b_row['MaxLosses']}L/{b_row['MaxTradesDay']}T",
                "XAU_Config": f"{x_row['Direction']} RR{x_row['RR']} Th{x_row['Threshold']:.2f} W({x_row['Weight_Exp']}, {x_row['Weight_Cont']}, {x_row['Weight_Fake']}) Rsk{x_row['Risk']}% Lmt{x_row['MaxLosses']}L/{x_row['MaxTradesDay']}T",
                "PF": pf,
                "TradesMo": trades_mo,
                "AvgPassRate": avg_pass,
                "AvgMedDays": avg_med_days,
                "MC_Fail10": mc_res.get('mc_fail_10_pct', 100.0),
                "MC_WorstDD": mc_res.get('mc_worst_dd', -100.0)
            })
            
    if len(portfolios) > 0:
        df_port = pd.DataFrame(portfolios)
        df_port.to_csv("portfolio_results.csv", index=False)
        
        print("\n" + "=" * 60)
        print(" TOP 10 PORTFOLIOS (BTC + XAU)")
        print("=" * 60)
        sort_cols_port = ["AvgPassRate", "MC_Fail10", "PF", "MC_WorstDD"]
        sort_asc_port = [False, True, False, False]
        df_port_sorted = df_port.sort_values(by=sort_cols_port, ascending=sort_asc_port)
        
        for i, (_, row) in enumerate(df_port_sorted.head(10).iterrows(), 1):
            print(f"{i:2d}. BTC: {row['BTC_Config']}\n    XAU: {row['XAU_Config']}")
            print(f"    PF: {row['PF']:.2f} | T/Mo: {row['TradesMo']:.1f} | AvgPass: {row['AvgPassRate']:.1f}% (Med {row['AvgMedDays']:.1f}d) | MC Fail10%: {row['MC_Fail10']:.1f}% (WorstDD: {row['MC_WorstDD']:.1f}%)\n")
    else:
        print("  -> No valid portfolios generated.")
        
    print("\n[CHECKPOINT 7] Portfolio Sweep complete.")
    sys.exit(0)

if __name__ == "__main__":
    main()
