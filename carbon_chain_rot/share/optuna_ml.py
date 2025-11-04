#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Note: requires optuna installed in the runtime environment
try:
    import optuna
except Exception as e:
    optuna = None


# -----------------------
# par helpers
# -----------------------

def load_par(path: str) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if not path or not os.path.isfile(path):
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def get_par_path(default_base: str) -> str:
    # priority: env PAR_FILE -> <base>/par -> <base>/par_all
    env_path = os.environ.get("PAR_FILE")
    if env_path and os.path.isfile(env_path):
        return env_path
    cand = os.path.join(default_base, "par")
    if os.path.isfile(cand):
        return cand
    cand2 = os.path.join(default_base, "par_all")
    if os.path.isfile(cand2):
        return cand2
    return ""


def as_list(val: str) -> List[str]:
    if val is None:
        return []
    return [x.strip() for x in str(val).split(',') if x.strip()]


def parse_fixed_features(val: str) -> Dict[str, float]:
    # format: name1=1.0,name2=2.5
    out: Dict[str, float] = {}
    if not val:
        return out
    for chunk in val.split(','):
        if not chunk.strip():
            continue
        if '=' not in chunk:
            continue
        k, v = chunk.split('=', 1)
        try:
            out[k.strip()] = float(v.strip())
        except Exception:
            continue
    return out


# -----------------------
# simple linear regressor
# -----------------------

class LinearRegressor:
    def __init__(self):
        self.coef_: np.ndarray = None
        self.intercept_: float = 0.0
        self.columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.columns_ = list(X.columns)
        Xm = X.values.astype(float)
        ym = y.values.astype(float).reshape(-1, 1)
        # add bias
        Xb = np.hstack([Xm, np.ones((Xm.shape[0], 1))])
        beta, *_ = np.linalg.lstsq(Xb, ym, rcond=None)
        self.coef_ = beta[:-1, 0]
        self.intercept_ = float(beta[-1, 0])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xm = X[self.columns_].values.astype(float)
        return Xm.dot(self.coef_) + self.intercept_


# -----------------------
# core objective and runner
# -----------------------

def build_objective(model: LinearRegressor,
                    feature_cols: List[str],
                    fixed_vals: Dict[str, float],
                    base_vals: Dict[str, float],
                    angle_range: Tuple[float, float],
                    dist_range: Tuple[float, float],
                    direction: str = 'min'):
    def objective(trial):
        angle_min, angle_max = angle_range
        dist_min, dist_max = dist_range
        angle = trial.suggest_float('angle', angle_min, angle_max)
        dist = trial.suggest_float('distance', dist_min, dist_max)

        # compose one-row DataFrame for prediction
        row = {}
        for col in feature_cols:
            if col.lower() in ('angle', 'theta'):
                row[col] = angle
            elif col.lower() in ('distance', 'dist', 'r'):
                row[col] = dist
            elif col in fixed_vals:
                row[col] = fixed_vals[col]
            else:
                # fallback to base mean
                row[col] = base_vals.get(col, 0.0)
        xdf = pd.DataFrame([row], columns=feature_cols)
        pred = float(model.predict(xdf)[0])
        return pred if direction == 'min' else -pred
    return objective


def main():
    parser = argparse.ArgumentParser(description='Optuna-based optimizer for angle/distance using par control')
    parser.add_argument('--par', default=None, help='Path to par file (default: auto-detect)')
    parser.add_argument('--done', default=None, help='Path to done CSV (override par DONE_CSV)')
    parser.add_argument('--trials', type=int, default=None, help='Number of Optuna trials (override par)')
    parser.add_argument('--direction', choices=['min','max'], default=None, help='Optimization direction (default from par or min)')
    parser.add_argument('--out', default=None, help='Write best params to JSON file')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(__file__))  # carbon_chain_rot
    par_path = args.par or get_par_path(base_dir)
    cfg = load_par(par_path)

    # data locations
    done_csv = args.done or cfg.get('DONE_CSV')
    if not done_csv or not os.path.isfile(done_csv):
        print('ERROR: DONE_CSV not provided or file missing. Set in par or via --done', file=sys.stderr)
        sys.exit(2)

    # columns and filters
    target_col = cfg.get('TARGET_COLUMN', 'result')
    feat_cols = as_list(cfg.get('FEATURE_COLUMNS'))
    filter_done = cfg.get('FILTER_DONE')

    # ranges and trials
    try:
        angle_min = float(cfg.get('ANGLE_MIN'))
        angle_max = float(cfg.get('ANGLE_MAX'))
        dist_min = float(cfg.get('DIST_MIN'))
        dist_max = float(cfg.get('DIST_MAX'))
    except Exception:
        print('ERROR: ANGLE_MIN/ANGLE_MAX and DIST_MIN/DIST_MAX must be set in par', file=sys.stderr)
        sys.exit(2)

    n_trials = args.trials if args.trials is not None else int(cfg.get('OPTUNA_N_TRIALS', 50))
    direction = args.direction or cfg.get('OPTUNA_DIRECTION', 'min').lower()
    if direction not in ('min','max'):
        direction = 'min'

    fixed_map = parse_fixed_features(cfg.get('FIXED_FEATURES'))  # optional

    # read data
    df = pd.read_csv(done_csv)
    if filter_done:
        try:
            df = df.query(filter_done)
        except Exception:
            pass

    # infer feature columns if not provided
    if not feat_cols:
        feat_cols = [c for c in df.columns if c != target_col]

    # check required columns
    missing = [c for c in feat_cols + [target_col] if c not in df.columns]
    if missing:
        print(f'ERROR: missing columns in DONE_CSV: {missing}', file=sys.stderr)
        sys.exit(2)

    # fit simple linear regressor as surrogate
    X = df[feat_cols]
    y = df[target_col]
    model = LinearRegressor().fit(X, y)

    # base means for non-optimized features
    base_vals = {c: float(X[c].mean()) for c in feat_cols}

    # require optuna
    if optuna is None:
        print('ERROR: optuna is not installed. Please `pip install optuna` and retry.', file=sys.stderr)
        sys.exit(2)

    sampler = optuna.samplers.TPESampler(seed=int(cfg.get('OPTUNA_SEED', 42)))
    study = optuna.create_study(direction='minimize' if direction=='min' else 'maximize', sampler=sampler)
    objective = build_objective(model, feat_cols, fixed_map, base_vals,
                                (angle_min, angle_max), (dist_min, dist_max), direction)
    study.optimize(objective, n_trials=n_trials)

    best = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'direction': direction,
        'par_path': par_path,
        'done_csv': done_csv,
    }

    out_text = json.dumps(best, indent=2)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(out_text + '\n')
    else:
        print(out_text)


if __name__ == '__main__':
    main()

