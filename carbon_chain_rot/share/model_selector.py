#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Any, Dict, Tuple

from .bayes_opt import make_candidates as _bayes_make
from .optuna_ml import run_optuna as _run_optuna


def _load_par(path: str) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    if not path or not os.path.isfile(path):
        return cfg
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip()
    return cfg


def _get_par_path(base_dir: str) -> str:
    # priority: env PAR_FILE -> <base>/par -> <base>/par_all
    env_path = os.environ.get('PAR_FILE')
    if env_path and os.path.isfile(env_path):
        return env_path
    cand = os.path.join(base_dir, 'par')
    if os.path.isfile(cand):
        return cand
    cand2 = os.path.join(base_dir, 'par_all')
    if os.path.isfile(cand2):
        return cand2
    return ''


def select_and_run(all_env: Dict[str, Any],
                   df_with_energy,
                   df_without_energy):
    """
    Select model by par: MODEL=bayes|optuna and run accordingly.

    Returns:
      - If MODEL=bayes: ("bayes", {"candidates": list, "calc_dir": str})
      - If MODEL=optuna: ("optuna", best_dict)  # from optuna_ml.run_optuna
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))  # carbon_chain_rot
    par_path = all_env.get('PAR_FILE') if isinstance(all_env, dict) else None
    if not par_path:
        par_path = _get_par_path(base_dir)
    cfg = _load_par(par_path)
    model = cfg.get('MODEL', 'bayes').strip().lower()

    if model == 'optuna':
        best = _run_optuna(par_path=par_path)
        return 'optuna', best
    # default: bayes
    candidates, calc_dir = _bayes_make(all_env, df_with_energy, df_without_energy)
    return 'bayes', { 'candidates': candidates, 'calc_dir': calc_dir }

