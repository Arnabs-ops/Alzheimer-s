"""
Hyperparameter tuning utilities (sklearn Grid/Random search, optional Optuna)

This module defines parameter spaces and helper functions to perform
hyperparameter optimization for common classifiers used in this project.

Important: This file only provides functions; it does not execute any tuning
by itself. Import and call from a script/notebook when you are ready to run
experiments.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import numpy as np

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score


def get_param_space(model_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Return (grid_params, random_params) for a given model name.

    grid_params is a smaller, more exhaustive set (for GridSearchCV).
    random_params is a wider distribution/range set (for RandomizedSearchCV).

    If a model is unknown, returns (None, None).
    """
    name = model_name.lower()

    if "random forest" in name:
        grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 6, 8, 12],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5, 0.7],
        }
        random = {
            "n_estimators": [50, 75, 100, 150, 200, 300],
            "max_depth": [None, 4, 6, 8, 10, 12],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        }
        return grid, random

    if "extra trees" in name:
        grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 6, 8, 12],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.5, 0.7],
        }
        random = {
            "n_estimators": [50, 75, 100, 150, 200, 300],
            "max_depth": [None, 4, 6, 8, 10, 12],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        }
        return grid, random

    if "xgboost" in name or "xgb" in name:
        grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 6],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_alpha": [0.0, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0],
        }
        random = {
            "n_estimators": np.arange(80, 401, 40).tolist(),
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
            "reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0],
        }
        return grid, random

    if "lightgbm" in name or "lgb" in name:
        grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [-1, 4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_alpha": [0.0, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0],
        }
        random = {
            "n_estimators": np.arange(80, 401, 40).tolist(),
            "max_depth": [-1, 3, 4, 5, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
            "reg_lambda": [0.1, 0.5, 1.0, 2.0, 5.0],
        }
        return grid, random

    if "svm" in name:
        grid = {
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto", 0.01, 0.1],
            "kernel": ["rbf", "poly", "sigmoid"],
        }
        random = {
            "C": [0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
            "gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1],
            "kernel": ["rbf", "poly", "sigmoid"],
        }
        return grid, random

    if "logistic regression" in name:
        grid = {
            "C": [0.01, 0.1, 1.0],
            "penalty": ["l1", "l2"],
        }
        random = {
            "C": [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            "penalty": ["l1", "l2"],
        }
        return grid, random

    if "mlp" in name or "neural network" in name:
        grid = {
            "hidden_layer_sizes": [(50,), (50, 25)],
            "alpha": [1e-3, 1e-2, 1e-1],
            "learning_rate": ["constant", "adaptive"],
        }
        random = {
            "hidden_layer_sizes": [(30,), (50,), (50, 25), (64, 32)],
            "alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            "learning_rate": ["constant", "adaptive"],
        }
        return grid, random

    return None, None


def run_grid_search(model, model_name: str, X: np.ndarray, y: np.ndarray,
                    cv_folds: int = 3, n_jobs: int = 1, scoring: str = "accuracy") -> Tuple[Any, Dict[str, Any]]:
    """Execute GridSearchCV on the provided model using its grid space."""
    grid, _ = get_param_space(model_name)
    if not grid:
        return model, {"message": "No grid defined; returned original model."}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    gs = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=0,
    )
    gs.fit(X, y)
    return gs.best_estimator_, {
        "best_params": gs.best_params_,
        "best_score": float(gs.best_score_),
    }


def run_random_search(model, model_name: str, X: np.ndarray, y: np.ndarray,
                      n_iter: int = 25, cv_folds: int = 3, n_jobs: int = 1,
                      scoring: str = "accuracy", random_state: int = 42) -> Tuple[Any, Dict[str, Any]]:
    """Execute RandomizedSearchCV on the provided model using its random space."""
    _, random_space = get_param_space(model_name)
    if not random_space:
        return model, {"message": "No random space defined; returned original model."}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=random_space,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=0,
    )
    rs.fit(X, y)
    return rs.best_estimator_, {
        "best_params": rs.best_params_,
        "best_score": float(rs.best_score_),
    }


def run_optuna_tuning(model_builder, model_name: str, X: np.ndarray, y: np.ndarray,
                      n_trials: int = 30, timeout: Optional[int] = 600,
                      cv_folds: int = 3, scoring: str = "accuracy",
                      random_state: int = 42, n_jobs: int = -1,
                      enable_pruning: bool = True, verbose: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """Optional Optuna tuning with pruning and timeout. Requires `optuna`.

    model_builder: a function(trial) -> estimator with trial-suggested params
    """
    try:
        import optuna
        from optuna.exceptions import TrialPruned
    except Exception:
        return None, {"message": "Optuna not installed; skipping."}

    def objective(trial: "optuna.Trial") -> float:
        estimator = model_builder(trial)

        # Some sklearn estimators support n_jobs; set when available
        if hasattr(estimator, "n_jobs") and n_jobs is not None:
            try:
                setattr(estimator, "n_jobs", n_jobs)
            except Exception:
                pass

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores: list[float] = []
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
            estimator.fit(X[tr_idx], y[tr_idx])
            preds = estimator.predict(X[va_idx])
            score = accuracy_score(y[va_idx], preds)
            scores.append(score)

            # Report intermediate value for pruning
            if enable_pruning:
                trial.report(float(np.mean(scores)), step=fold_idx)
                if trial.should_prune():
                    raise TrialPruned()

        mean_score = float(np.mean(scores))
        if verbose:
            print(f"[Optuna] {model_name} trial finished: CV mean={mean_score:.4f}")
        return mean_score

    pruner = optuna.pruners.MedianPruner() if enable_pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    # Build the best estimator using the found params
    def build_from_best(trial_values: Dict[str, Any]):
        class DummyTrial:
            def __init__(self, params: Dict[str, Any]):
                self.params = params
            def suggest_float(self, name, low, high, log=False):
                return float(self.params[name])
            def suggest_int(self, name, low, high, step=1):
                return int(self.params[name])
            def suggest_categorical(self, name, choices):
                return self.params[name]
        return model_builder(DummyTrial(study.best_params))

    best_estimator = build_from_best(study.best_params)
    return best_estimator, {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
        "pruned": int(sum(t.state.name == "PRUNED" for t in study.trials)),
        "timeout_sec": timeout,
    }


