from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb


def get_models(random_state: int = 42) -> Dict[str, Any]:
	"""Return a dictionary of initialized models for classification."""
	models: Dict[str, Any] = {
		"Logistic Regression": LogisticRegression(random_state=random_state, max_iter=1000),
		"Random Forest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1),
		"XGBoost": xgb.XGBClassifier(
			random_state=random_state,
			verbosity=0,
			use_label_encoder=False,
			eval_metric="logloss",
			n_estimators=500,
			n_jobs=-1,
		),
		"LightGBM": lgb.LGBMClassifier(random_state=random_state, verbose=-1, n_estimators=500),
		"SVM": SVC(random_state=random_state, probability=True),
	}
	return models


def train_and_eval(
	models: Dict[str, Any],
	X_train: np.ndarray,
	y_train: np.ndarray,
	X_test: np.ndarray,
	y_test: np.ndarray,
	cv_folds: int = 5,
) -> Dict[str, Dict[str, Any]]:
	"""Train models and return metrics, predictions, and probabilities.

	Returns: dict mapping model_name to metrics and artifacts
	{"accuracy", "cv_mean", "cv_std", "pred", "proba", "model"}
	"""
	results: Dict[str, Dict[str, Any]] = {}
	for name, model in models.items():
		# Cross-validation on train set
		cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="accuracy", n_jobs=-1)

		# Fit and evaluate on held-out set
		model.fit(X_train, y_train)
		pred = model.predict(X_test)
		accuracy = accuracy_score(y_test, pred)
		proba = None
		if hasattr(model, "predict_proba"):
			try:
				proba = model.predict_proba(X_test)[:, 1]
			except Exception:
				proba = None

		results[name] = {
			"accuracy": float(accuracy),
			"cv_mean": float(cv_scores.mean()),
			"cv_std": float(cv_scores.std()),
			"pred": pred,
			"proba": proba,
			"model": model,
		}
	return results
