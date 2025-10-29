from __future__ import annotations

import os
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def ensure_dirs(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
	return pd.read_csv(csv_path)


def split_data(
	df: pd.DataFrame,
	target_col: str,
	test_size: float = 0.2,
	random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	X = df.drop(columns=[target_col])
	y = df[target_col]
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state, stratify=y
	)
	return X.values, y.values, X_train.values, X_test.values, y_train.values, y_test.values


def save_artifacts(model: Any, metrics_df: pd.DataFrame, out_dir: str, model_name: str = "best_model.pkl") -> Dict[str, str]:
	ensure_dirs(out_dir)
	model_path = os.path.join(out_dir, model_name)
	metrics_path = os.path.join(out_dir, "metrics.csv")
	joblib.dump(model, model_path)
	metrics_df.to_csv(metrics_path, index=False)
	return {"model": model_path, "metrics": metrics_path}


def plot_roc_curves(results: Dict[str, Dict[str, Any]], y_test: np.ndarray) -> None:
	plt.figure(figsize=(10, 7))
	for name, res in results.items():
		proba = res.get("proba")
		if proba is None:
			continue
		fpr, tpr, _ = roc_curve(y_test, proba)
		plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
	plt.plot([0, 1], [0, 1], "k--", alpha=0.6)
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("ROC Curves")
	plt.legend()
	plt.tight_layout()
	plt.show()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[np.ndarray] = None, normalize: bool = True) -> None:
	cm = confusion_matrix(y_true, y_pred, labels=labels)
	norm = "true" if normalize else None
	ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_true, y_pred, normalize=norm)).plot(cmap="Blues")
	plt.title("Confusion Matrix")
	plt.tight_layout()
	plt.show()
