"""
BoW TF-IDF text baseline method.

Trains a TF-IDF vectorizer + LogisticRegressionCV classifier on text
(e.g. chain-of-thought prefixes) to predict binary labels.
"""

import json
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score

from .base import BaseMethod


@dataclass
class BowTfidfConfig:
    max_features: int = 5000
    sublinear_tf: bool = True
    cv: int = 5
    max_iter: int = 5000
    seed: int = 42
    text_key: str = "prefix_text"
    label_key: str = "label"
    positive_label: str = "yes"
    class_weight: Optional[str] = None


class BoWTfidf(BaseMethod):
    """
    BoW TF-IDF (full prefix) text baseline.

    Trains a TfidfVectorizer + LogisticRegressionCV on full prefix text
    to predict binary labels (yes/no).

    Usage:
        method = BoWTfidf()
        method.set_task(task)
        method.train(train_entries)   # list of dicts with prefix_text + label
        method.infer(eval_entries)    # list of dicts with prefix_text + label
        method._output.mark_success()
    """

    def __init__(
        self,
        max_features: int = 5000,
        sublinear_tf: bool = True,
        cv: int = 5,
        max_iter: int = 5000,
        seed: int = 42,
        text_key: str = "prefix_text",
        label_key: str = "label",
        positive_label: str = "yes",
        class_weight: Optional[str] = None,
        name: Optional[str] = None,
    ):
        if name is None:
            name = "bow_tfidf"
        super().__init__(name)

        self.bow_config = BowTfidfConfig(
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            cv=cv,
            max_iter=max_iter,
            seed=seed,
            text_key=text_key,
            label_key=label_key,
            positive_label=positive_label,
            class_weight=class_weight,
        )

        self._vectorizer: Optional[TfidfVectorizer] = None
        self._classifier: Optional[LogisticRegressionCV] = None

    def train(self, data: Any) -> None:
        """
        Train TF-IDF + LogisticRegressionCV on text data.

        Args:
            data: list of dicts, each with text_key and label_key fields.
        """
        cfg = self.bow_config
        texts = [e[cfg.text_key] for e in data]
        y = np.array([1 if e[cfg.label_key] == cfg.positive_label else 0 for e in data])

        self._vectorizer = TfidfVectorizer(
            max_features=cfg.max_features,
            sublinear_tf=cfg.sublinear_tf,
        )
        X = self._vectorizer.fit_transform(texts)

        self._classifier = LogisticRegressionCV(
            cv=cfg.cv,
            max_iter=cfg.max_iter,
            random_state=cfg.seed,
            class_weight=cfg.class_weight,
        )
        self._classifier.fit(X, y)

        train_acc = float(accuracy_score(y, self._classifier.predict(X)))

        # Save artifacts
        folder = self._output.run_folder
        with open(folder / "vectorizer.pkl", "wb") as f:
            pickle.dump(self._vectorizer, f)
        with open(folder / "classifier.pkl", "wb") as f:
            pickle.dump(self._classifier, f)

        train_info = {
            "n_train": len(data),
            "train_accuracy": train_acc,
            "n_features": X.shape[1],
            "class_distribution": {"positive": int(y.sum()), "negative": int(len(y) - y.sum())},
        }
        with open(folder / "train_info.json", "w") as f:
            json.dump(train_info, f, indent=2)

        print(f"BoW TF-IDF trained: {len(data)} samples, train_acc={train_acc:.3f}")

    def infer(self, data: Any) -> List[Dict]:
        """
        Run inference on text data.

        Args:
            data: list of dicts, each with text_key (and optionally label_key).

        Returns:
            List of result dicts with predictions and probabilities.
        """
        if self._vectorizer is None or self._classifier is None:
            raise RuntimeError("Model not trained. Call train() first.")

        cfg = self.bow_config
        texts = [e[cfg.text_key] for e in data]
        X = self._vectorizer.transform(texts)

        y_pred = self._classifier.predict(X)
        y_prob = self._classifier.predict_proba(X)[:, 1]

        has_labels = cfg.label_key in data[0]
        if has_labels:
            y_true = np.array([1 if e[cfg.label_key] == cfg.positive_label else 0 for e in data])

        results = []
        for i, entry in enumerate(data):
            row = {
                "prediction": cfg.positive_label if y_pred[i] == 1 else "no",
                "prob_positive": float(y_prob[i]),
            }
            if has_labels:
                row["label"] = entry[cfg.label_key]
            results.append(row)

        # Compute and save metrics
        folder = self._output.run_folder
        output = {"predictions": results}

        if has_labels:
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )

            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
            try:
                metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                metrics["auc_roc"] = None
            output["metrics"] = metrics

            auc_s = f"{metrics['auc_roc']:.3f}" if metrics['auc_roc'] is not None else "N/A"
            print(f"BoW TF-IDF eval: acc={metrics['accuracy']:.3f} "
                  f"f1={metrics['f1']:.3f} auc={auc_s}")

        with open(folder / "results.json", "w") as f:
            json.dump(output, f, indent=2)

        return results
