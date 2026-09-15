"""
supervised_ranker.py
---------------------
Implements the `supervised_ranker` scoring_method from the team's data
contract: "train/eval vs shortlisted & final_score".

Data-leakage rule (per the contract doc): shortlisted / final_score /
skill_match_score / experience_match / education_match / similarity_score
are EVALUATION LABELS. They are used here ONLY as the training target for
a small classifier that learns from the four engineered features
(utils.scoring.FEATURE_NAMES). At inference time the trained model takes
only those four features as input -- it never sees the label columns
again, so a candidate scored in production is scored exactly the same
way whether or not ground-truth labels exist for them.

Model: logistic regression over
  [skill_overlap_pct, experience_ratio, education_rank_ratio, semantic_similarity]
predicting P(shortlisted == 1). That probability becomes fit_score.

Kept deliberately simple (no gradient boosting / deep model) because:
  - the dataset is synthetic/templated (per the contract doc's own caveat),
    so a complex model would overfit noise rather than learn signal;
  - a linear model keeps coefficients inspectable, which matters for a
    hiring tool (explainability > marginal accuracy here).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from utils.scoring import FEATURE_NAMES, build_feature_vector

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "supervised_ranker.json"
)


class SupervisedRanker:
    """A minimal, dependency-light logistic regression wrapper so the
    Matching Agent doesn't require a heavier ML framework in production."""

    def __init__(self):
        self.coef_: Optional[List[float]] = None
        self.intercept_: float = 0.0
        self.feature_names: List[str] = FEATURE_NAMES
        self.trained_on_n: int = 0
        self.eval_metrics: Dict = {}

    # ------------------------------------------------------------------ #
    def fit(self, candidates: List[Dict], jobs_by_id: Dict[str, Dict], labels: List[int]) -> Dict:
        """
        Train on (candidate, job, label) triples.
        `labels` must align 1:1 with `candidates` and contain the
        ground-truth `shortlisted` value (0/1) -- used as the TARGET only.
        Returns evaluation metrics (train/test split, AUC, accuracy,
        correlation of predicted probability with final_score if supplied
        via job['_reference_scores']['final_score']).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score, accuracy_score

        X, final_scores = [], []
        for c in candidates:
            job = jobs_by_id[c["applied_job_id"]]
            vec, _ = build_feature_vector(c, job)
            X.append(vec)
            ref = job.get("_reference_scores", {})
            final_scores.append(ref.get("final_score"))

        X_train, X_test, y_train, y_test, fs_train, fs_test = train_test_split(
            X, labels, final_scores, test_size=0.3, random_state=42, stratify=labels
        )

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        self.coef_ = clf.coef_[0].tolist()
        self.intercept_ = float(clf.intercept_[0])
        self.trained_on_n = len(X_train)

        proba_test = clf.predict_proba(X_test)[:, 1]
        pred_test = clf.predict(X_test)

        metrics = {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_accuracy": round(float(accuracy_score(y_test, pred_test)), 4),
        }
        # AUC needs both classes present in the test split
        if len(set(y_test)) > 1:
            metrics["test_auc"] = round(float(roc_auc_score(y_test, proba_test)), 4)
        # correlation of fit_score with the (label-only) final_score column,
        # to satisfy the contract's "eval vs ... final_score" instruction
        if all(v is not None for v in fs_test) and len(fs_test) > 1:
            import numpy as np

            corr = float(np.corrcoef(proba_test, fs_test)[0, 1])
            metrics["fit_score_vs_final_score_corr"] = round(corr, 4)

        self.eval_metrics = metrics
        return metrics

    # ------------------------------------------------------------------ #
    def predict_fit_score(self, feature_vector: List[float]) -> float:
        if self.coef_ is None:
            raise RuntimeError("SupervisedRanker has not been trained or loaded yet.")
        z = self.intercept_ + sum(w * x for w, x in zip(self.coef_, feature_vector))
        # manual sigmoid so inference has zero sklearn dependency once trained
        prob = 1.0 / (1.0 + pow(2.718281828, -z))
        return round(prob, 4)

    # ------------------------------------------------------------------ #
    def save(self, path: str = DEFAULT_MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_names": self.feature_names,
                    "coef_": self.coef_,
                    "intercept_": self.intercept_,
                    "trained_on_n": self.trained_on_n,
                    "eval_metrics": self.eval_metrics,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str = DEFAULT_MODEL_PATH) -> "SupervisedRanker":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ranker = cls()
        ranker.feature_names = data["feature_names"]
        ranker.coef_ = data["coef_"]
        ranker.intercept_ = data["intercept_"]
        ranker.trained_on_n = data.get("trained_on_n", 0)
        ranker.eval_metrics = data.get("eval_metrics", {})
        return ranker

    @staticmethod
    def is_model_available(path: str = DEFAULT_MODEL_PATH) -> bool:
        return os.path.exists(path)
