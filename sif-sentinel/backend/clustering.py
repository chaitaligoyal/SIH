"""
Emerging-threat clustering.

The original re-embedded the entire corpus and re-fit HDBSCAN from scratch on
every single API call. At six documents that is invisible; at OIL's real
reporting volume it is O(n) embedding plus a full clustering fit per request,
which will not run. It also used euclidean distance on unnormalised
sentence-transformer vectors, where cosine is the correct metric.

This version fits once at startup with `prediction_data=True` and assigns new
reports with `approximate_predict`, which is O(1)-ish per request.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:  # pragma: no cover
    HDBSCAN_AVAILABLE = False


class ThreatClusterer:
    def __init__(self, corpus: List[Dict[str, Any]], embedder, min_cluster_size: int = 3):
        self.corpus = corpus
        self.embedder = embedder
        self.fitted = False
        self.labels = np.array([])
        self.embeddings = np.array([])
        self.clusterer = None
        self.min_cluster_size = min_cluster_size
        self._fit()

    def _fit(self) -> None:
        if not HDBSCAN_AVAILABLE:
            print("[clustering] hdbscan not installed; clustering disabled.")
            return
        if len(self.corpus) < self.min_cluster_size * 2:
            print(
                f"[clustering] Corpus of {len(self.corpus)} reports is too small "
                f"to cluster meaningfully; clustering disabled. "
                f"Load more historical reports into data/."
            )
            return

        texts = [r["text"] for r in self.corpus]
        # Normalising lets euclidean distance stand in for cosine, which is the
        # right geometry for sentence-transformer embeddings.
        self.embeddings = self.embedder.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float64)

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=2,
            metric="euclidean",
            prediction_data=True,
        )
        self.labels = self.clusterer.fit_predict(self.embeddings)
        self.fitted = True

        counts = Counter(int(l) for l in self.labels if l != -1)
        print(
            f"[clustering] Fitted on {len(texts)} reports: "
            f"{len(counts)} cluster(s), "
            f"{int((self.labels == -1).sum())} unclustered."
        )

    def assign(self, text: str) -> Dict[str, Any]:
        if not self.fitted:
            return {
                "cluster_id": None,
                "is_emerging_threat": False,
                "cluster_summary": "Clustering unavailable — insufficient historical corpus.",
                "correlated_precursors": [],
            }

        emb = self.embedder.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float64)
        label, strength = hdbscan.approximate_predict(self.clusterer, emb)
        label = int(label[0])
        strength = float(strength[0])

        if label == -1:
            return {
                "cluster_id": -1,
                "membership_strength": round(strength, 3),
                "is_emerging_threat": False,
                "cluster_summary": "Isolated report — no matching precursor pattern in the historical corpus.",
                "correlated_precursors": [],
            }

        members = [i for i, l in enumerate(self.labels) if l == label]
        samples = [self.corpus[i]["text"][:220] for i in members[:5]]
        sources = Counter(self.corpus[i]["source"] for i in members)

        return {
            "cluster_id": label,
            "membership_strength": round(strength, 3),
            "is_emerging_threat": True,
            "cluster_size": len(members),
            "cluster_summary": (
                f"Matches precursor pattern #{label}, shared with {len(members)} "
                f"historical report(s) across {len(sources)} source(s)."
            ),
            "correlated_precursors": samples,
        }

    def cluster_overview(self, top_n: int = 8) -> List[Dict[str, Any]]:
        """Cluster sizes and representative text, for the dashboard feed."""
        if not self.fitted:
            return []
        counts = Counter(int(l) for l in self.labels if l != -1)
        out = []
        for cid, size in counts.most_common(top_n):
            members = [i for i, l in enumerate(self.labels) if l == cid]
            out.append({
                "cluster_id": cid,
                "size": size,
                "representative": self.corpus[members[0]]["text"][:200],
                "sources": sorted({self.corpus[i]["source"] for i in members}),
            })
        return out
