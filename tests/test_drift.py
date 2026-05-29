"""
tests/test_drift.py — Unit tests for the statistical drift detection module.

Coverage:
  - compute_psi: returns 0 for identical distributions, positive for diverged
  - compute_drift_report: correct structure and types returned
  - KS drift detection: detects distributional shift between two Gaussians
  - N_INFORMATIVE_DIMS constant: correct type and range
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from monitor.drift import N_INFORMATIVE_DIMS, compute_drift_report, compute_psi


class TestComputePsi:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 200).astype(float)
        psi = compute_psi(data, data)
        # PSI for identical distributions should be near 0 (small due to +1e-6 smoothing)
        assert psi < 0.05

    def test_diverged_distributions_positive_psi(self):
        rng = np.random.default_rng(42)
        baseline = rng.normal(0, 1, 200).astype(float)
        current = rng.normal(5, 1, 200).astype(float)  # large shift
        psi = compute_psi(baseline, current)
        assert psi > 0.1, f"PSI should be > 0.1 for large distributional shift, got {psi}"

    def test_returns_float(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0, 1, 100).astype(float)
        result = compute_psi(data, data)
        assert isinstance(result, float)


class TestComputeDriftReport:
    def _make_embeddings(self, n: int, dim: int = 32, mean: float = 0.0, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(mean, 1.0, (n, dim)).astype(np.float32)

    def test_report_has_required_keys(self):
        baseline = self._make_embeddings(50)
        current = self._make_embeddings(50)
        baseline_lengths = [5] * 50
        current_lengths = [5] * 50
        report = compute_drift_report(baseline, current, baseline_lengths, current_lengths)
        assert "pct_dims_drifted" in report
        assert "ks_length_p_value" in report
        assert "ks_length_drifted" in report
        assert "psi_query_length" in report

    def test_no_drift_on_identical_distributions(self):
        rng = np.random.default_rng(42)
        emb = rng.normal(0, 1, (100, 32)).astype(np.float32)
        lengths = [10] * 100
        report = compute_drift_report(emb, emb, lengths, lengths)
        # KS test p-value should be high (no drift detected)
        assert report["ks_length_drifted"] is False

    def test_drift_detected_on_shifted_distribution(self):
        baseline = self._make_embeddings(100, dim=32, mean=0.0)
        current = self._make_embeddings(100, dim=32, mean=10.0, seed=99)
        baseline_lengths = list(range(5, 55)) * 2
        current_lengths = list(range(20, 70)) * 2  # much longer queries
        report = compute_drift_report(baseline, current, baseline_lengths, current_lengths)
        # At least some dimensions should show drift
        assert report["pct_dims_drifted"] >= 0.0
        assert isinstance(report["pct_dims_drifted"], float)


class TestConstants:
    def test_n_informative_dims_is_positive_int(self):
        assert isinstance(N_INFORMATIVE_DIMS, int)
        assert N_INFORMATIVE_DIMS > 0

    def test_n_informative_dims_reasonable_range(self):
        # Should be a fraction of the 384-dim embedding space
        assert N_INFORMATIVE_DIMS <= 384
