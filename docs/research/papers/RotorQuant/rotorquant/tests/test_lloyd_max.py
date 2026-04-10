"""Tests for Lloyd-Max codebook construction."""
import pytest
import torch
import math

from turboquant.lloyd_max import LloydMaxCodebook, solve_lloyd_max, compute_expected_distortion


class TestLloydMaxCodebook:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_correct_number_of_centroids(self, bits):
        cb = LloydMaxCodebook(128, bits)
        assert cb.n_levels == 2 ** bits
        assert len(cb.centroids) == 2 ** bits

    @pytest.mark.parametrize("bits", [1, 2, 3])
    def test_centroids_are_symmetric(self, bits):
        cb = LloydMaxCodebook(128, bits)
        centroid_sum = cb.centroids.sum().abs().item()
        assert centroid_sum < 0.01, f"Centroids not symmetric: sum={centroid_sum}"

    def test_centroids_are_sorted(self):
        cb = LloydMaxCodebook(128, 3)
        for i in range(len(cb.centroids) - 1):
            assert cb.centroids[i] < cb.centroids[i+1]

    @pytest.mark.parametrize("d", [64, 128, 256])
    def test_centroids_scale_with_dimension(self, d):
        """Centroids should scale as ~1/sqrt(d)."""
        cb = LloydMaxCodebook(d, 2)
        max_c = cb.centroids.abs().max().item()
        expected_scale = 1.0 / math.sqrt(d)
        assert max_c < 5 * expected_scale, f"Centroids too large for d={d}"

    def test_distortion_decreases_with_bits(self):
        distortions = []
        for bits in [1, 2, 3, 4]:
            cb = LloydMaxCodebook(128, bits)
            distortions.append(cb.distortion)
        for i in range(len(distortions) - 1):
            assert distortions[i] > distortions[i+1]

    def test_quantize_dequantize(self):
        cb = LloydMaxCodebook(128, 3)
        x = torch.randn(100)
        idx = cb.quantize(x)
        x_hat = cb.dequantize(idx)
        # All reconstructed values should be valid centroids
        for v in x_hat.unique():
            assert v in cb.centroids

    def test_boundaries_count(self):
        cb = LloydMaxCodebook(128, 3)
        assert len(cb.boundaries) == cb.n_levels - 1


class TestSolveLloydMax:
    def test_exact_vs_approx(self):
        """Exact Beta PDF and Gaussian approx should give similar centroids for d>=64."""
        c_exact, _ = solve_lloyd_max(128, 2, use_exact=True)
        c_approx, _ = solve_lloyd_max(128, 2, use_exact=False)
        assert torch.allclose(c_exact, c_approx, atol=0.01)

    def test_convergence(self):
        """Should converge (not error) for various configs."""
        for d in [32, 64, 128]:
            for bits in [1, 2, 3]:
                c, b = solve_lloyd_max(d, bits)
                assert len(c) == 2 ** bits
                assert len(b) == 2 ** bits - 1
