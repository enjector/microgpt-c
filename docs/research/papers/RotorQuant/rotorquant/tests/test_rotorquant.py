"""Tests for RotorQuant: MSE, inner product, KV cache."""
import pytest
import torch
import math

from turboquant.rotorquant import RotorQuantMSE, RotorQuantProd, RotorQuantKVCache


class TestRotorQuantMSE:
    @pytest.fixture
    def unit_vectors(self):
        torch.manual_seed(42)
        x = torch.randn(500, 128)
        return x / x.norm(dim=-1, keepdim=True)

    def test_output_shape(self, unit_vectors):
        rq = RotorQuantMSE(128, bits=3, seed=42)
        x_hat, indices = rq(unit_vectors)
        assert x_hat.shape == unit_vectors.shape

    def test_indices_are_dict(self, unit_vectors):
        rq = RotorQuantMSE(128, bits=3, seed=42)
        _, indices = rq(unit_vectors)
        assert isinstance(indices, dict)
        assert 'vector' in indices

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_mse_within_bounds(self, unit_vectors, bits):
        """MSE should be reasonable (not exploding)."""
        rq = RotorQuantMSE(128, bits=bits, seed=42)
        x_hat, _ = rq(unit_vectors)
        mse = ((unit_vectors - x_hat) ** 2).sum(dim=-1).mean().item()
        # Should be less than 2.0 even at 1-bit (random would be ~2.0)
        assert mse < 2.0, f"MSE {mse} too high for {bits}-bit"

    def test_mse_decreases_with_bits(self, unit_vectors):
        """More bits should give lower MSE."""
        mses = []
        for bits in [1, 2, 3, 4]:
            rq = RotorQuantMSE(128, bits=bits, seed=42)
            x_hat, _ = rq(unit_vectors)
            mse = ((unit_vectors - x_hat) ** 2).sum(dim=-1).mean().item()
            mses.append(mse)
        for i in range(len(mses) - 1):
            assert mses[i] > mses[i+1], f"MSE should decrease: {mses}"

    def test_deterministic(self, unit_vectors):
        rq1 = RotorQuantMSE(128, bits=3, seed=42)
        rq2 = RotorQuantMSE(128, bits=3, seed=42)
        x_hat1, _ = rq1(unit_vectors[:10])
        x_hat2, _ = rq2(unit_vectors[:10])
        assert torch.allclose(x_hat1, x_hat2)

    def test_different_seeds(self, unit_vectors):
        rq1 = RotorQuantMSE(128, bits=3, seed=42)
        rq2 = RotorQuantMSE(128, bits=3, seed=99)
        x_hat1, _ = rq1(unit_vectors[:10])
        x_hat2, _ = rq2(unit_vectors[:10])
        assert not torch.allclose(x_hat1, x_hat2)

    @pytest.mark.parametrize("d", [3, 6, 64, 127, 128, 129, 256])
    def test_various_dimensions(self, d):
        torch.manual_seed(42)
        x = torch.randn(20, d)
        x = x / x.norm(dim=-1, keepdim=True)
        rq = RotorQuantMSE(d, bits=3, seed=42)
        x_hat, _ = rq(x)
        assert x_hat.shape == (20, d)
        mse = ((x - x_hat) ** 2).sum(dim=-1).mean().item()
        assert mse < 2.0


class TestRotorQuantProd:
    @pytest.fixture
    def unit_vectors(self):
        torch.manual_seed(42)
        x = torch.randn(500, 128)
        return x / x.norm(dim=-1, keepdim=True)

    def test_quantize_returns_dict(self, unit_vectors):
        rq = RotorQuantProd(128, bits=3, seed=42)
        comp = rq.quantize(unit_vectors)
        assert 'mse_indices' in comp
        assert 'qjl_signs' in comp
        assert 'residual_norm' in comp

    def test_dequantize_shape(self, unit_vectors):
        rq = RotorQuantProd(128, bits=3, seed=42)
        comp = rq.quantize(unit_vectors)
        x_hat = rq.dequantize(comp)
        assert x_hat.shape == unit_vectors.shape

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_inner_product_unbiased(self, bits):
        """QJL correction should make inner product nearly unbiased."""
        torch.manual_seed(42)
        n = 1000
        d = 128
        x = torch.randn(n, d)
        x = x / x.norm(dim=-1, keepdim=True)
        y = torch.randn(n, d)
        y = y / y.norm(dim=-1, keepdim=True)

        rq = RotorQuantProd(d, bits, seed=42)
        comp = rq.quantize(x)
        est_ip = rq.inner_product(y, comp)
        true_ip = (x * y).sum(dim=-1)

        bias = (est_ip - true_ip).mean().item()
        assert abs(bias) < 0.05, f"Bias {bias} too large for {bits}-bit"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_inner_product_correlation(self, bits):
        """Estimated IP should correlate with true IP."""
        torch.manual_seed(42)
        n = 1000
        d = 128
        x = torch.randn(n, d)
        x = x / x.norm(dim=-1, keepdim=True)
        y = torch.randn(n, d)
        y = y / y.norm(dim=-1, keepdim=True)

        rq = RotorQuantProd(d, bits, seed=42)
        comp = rq.quantize(x)
        est_ip = rq.inner_product(y, comp)
        true_ip = (x * y).sum(dim=-1)

        corr = torch.corrcoef(torch.stack([true_ip, est_ip]))[0, 1].item()
        assert corr > 0.5, f"Correlation {corr} too low for {bits}-bit"

    def test_needle_in_haystack(self):
        """Should find the exact match in a haystack of vectors."""
        torch.manual_seed(42)
        d = 128
        seq_len = 2048

        keys = torch.randn(seq_len, d)
        keys = keys / keys.norm(dim=-1, keepdim=True)
        needle_pos = seq_len // 3
        query = keys[needle_pos].clone()

        rq = RotorQuantProd(d, bits=3, seed=42)
        comp = rq.quantize(keys)
        ips = rq.inner_product(query.unsqueeze(0).expand(seq_len, -1), comp)

        assert ips.argmax().item() == needle_pos


class TestRotorQuantKVCache:
    def test_append_and_score(self):
        d = 64
        cache = RotorQuantKVCache(d, d, bits=3, seed=42)

        keys = torch.randn(32, d)
        values = torch.randn(32, d)
        cache.append(keys, values)

        assert len(cache) == 32

        query = torch.randn(32, d)
        scores = cache.attention_scores(query)
        assert scores.shape[-1] == 32

    def test_get_values(self):
        d = 64
        cache = RotorQuantKVCache(d, d, bits=3, seed=42)
        values = torch.randn(16, d)
        cache.append(torch.randn(16, d), values)
        v_hat = cache.get_values()
        assert v_hat.shape == (16, d)
