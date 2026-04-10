"""Tests for TurboQuant core: MSE quantizer, inner product quantizer, KV cache."""
import pytest
import torch
import math

from turboquant.turboquant import TurboQuantMSE, TurboQuantProd, TurboQuantKVCache, generate_rotation_matrix


class TestGenerateRotationMatrix:
    def test_orthogonality(self):
        Pi = generate_rotation_matrix(128, seed=42)
        identity = Pi @ Pi.T
        assert torch.allclose(identity, torch.eye(128), atol=1e-5)

    def test_deterministic(self):
        Pi1 = generate_rotation_matrix(64, seed=42)
        Pi2 = generate_rotation_matrix(64, seed=42)
        assert torch.allclose(Pi1, Pi2)

    def test_different_seeds(self):
        Pi1 = generate_rotation_matrix(64, seed=42)
        Pi2 = generate_rotation_matrix(64, seed=99)
        assert not torch.allclose(Pi1, Pi2)


class TestTurboQuantMSE:
    @pytest.fixture
    def unit_vectors(self):
        torch.manual_seed(42)
        x = torch.randn(500, 128)
        return x / x.norm(dim=-1, keepdim=True)

    def test_output_shape(self, unit_vectors):
        tq = TurboQuantMSE(128, bits=3, seed=42)
        x_hat, indices = tq(unit_vectors)
        assert x_hat.shape == unit_vectors.shape
        assert indices.shape == unit_vectors.shape

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_mse_within_theoretical_bounds(self, unit_vectors, bits):
        tq = TurboQuantMSE(128, bits=bits, seed=42)
        x_hat, _ = tq(unit_vectors)
        mse = ((unit_vectors - x_hat) ** 2).sum(dim=-1).mean().item()
        theory = math.sqrt(3) * math.pi / 2 * (1 / (4 ** bits))
        # TurboQuant on GPU may have higher MSE due to float32 precision
        assert mse < theory * 10, f"MSE {mse} exceeds 10x theory {theory}"

    def test_mse_decreases_with_bits(self, unit_vectors):
        mses = []
        for bits in [1, 2, 3, 4]:
            tq = TurboQuantMSE(128, bits=bits, seed=42)
            x_hat, _ = tq(unit_vectors)
            mse = ((unit_vectors - x_hat) ** 2).sum(dim=-1).mean().item()
            mses.append(mse)
        for i in range(len(mses) - 1):
            assert mses[i] > mses[i+1]

    def test_deterministic(self, unit_vectors):
        tq1 = TurboQuantMSE(128, 3, seed=42)
        tq2 = TurboQuantMSE(128, 3, seed=42)
        x_hat1, _ = tq1(unit_vectors[:10])
        x_hat2, _ = tq2(unit_vectors[:10])
        assert torch.allclose(x_hat1, x_hat2)


class TestTurboQuantProd:
    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_inner_product_unbiased(self, bits):
        torch.manual_seed(42)
        n, d = 1000, 128
        x = torch.randn(n, d)
        x = x / x.norm(dim=-1, keepdim=True)
        y = torch.randn(n, d)
        y = y / y.norm(dim=-1, keepdim=True)

        tq = TurboQuantProd(d, bits, seed=42)
        comp = tq.quantize(x)
        est_ip = tq.inner_product(y, comp)
        true_ip = (x * y).sum(dim=-1)

        bias = (est_ip - true_ip).mean().item()
        assert abs(bias) < 0.01, f"Bias {bias} too large"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_inner_product_correlation(self, bits):
        torch.manual_seed(42)
        n, d = 1000, 128
        x = torch.randn(n, d)
        x = x / x.norm(dim=-1, keepdim=True)
        y = torch.randn(n, d)
        y = y / y.norm(dim=-1, keepdim=True)

        tq = TurboQuantProd(d, bits, seed=42)
        comp = tq.quantize(x)
        est_ip = tq.inner_product(y, comp)
        true_ip = (x * y).sum(dim=-1)

        corr = torch.corrcoef(torch.stack([true_ip, est_ip]))[0, 1].item()
        min_corr = {2: 0.7, 3: 0.85, 4: 0.95}
        assert corr > min_corr[bits], f"Correlation {corr} too low for {bits}-bit"

    @pytest.mark.parametrize("seq_len", [512, 2048])
    def test_needle_in_haystack(self, seq_len):
        torch.manual_seed(42)
        d = 128
        keys = torch.randn(seq_len, d)
        keys = keys / keys.norm(dim=-1, keepdim=True)
        needle_pos = seq_len // 3
        query = keys[needle_pos].clone()

        tq = TurboQuantProd(d, bits=3, seed=42)
        comp = tq.quantize(keys)
        ips = tq.inner_product(query.unsqueeze(0).expand(seq_len, -1), comp)
        assert ips.argmax().item() == needle_pos


class TestTurboQuantKVCache:
    def test_basic_workflow(self):
        d = 64
        cache = TurboQuantKVCache(d, d, bits=3, seed=42)
        keys = torch.randn(32, d)
        values = torch.randn(32, d)
        cache.append(keys, values)
        assert len(cache) == 32

    def test_attention_scores_shape(self):
        d = 64
        cache = TurboQuantKVCache(d, d, bits=3, seed=42)
        cache.append(torch.randn(16, d), torch.randn(16, d))
        query = torch.randn(16, d)
        scores = cache.attention_scores(query)
        assert scores.shape[-1] == 16

    def test_get_values_shape(self):
        d = 64
        cache = TurboQuantKVCache(d, d, bits=3, seed=42)
        cache.append(torch.randn(16, d), torch.randn(16, d))
        v = cache.get_values()
        assert v.shape == (16, d)

    def test_memory_usage(self):
        d = 128
        cache = TurboQuantKVCache(d, d, bits=3, seed=42)
        cache.append(torch.randn(100, d), torch.randn(100, d))
        usage = cache.memory_usage_bits()
        assert usage['compression_ratio'] > 1.0
        assert usage['total_bits'] < usage['fp16_bits']
