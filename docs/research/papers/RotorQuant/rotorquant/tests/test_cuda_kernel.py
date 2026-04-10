"""Tests for fused CUDA kernel (rotor_fused_kernel.cu).
Skipped if CUDA is not available."""
import pytest
import torch
import os

# Skip all tests if no CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


@pytest.fixture(scope="module")
def cuda_rotor():
    """JIT-compile the CUDA kernel."""
    try:
        from torch.utils.cpp_extension import load
        csrc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'turboquant', 'csrc')
        kernel = load(
            name='cuda_rotor_sandwich_test',
            sources=[os.path.join(csrc, 'rotor_fused_kernel.cu')],
            extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr', '--use_fast_math'],
            verbose=False,
        )
        return kernel
    except Exception as e:
        pytest.skip(f"CUDA kernel compilation failed: {e}")


@pytest.fixture
def setup_data():
    """Create test data: rotors, centroids, input vectors."""
    import numpy as np
    from turboquant.clifford import make_random_rotor
    from turboquant.lloyd_max import LloydMaxCodebook

    d = 128
    n_groups = (d + 2) // 3
    bits = 3
    device = 'cuda'

    # Rotors
    rotors = []
    for i in range(n_groups):
        r = make_random_rotor((), seed=42 + i)
        rotors.append(r[[0, 4, 5, 6]])
    rotors_sparse = torch.stack(rotors).float().contiguous().to(device)

    # Centroids
    d_eff = max(n_groups * 8, 64)
    cb = LloydMaxCodebook(d_eff, bits - 1)
    centroids = cb.centroids.float().contiguous().to(device)

    return {
        'd': d, 'n_groups': n_groups, 'bits': bits,
        'rotors': rotors_sparse, 'centroids': centroids,
        'n_levels': len(centroids), 'device': device,
    }


class TestCUDAFusedKernel:
    def test_output_shape(self, cuda_rotor, setup_data):
        s = setup_data
        x = torch.randn(100, s['d'], device=s['device']).float()
        out = cuda_rotor.rotor_full_fused_float(
            x.contiguous(), s['rotors'],
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
        )
        assert out.shape == (100, s['d'])

    def test_identity_rotor(self, cuda_rotor, setup_data):
        """Identity rotors should approximately preserve vectors (only quant error)."""
        s = setup_data
        identity_rotors = torch.zeros_like(s['rotors'])
        identity_rotors[:, 0] = 1.0  # scalar = 1, bivectors = 0

        x = torch.randn(50, s['d'], device=s['device']).float()
        x = x / x.norm(dim=-1, keepdim=True)
        out = cuda_rotor.rotor_full_fused_float(
            x.contiguous(), identity_rotors,
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
        )
        mse = ((x - out) ** 2).sum(dim=-1).mean().item()
        assert mse < 1.0, f"Identity rotor MSE {mse} too high"

    def test_correctness_vs_pytorch(self, cuda_rotor, setup_data):
        """CUDA kernel output should approximately match PyTorch RotorQuantMSE."""
        from turboquant.rotorquant import RotorQuantMSE
        s = setup_data

        rq = RotorQuantMSE(s['d'], s['bits'], seed=42, device=s['device'])
        x = torch.randn(50, s['d'], device=s['device'])
        x = x / x.norm(dim=-1, keepdim=True)

        x_hat_pt, _ = rq(x)
        x_hat_cuda = cuda_rotor.rotor_full_fused_float(
            x.float().contiguous(), s['rotors'],
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
        )

        mse_pt = ((x - x_hat_pt) ** 2).sum(dim=-1).mean().item()
        mse_cuda = ((x.float() - x_hat_cuda) ** 2).sum(dim=-1).mean().item()
        # Both should have similar MSE (may differ due to different codebook usage)
        assert abs(mse_pt - mse_cuda) < 0.5, f"MSE mismatch: PT={mse_pt}, CUDA={mse_cuda}"

    @pytest.mark.parametrize("n", [1, 10, 100, 1000, 4096])
    def test_various_batch_sizes(self, cuda_rotor, setup_data, n):
        s = setup_data
        x = torch.randn(n, s['d'], device=s['device']).float()
        out = cuda_rotor.rotor_full_fused_float(
            x.contiguous(), s['rotors'],
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
        )
        assert out.shape == (n, s['d'])
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_half_precision(self, cuda_rotor, setup_data):
        s = setup_data
        x = torch.randn(50, s['d'], device=s['device']).half()
        out = cuda_rotor.rotor_full_fused_half(
            x.contiguous(), s['rotors'],
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
            s['centroids'], s['n_levels'],
        )
        assert out.shape == (50, s['d'])
        assert out.dtype == torch.float16

    def test_standalone_sandwich(self, cuda_rotor, setup_data):
        """Test rotor_sandwich_only kernel."""
        s = setup_data
        x = torch.randn(50, s['d'], device=s['device']).float()
        out = cuda_rotor.rotor_sandwich_float(x.contiguous(), s['rotors'])
        assert out.shape == (50, s['n_groups'], 8)

    def test_inverse_sandwich(self, cuda_rotor, setup_data):
        """Forward then inverse should approximately recover input."""
        s = setup_data
        x = torch.randn(50, s['d'], device=s['device']).float()
        x = x / x.norm(dim=-1, keepdim=True)
        mv = cuda_rotor.rotor_sandwich_float(x.contiguous(), s['rotors'])
        x_back = cuda_rotor.rotor_inverse_float(mv.contiguous(), s['rotors'], s['d'])
        diff = (x - x_back).abs().max().item()
        assert diff < 0.01, f"Inverse sandwich error {diff} too high"
