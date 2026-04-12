import torch
import pytest
from photonlib.experimental import PhotonLib


class TestPhotonLibConstruction:
    def test_from_meta_and_vis(self, meta, vis):
        plib = PhotonLib(meta, vis)
        assert torch.equal(plib.vis, vis)
        assert plib.meta.ndim == meta.ndim

    def test_vis_shape_mismatch(self, meta):
        wrong_vis = torch.rand(999, 10)
        with pytest.raises((ValueError, RuntimeError)):
            PhotonLib(meta, wrong_vis)


class TestPhotonLibVisibility:
    def test_single_point(self, plib, meta, vis):
        center = meta.mins + meta.voxel_size / 2.0
        result = plib.visibility(center.unsqueeze(0))
        expected = vis[0]
        assert torch.allclose(result.squeeze(0), expected, atol=1e-6)

    def test_batch(self, plib, meta, vis):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        n_test = min(50, n_voxels)
        idx = torch.randint(0, n_voxels, (n_test,))
        coords = meta.voxel_to_coord(idx)
        result = plib.visibility(coords)
        expected = vis[idx]
        assert torch.allclose(result.cpu(), expected.cpu(), atol=1e-6)

    def test_output_shape(self, plib, meta, n_channels):
        n = 10
        coords = (
            meta.mins.unsqueeze(0)
            + torch.rand(n, meta.ndim)
            * (meta.maxs - meta.mins).unsqueeze(0)
        )
        result = plib.visibility(coords)
        assert result.shape == (n, n_channels)

    def test_oob_does_not_crash(self, plib, meta):
        far = (meta.maxs + 999.0).unsqueeze(0)
        result = plib.visibility(far)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestPhotonLibDevice:
    def test_to_device(self, plib, device):
        plib = plib.to(device)
        assert plib.vis.device.type == device.type

    def test_meta_moves_with_plib(self, plib, device):
        plib = plib.to(device)
        for name, buf in plib.meta.named_buffers():
            assert buf.device.type == device.type, (
                f"meta buffer '{name}' on {buf.device}, expected {device}"
            )

    def test_visibility_on_device(self, plib, meta, device):
        plib = plib.to(device)
        coords = (meta.mins + meta.voxel_size / 2.0).unsqueeze(0).to(device)
        result = plib.visibility(coords)
        assert result.device.type == device.type


class TestPhotonLibInputValidation:
    def test_wrong_type(self, plib, meta):
        import numpy as np
        with pytest.raises((TypeError, RuntimeError)):
            plib.visibility(np.random.randn(5, meta.ndim))

    def test_wrong_ndim_1d(self, plib, meta):
        wrong_len = meta.ndim + 1
        with pytest.raises(RuntimeError):
            plib.visibility(torch.randn(wrong_len))

    def test_wrong_ndim_2d(self, plib, meta):
        wrong_dim = meta.ndim + 1
        with pytest.raises(RuntimeError):
            plib.visibility(torch.randn(10, wrong_dim))

    def test_transposed(self, plib, meta):
        with pytest.raises(RuntimeError):
            plib.visibility(torch.randn(meta.ndim, 100))

    def test_3d_tensor_wrong_last_dim(self, plib, meta):
        """3D tensor with wrong last dim should fail."""
        wrong_dim = meta.ndim + 1
        with pytest.raises(RuntimeError):
            plib.visibility(torch.randn(10, 5, wrong_dim))

    def test_3d_tensor_matching_last_dim(self, plib, meta):
        """3D tensor with matching last dim is accepted (batch of batches)."""
        result = plib.visibility(torch.randn(10, meta.ndim, meta.ndim))
        # coord_to_voxel accepts (..., ndim), so this should work
        assert result is not None

    def test_integer_coords(self, plib, meta):
        """Integer coords are accepted (implicit cast by PyTorch)."""
        result = plib.visibility(
            torch.randint(0, 10, (5, meta.ndim))
        )
        assert result is not None

    def test_single_point_2d(self, plib, meta):
        """Single point as (1, ndim) tensor should work."""
        point = (meta.mins + meta.voxel_size / 2.0).unsqueeze(0)
        result = plib.visibility(point)
        assert result.shape[0] == 1

    def test_single_point_1d(self, plib, meta):
        """Single point as 1D (ndim,) tensor — check if accepted."""
        point = meta.mins + meta.voxel_size / 2.0
        # coord_to_voxel accepts (..., ndim), so (ndim,) should work
        result = plib.visibility(point)
        assert result is not None
