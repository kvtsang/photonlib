import torch
import pytest
from photonlib.experimental import VoxelMeta


class TestVoxelMetaConstruction:
    def test_basic(self, ranges, shape):
        meta = VoxelMeta(ranges, shape)
        assert meta.ndim == ranges.shape[0]
        assert list(meta.voxel_shape) == shape.tolist()

    def test_voxel_size(self, ranges, shape):
        meta = VoxelMeta(ranges, shape)
        expected = (ranges[:, 1] - ranges[:, 0]) / shape.float()
        assert torch.allclose(meta.voxel_size, expected)

    def test_n_voxels(self, meta, shape):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        assert n_voxels == shape.prod().item()

    def test_shape_mismatch_ndim(self, ranges):
        wrong_shape = torch.tensor([10], dtype=torch.long)
        with pytest.raises((ValueError, RuntimeError)):
            VoxelMeta(ranges, wrong_shape)


class TestVoxelMetaCoordToIdx:
    def test_origin_voxel(self, meta):
        """First voxel center should map to flat index 0."""
        voxel_center = meta.mins + meta.voxel_size / 2.0
        idx = meta.coord_to_voxel(voxel_center.unsqueeze(0))
        assert idx.item() == 0

    def test_last_voxel(self, meta):
        """Last voxel center should map to last flat index."""
        voxel_center = meta.maxs - meta.voxel_size / 2.0
        idx = meta.coord_to_voxel(voxel_center.unsqueeze(0))
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        assert idx.item() == n_voxels - 1

    def test_sequential_along_last_axis(self, meta):
        """Adjacent voxels along last axis should have consecutive flat indices."""
        base = meta.mins + meta.voxel_size / 2.0
        p0 = base.clone().unsqueeze(0)
        p1 = base.clone()
        p1[-1] += meta.voxel_size[-1]
        p1 = p1.unsqueeze(0)
        idx0 = meta.coord_to_voxel(p0)
        idx1 = meta.coord_to_voxel(p1)
        assert idx1.item() - idx0.item() == 1

    def test_oob_clamped(self, meta):
        """Out-of-bounds coordinates should be silently clamped."""
        below = meta.mins - 100.0
        above = meta.maxs + 100.0
        coords = torch.stack([below, above])
        idx = meta.coord_to_voxel(coords)
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        assert (idx >= 0).all()
        assert (idx < n_voxels).all()

    def test_batch_shape(self, meta):
        n = 50
        coords = (
            meta.mins.unsqueeze(0)
            + torch.rand(n, meta.ndim)
            * (meta.maxs - meta.mins).unsqueeze(0)
        )
        idx = meta.coord_to_voxel(coords)
        assert idx.shape == (n,)


class TestVoxelMetaIdxToCoord:
    def test_first_voxel(self, meta):
        idx = torch.tensor([0], dtype=torch.long)
        coord = meta.voxel_to_coord(idx)
        expected = meta.mins + meta.voxel_size / 2.0
        assert torch.allclose(coord.squeeze(0), expected)

    def test_last_voxel(self, meta):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        idx = torch.tensor([n_voxels - 1], dtype=torch.long)
        coord = meta.voxel_to_coord(idx)
        expected = meta.maxs - meta.voxel_size / 2.0
        assert torch.allclose(coord.squeeze(0), expected)

    def test_roundtrip(self, meta):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        n_test = min(100, n_voxels)
        idx = torch.randint(0, n_voxels, (n_test,))
        coords = meta.voxel_to_coord(idx)
        recovered_idx = meta.coord_to_voxel(coords)
        assert torch.equal(idx, recovered_idx)


class TestVoxelMetaDevice:
    def test_to_device(self, meta, device):
        meta = meta.to(device)
        assert meta.voxel_size.device.type == device.type
        assert meta.mins.device.type == device.type
        assert meta.maxs.device.type == device.type

    def test_coord_to_voxel_same_device(self, meta, device):
        meta = meta.to(device)
        coords = (meta.mins + meta.voxel_size / 2.0).unsqueeze(0)
        idx = meta.coord_to_voxel(coords)
        assert idx.device.type == device.type
