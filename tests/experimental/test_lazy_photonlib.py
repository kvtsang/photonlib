import torch
import pytest

from photonlib.experimental import PhotonLib
from photonlib.experimental.photonlib import LazyPhotonLib


# ---- Construction --------------------------------------------

class TestLazyPhotonLibConstruction:
    def test_from_filepath(self, plib_h5):
        filepath, _, _ = plib_h5
        lplib = LazyPhotonLib(filepath)
        assert lplib is not None

    def test_from_filepath_with_group(self, plib_h5_grouped):
        filepath, _, _ = plib_h5_grouped
        lplib = LazyPhotonLib(filepath, group='vol_a')
        assert lplib is not None

    def test_attributes(self, plib_h5):
        filepath, _, _ = plib_h5
        lplib = LazyPhotonLib(filepath)
        assert lplib.filepath == filepath
        assert lplib.group is None

    def test_group_attribute(self, plib_h5_grouped):
        filepath, _, _ = plib_h5_grouped
        lplib = LazyPhotonLib(filepath, group='vol_a')
        assert lplib.group == 'vol_a'

    def test_load_classmethod(self, plib_h5):
        filepath, _, _ = plib_h5
        lplib = LazyPhotonLib.load(filepath)
        assert isinstance(lplib, LazyPhotonLib)

    def test_load_classmethod_with_group(self, plib_h5_grouped):
        filepath, _, _ = plib_h5_grouped
        lplib = LazyPhotonLib.load(filepath, group='vol_a')
        assert isinstance(lplib, LazyPhotonLib)


# ---- Properties ----------------------------------------------

class TestLazyPhotonLibProperties:
    def test_n_voxels(self, plib_h5):
        filepath, meta, _ = plib_h5
        lplib = LazyPhotonLib(filepath)
        expected = torch.tensor(meta.voxel_shape).prod().item()
        assert lplib.n_voxels == expected

    def test_n_pmts(self, plib_h5, n_channels):
        filepath, _, _ = plib_h5
        lplib = LazyPhotonLib(filepath)
        assert lplib.n_pmts == n_channels

    def test_len(self, plib_h5):
        filepath, meta, _ = plib_h5
        lplib = LazyPhotonLib(filepath)
        expected = torch.tensor(meta.voxel_shape).prod().item()
        assert len(lplib) == expected

    def test_meta(self, plib_h5):
        filepath, meta, _ = plib_h5
        lplib = LazyPhotonLib(filepath)
        assert lplib.meta.ndim == meta.ndim
        assert torch.allclose(lplib.meta.mins, meta.mins)
        assert torch.allclose(lplib.meta.maxs, meta.maxs)


# ---- Voxel Lookup --------------------------------------------

class TestLazyPhotonLibVoxelLookup:
    def test_single_voxel(self, plib_h5):
        filepath, _, vis = plib_h5
        lplib = LazyPhotonLib(filepath)
        idx = torch.tensor([0])
        result = lplib.vis_at_voxel(idx)
        assert torch.allclose(result.squeeze(0), vis[0], atol=1e-6)

    def test_batch_voxels(self, plib_h5):
        filepath, meta, vis = plib_h5
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        n_test = min(50, n_voxels)
        indices = torch.randint(0, n_voxels, (n_test,))
        lplib = LazyPhotonLib(filepath)
        result = lplib.vis_at_voxel(indices)
        expected = vis[indices]
        assert torch.allclose(result, expected, atol=1e-6)

    def test_getitem(self, plib_h5):
        filepath, _, vis = plib_h5
        lplib = LazyPhotonLib(filepath)
        indices = torch.tensor([0, 1, 2])
        result = lplib[indices]
        expected = vis[indices]
        assert torch.allclose(result, expected, atol=1e-6)

    def test_output_shape(self, plib_h5, n_channels):
        filepath, _, _ = plib_h5
        n = 10
        lplib = LazyPhotonLib(filepath)
        indices = torch.zeros(n, dtype=torch.long)
        result = lplib.vis_at_voxel(indices)
        assert result.shape == (n, n_channels)

    def test_first_and_last_voxel(self, plib_h5):
        filepath, meta, vis = plib_h5
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        lplib = LazyPhotonLib(filepath)
        first = lplib.vis_at_voxel(torch.tensor([0]))
        last = lplib.vis_at_voxel(torch.tensor([n_voxels - 1]))
        assert torch.allclose(first.squeeze(0), vis[0], atol=1e-6)
        assert torch.allclose(last.squeeze(0), vis[-1], atol=1e-6)


# ---- Coordinate Lookup --------------------------------------

class TestLazyPhotonLibCoordLookup:
    def test_single_point(self, plib_h5):
        filepath, meta, vis = plib_h5
        lplib = LazyPhotonLib(filepath)
        point = (meta.mins + meta.voxel_size / 2.0).unsqueeze(0)
        result = lplib.vis_at_coord(point)
        expected = vis[0]
        assert torch.allclose(result.squeeze(0), expected, atol=1e-6)

    def test_batch_coords(self, plib_h5):
        filepath, meta, vis = plib_h5
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        n_test = min(50, n_voxels)
        indices = torch.randint(0, n_voxels, (n_test,))
        coords = meta.voxel_to_coord(indices)
        lplib = LazyPhotonLib(filepath)
        result = lplib.vis_at_coord(coords)
        expected = vis[indices]
        assert torch.allclose(result, expected, atol=1e-6)

    def test_output_shape(self, plib_h5, n_channels):
        filepath, meta, _ = plib_h5
        n = 10
        coords = (
            meta.mins.unsqueeze(0)
            + torch.rand(n, meta.ndim)
            * (meta.maxs - meta.mins).unsqueeze(0)
        )
        lplib = LazyPhotonLib(filepath)
        result = lplib.vis_at_coord(coords)
        assert result.shape == (n, n_channels)

    def test_visibility_backward_compat(self, plib_h5):
        filepath, meta, _ = plib_h5
        point = (meta.mins + meta.voxel_size / 2.0).unsqueeze(0)
        lplib = LazyPhotonLib(filepath)
        r1 = lplib.vis_at_coord(point)
        r2 = lplib.visibility(point)
        assert torch.equal(r1, r2)

    def test_oob_does_not_crash(self, plib_h5):
        filepath, meta, _ = plib_h5
        far = (meta.maxs + 999.0).unsqueeze(0)
        lplib = LazyPhotonLib(filepath)
        result = lplib.vis_at_coord(far)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


# ---- Consistency with PhotonLib ------------------------------

class TestLazyPhotonLibConsistency:
    def test_matches_photonlib(self, plib_h5):
        filepath, meta, vis = plib_h5
        plib = PhotonLib(meta, vis)
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        n_test = min(100, n_voxels)
        indices = torch.randint(0, n_voxels, (n_test,))
        coords = meta.voxel_to_coord(indices)

        lplib = LazyPhotonLib(filepath)
        lazy_result = lplib.vis_at_coord(coords)
        eager_result = plib.visibility(coords)
        assert torch.allclose(lazy_result, eager_result, atol=1e-6)

    def test_materialize_returns_photonlib(self, plib_h5):
        filepath, _, _ = plib_h5
        lplib = LazyPhotonLib(filepath)
        plib = lplib.materialize()
        assert isinstance(plib, PhotonLib)
        assert plib.n_pmts == lplib.n_pmts

    def test_materialize_matches_original(self, plib_h5):
        filepath, meta, vis = plib_h5
        lplib = LazyPhotonLib(filepath)
        plib = lplib.materialize()
        point = (meta.mins + meta.voxel_size / 2.0).unsqueeze(0)
        lazy_result = lplib.vis_at_coord(point)
        eager_result = plib.visibility(point)
        assert torch.allclose(lazy_result, eager_result, atol=1e-6)


# ---- Pickling / Fork Safety ----------------------------------

class TestLazyPhotonLibPickling:
    def test_picklable(self, plib_h5):
        """LazyPhotonLib must be picklable for spawn/forkserver."""
        import pickle
        filepath, _, vis = plib_h5
        lplib = LazyPhotonLib(filepath)
        # force file open by accessing data
        _ = lplib.vis_at_voxel(torch.tensor([0]))
        # pickle roundtrip
        restored = pickle.loads(pickle.dumps(lplib))
        result = restored.vis_at_voxel(torch.tensor([0]))
        assert torch.allclose(result.squeeze(0), vis[0], atol=1e-6)

    def test_getstate_closes_handle(self, plib_h5):
        """__getstate__ should exclude HDF5 handle."""
        import pickle
        filepath, _, _ = plib_h5
        lplib = LazyPhotonLib(filepath)
        _ = lplib.vis_at_voxel(torch.tensor([0]))
        state = lplib.__getstate__()
        # state should not contain open h5py objects
        pickled = pickle.dumps(state)
        assert pickled is not None

    def test_lazy_reopen_after_pickle(self, plib_h5):
        """After unpickling, file should reopen lazily on access."""
        import pickle
        filepath, meta, vis = plib_h5
        lplib = LazyPhotonLib(filepath)
        restored = pickle.loads(pickle.dumps(lplib))
        # should work — file reopens on demand
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        idx = torch.randint(0, n_voxels, (10,))
        result = restored.vis_at_voxel(idx)
        expected = vis[idx]
        assert torch.allclose(result, expected, atol=1e-6)


# ---- Grouped HDF5 -------------------------------------------

class TestLazyPhotonLibGrouped:
    def test_load_group_a(self, plib_h5_grouped):
        filepath, metas, vises = plib_h5_grouped
        meta_a, _ = metas
        vis_a, _ = vises
        lplib = LazyPhotonLib(filepath, group='vol_a')
        assert lplib.n_pmts == vis_a.shape[1]
        assert torch.allclose(lplib.meta.mins, meta_a.mins)
        idx = torch.tensor([0])
        result = lplib.vis_at_voxel(idx)
        assert torch.allclose(result.squeeze(0), vis_a[0], atol=1e-6)

    def test_load_group_b(self, plib_h5_grouped):
        filepath, metas, vises = plib_h5_grouped
        _, meta_b = metas
        _, vis_b = vises
        lplib = LazyPhotonLib(filepath, group='vol_b')
        assert lplib.n_pmts == vis_b.shape[1]
        assert torch.allclose(lplib.meta.mins, meta_b.mins)
        idx = torch.tensor([0])
        result = lplib.vis_at_voxel(idx)
        assert torch.allclose(result.squeeze(0), vis_b[0], atol=1e-6)

    def test_groups_independent(self, plib_h5_grouped):
        filepath, _, vises = plib_h5_grouped
        vis_a, vis_b = vises
        la = LazyPhotonLib(filepath, group='vol_a')
        lb = LazyPhotonLib(filepath, group='vol_b')
        idx = torch.tensor([0])
        ra = la.vis_at_voxel(idx)
        rb = lb.vis_at_voxel(idx)
        assert torch.allclose(ra.squeeze(0), vis_a[0], atol=1e-6)
        assert torch.allclose(rb.squeeze(0), vis_b[0], atol=1e-6)


# ---- Meta Device ---------------------------------------------

class TestLazyPhotonLibMeta:
    def test_meta_is_submodule(self, plib_h5):
        filepath, _, _ = plib_h5
        lplib = LazyPhotonLib(filepath)
        assert 'meta' in dict(lplib.named_modules())

# ---- Unique query -------------------------------------------

class TestLazyPhotonLibUniquePolicy:
    def test_default_assume_unique_false(self, plib_h5):
        filepath, _, _ = plib_h5
        lplib = LazyPhotonLib(filepath)
        assert lplib.assume_unique is False

    def test_set_assume_unique(self, plib_h5):
        filepath, _, _ = plib_h5
        lplib = LazyPhotonLib(filepath)
        lplib.assume_unique = True
        assert lplib.assume_unique is True

    def test_dedup_with_duplicates(self, plib_h5):
        """Default (dedup) should handle duplicate indices correctly."""
        filepath, _, vis = plib_h5
        lplib = LazyPhotonLib(filepath)
        indices = torch.tensor([0, 1, 0, 2, 1])
        result = lplib.vis_at_voxel(indices)
        assert torch.allclose(result[0], vis[0], atol=1e-6)
        assert torch.allclose(result[1], vis[1], atol=1e-6)
        assert torch.allclose(result[2], vis[0], atol=1e-6)
        assert torch.allclose(result[3], vis[2], atol=1e-6)
        assert torch.allclose(result[4], vis[1], atol=1e-6)

    def test_unique_flag_skip_dedup(self, plib_h5):
        """unique=True with unique indices should work."""
        filepath, _, vis = plib_h5
        lplib = LazyPhotonLib(filepath)
        indices = torch.tensor([0, 1, 2, 3, 4])
        result = lplib.vis_at_voxel(indices, unique=True)
        expected = vis[:5]
        assert torch.allclose(result, expected, atol=1e-6)

    def test_unique_matches_dedup(self, plib_h5):
        """unique=True and unique=False should give same result for unique indices."""
        filepath, meta, _ = plib_h5
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        n_test = min(50, n_voxels)
        indices = torch.randperm(n_voxels)[:n_test]

        lplib = LazyPhotonLib(filepath)
        result_dedup = lplib.vis_at_voxel(indices, unique=False)
        result_unique = lplib.vis_at_voxel(indices, unique=True)
        assert torch.allclose(result_dedup, result_unique, atol=1e-6)

    def test_assume_unique_affects_getitem(self, plib_h5):
        """__getitem__ should respect assume_unique."""
        filepath, _, vis = plib_h5
        indices = torch.tensor([3, 1, 4])

        lplib = LazyPhotonLib(filepath)
        lplib.assume_unique = False
        result_dedup = lplib[indices]

        lplib.assume_unique = True
        result_unique = lplib[indices]

        assert torch.allclose(result_dedup, result_unique, atol=1e-6)

    def test_per_call_overrides_class_default(self, plib_h5):
        """unique= kwarg should override assume_unique."""
        filepath, _, vis = plib_h5
        indices = torch.tensor([0, 1, 2])

        lplib = LazyPhotonLib(filepath)
        lplib.assume_unique = True
        result = lplib.vis_at_voxel(indices, unique=False)
        expected = vis[:3]
        assert torch.allclose(result, expected, atol=1e-6)

        lplib.assume_unique = False
        result = lplib.vis_at_voxel(indices, unique=True)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_getitem_int(self, plib_h5):
        """__getitem__ with int should work regardless of assume_unique."""
        filepath, _, vis = plib_h5
        lplib = LazyPhotonLib(filepath)

        for assume in [False, True]:
            lplib.assume_unique = assume
            result = lplib[0]
            assert result.shape == (vis.shape[1],)
            assert torch.allclose(result, vis[0], atol=1e-6)

    def test_getitem_list(self, plib_h5):
        """__getitem__ with list should work."""
        filepath, _, vis = plib_h5
        lplib = LazyPhotonLib(filepath)
        result = lplib[[0, 1, 2]]
        expected = vis[:3]
        assert torch.allclose(result, expected, atol=1e-6)

    def test_unsorted_unique_indices(self, plib_h5):
        """unique=True with unsorted indices should preserve order."""
        filepath, _, vis = plib_h5
        indices = torch.tensor([4, 2, 0, 3, 1])
        lplib = LazyPhotonLib(filepath)
        result = lplib.vis_at_voxel(indices, unique=True)
        for i, idx in enumerate(indices):
            assert torch.allclose(result[i], vis[idx], atol=1e-6)
