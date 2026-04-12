import torch
import pytest

from photonlib.experimental.multilib import MultiPhotonLib


# -- Construction --------------------------------------------

class TestMultiPhotonLibConstruction:
    def test_from_list(self, plib_pair):
        mlib = MultiPhotonLib(list(plib_pair))
        assert mlib is not None

    def test_empty_list(self):
        with pytest.raises(ValueError):
            MultiPhotonLib([])

    def test_single_lib(self, plib_pair):
        mlib = MultiPhotonLib([plib_pair[0]])
        assert mlib.n_libs == 1


# -- Properties ----------------------------------------------

class TestMultiPhotonLibProperties:
    def test_n_libs(self, mlib):
        assert mlib.n_libs == 2

    def test_ndim(self, mlib, ndim):
        assert mlib.ndim == ndim

    def test_n_pmts(self, mlib, n_pmts_a, n_pmts_b):
        assert mlib.n_pmts == n_pmts_a + n_pmts_b

    def test_n_pmts_per_lib(self, mlib, n_pmts_a, n_pmts_b):
        assert mlib.n_pmts_per_lib == (n_pmts_a, n_pmts_b)

    def test_bbox(self, mlib, meta_pair):
        meta_a, meta_b = meta_pair
        expected_mins = torch.minimum(meta_a.mins, meta_b.mins)
        expected_maxs = torch.maximum(meta_a.maxs, meta_b.maxs)
        assert torch.allclose(mlib.bbox.mins, expected_mins)
        assert torch.allclose(mlib.bbox.maxs, expected_maxs)

    def test_libs_attribute(self, mlib):
        assert len(mlib.libs) == 2


# -- PMT Slicing ---------------------------------------------

class TestMultiPhotonLibPmtSlice:
    def test_first_slice(self, mlib, n_pmts_a):
        s = mlib.pmt_slice(0)
        assert s == slice(0, n_pmts_a)

    def test_second_slice(self, mlib, n_pmts_a, n_pmts_b):
        s = mlib.pmt_slice(1)
        assert s == slice(n_pmts_a, n_pmts_a + n_pmts_b)

    def test_slices_cover_all_pmts(self, mlib):
        total = 0
        for i in range(mlib.n_libs):
            s = mlib.pmt_slice(i)
            total += s.stop - s.start
        assert total == mlib.n_pmts

    def test_slices_non_overlapping(self, mlib):
        s0 = mlib.pmt_slice(0)
        s1 = mlib.pmt_slice(1)
        assert s0.stop <= s1.start


# -- Contain -------------------------------------------------

class TestMultiPhotonLibContain:
    def test_point_in_first(self, mlib, meta_pair):
        meta_a, _ = meta_pair
        point = (meta_a.mins + meta_a.voxel_size / 2.0).unsqueeze(0)
        assert mlib.contain(point).all()

    def test_point_in_second(self, mlib, meta_pair):
        _, meta_b = meta_pair
        point = (meta_b.mins + meta_b.voxel_size / 2.0).unsqueeze(0)
        assert mlib.contain(point).all()

    def test_point_outside_all(self, mlib, meta_pair):
        _, meta_b = meta_pair
        far = (meta_b.maxs + 999.0).unsqueeze(0)
        assert not mlib.contain(far).any()

    def test_batch_mixed(self, mlib, meta_pair):
        meta_a, meta_b = meta_pair
        inside_a = meta_a.mins + meta_a.voxel_size / 2.0
        inside_b = meta_b.mins + meta_b.voxel_size / 2.0
        outside = meta_b.maxs + 999.0
        coords = torch.stack([inside_a, inside_b, outside])
        mask = mlib.contain(coords)
        assert mask[0] and mask[1] and not mask[2]


# -- Visibility ----------------------------------------------

class TestMultiPhotonLibVisibility:
    def test_output_shape(self, mlib, meta_pair):
        meta_a, _ = meta_pair
        n = 5
        coords = (
            meta_a.mins.unsqueeze(0)
            + torch.rand(n, meta_a.ndim)
            * (meta_a.maxs - meta_a.mins).unsqueeze(0)
        )
        result = mlib.vis_at_coord(coords)
        assert result.shape == (n, mlib.n_pmts)

    def test_point_in_first_volume(self, mlib, plib_pair, meta_pair, n_pmts_a):
        plib_a, _ = plib_pair
        meta_a, _ = meta_pair
        point = (meta_a.mins + meta_a.voxel_size / 2.0).unsqueeze(0)
        result = mlib.vis_at_coord(point)
        expected_a = plib_a.visibility(point)
        assert torch.allclose(result[:, :n_pmts_a], expected_a, atol=1e-6)

    def test_point_in_second_volume(self, mlib, plib_pair, meta_pair, n_pmts_a):
        _, plib_b = plib_pair
        _, meta_b = meta_pair
        point = (meta_b.mins + meta_b.voxel_size / 2.0).unsqueeze(0)
        result = mlib.vis_at_coord(point)
        expected_b = plib_b.visibility(point)
        assert torch.allclose(result[:, n_pmts_a:], expected_b, atol=1e-6)

    def test_other_columns_are_default(self, mlib, meta_pair, n_pmts_a):
        meta_a, _ = meta_pair
        point = (meta_a.mins + meta_a.voxel_size / 2.0).unsqueeze(0)
        result = mlib.vis_at_coord(point, default=0.0)
        assert (result[:, n_pmts_a:] == 0.0).all()

    def test_custom_default(self, mlib, meta_pair, n_pmts_a):
        meta_a, _ = meta_pair
        point = (meta_a.mins + meta_a.voxel_size / 2.0).unsqueeze(0)
        fill = -1.0
        result = mlib.vis_at_coord(point, default=fill)
        assert (result[:, n_pmts_a:] == fill).all()

    def test_oob_all_default(self, mlib, meta_pair):
        _, meta_b = meta_pair
        far = (meta_b.maxs + 999.0).unsqueeze(0)
        result = mlib.vis_at_coord(far, default=0.0)
        assert (result == 0.0).all()

    def test_oob_custom_default(self, mlib, meta_pair):
        _, meta_b = meta_pair
        far = (meta_b.maxs + 999.0).unsqueeze(0)
        fill = -99.0
        result = mlib.vis_at_coord(far, default=fill)
        assert (result == fill).all()

    def test_batch_mixed_volumes(self, mlib, plib_pair, meta_pair, n_pmts_a):
        plib_a, plib_b = plib_pair
        meta_a, meta_b = meta_pair
        pt_a = (meta_a.mins + meta_a.voxel_size / 2.0).unsqueeze(0)
        pt_b = (meta_b.mins + meta_b.voxel_size / 2.0).unsqueeze(0)
        coords = torch.cat([pt_a, pt_b], dim=0)
        result = mlib.vis_at_coord(coords)
        exp_a = plib_a.visibility(pt_a)
        exp_b = plib_b.visibility(pt_b)
        assert torch.allclose(result[0, :n_pmts_a], exp_a.squeeze(0), atol=1e-6)
        assert (result[0, n_pmts_a:] == 0.0).all()
        assert (result[1, :n_pmts_a] == 0.0).all()
        assert torch.allclose(result[1, n_pmts_a:], exp_b.squeeze(0), atol=1e-6)

    def test_visibility_backward_compat(self, mlib, meta_pair):
        meta_a, _ = meta_pair
        point = (meta_a.mins + meta_a.voxel_size / 2.0).unsqueeze(0)
        result1 = mlib.vis_at_coord(point)
        result2 = mlib.visibility(point)
        assert torch.equal(result1, result2)

    def test_no_nan_no_inf(self, mlib, meta_pair):
        meta_a, meta_b = meta_pair
        coords = torch.cat([
            (meta_a.mins + meta_a.voxel_size / 2.0).unsqueeze(0),
            (meta_b.maxs + 999.0).unsqueeze(0),
        ])
        result = mlib.vis_at_coord(coords)
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


# -- Device --------------------------------------------------

class TestMultiPhotonLibDevice:
    def test_to_device(self, mlib, device):
        mlib = mlib.to(device)
        for lib in mlib.libs:
            assert lib.vis.device.type == device.type

    def test_meta_moves(self, mlib, device):
        mlib = mlib.to(device)
        for lib in mlib.libs:
            for name, buf in lib.meta.named_buffers():
                assert buf.device.type == device.type, (
                    f"buffer '{name}' on {buf.device}, expected {device}"
                )

    def test_bbox_moves(self, mlib, device):
        mlib = mlib.to(device)
        assert mlib.bbox.mins.device.type == device.type
        assert mlib.bbox.maxs.device.type == device.type

    def test_visibility_on_device(self, mlib, meta_pair, device):
        mlib = mlib.to(device)
        meta_a, _ = meta_pair
        point = (meta_a.mins + meta_a.voxel_size / 2.0).unsqueeze(0).to(device)
        result = mlib.vis_at_coord(point)
        assert result.device.type == device.type


# -- Multiple LazyPhotonLib --------------------------------------------------
class TestMultiPhotonLibWithLazy:
    """MultiPhotonLib constructed from LazyPhotonLib instances."""

    def test_construction(self, lazy_plib_pair):
        mlib = MultiPhotonLib(list(lazy_plib_pair))
        assert mlib.n_libs == 2

    def test_properties(self, mlib_lazy, n_pmts_a, n_pmts_b):
        assert mlib_lazy.n_pmts == n_pmts_a + n_pmts_b
        assert mlib_lazy.n_pmts_per_lib == (n_pmts_a, n_pmts_b)

    def test_visibility(self, mlib_lazy, meta_pair, vis_pair, n_pmts_a):
        meta_a, _ = meta_pair
        vis_a, _ = vis_pair
        point = (meta_a.mins + meta_a.voxel_size / 2.0).unsqueeze(0)
        result = mlib_lazy.vis_at_coord(point)
        assert result.shape == (1, mlib_lazy.n_pmts)
        assert torch.allclose(result[:, :n_pmts_a], vis_a[0:1], atol=1e-6)
        assert (result[:, n_pmts_a:] == 0.0).all()

    def test_matches_eager(self, mlib_lazy, mlib, meta_pair):
        """Lazy and eager MultiPhotonLib should return same results."""
        meta_a, meta_b = meta_pair
        pt_a = (meta_a.mins + meta_a.voxel_size / 2.0).unsqueeze(0)
        pt_b = (meta_b.mins + meta_b.voxel_size / 2.0).unsqueeze(0)
        coords = torch.cat([pt_a, pt_b])
        result_lazy = mlib_lazy.vis_at_coord(coords)
        result_eager = mlib.vis_at_coord(coords)
        assert torch.allclose(result_lazy, result_eager, atol=1e-6)

    def test_mixed_lazy_eager(self, plib_pair, lazy_plib_pair):
        """MultiPhotonLib with one eager and one lazy sub-library."""
        eager_a, _ = plib_pair
        _, lazy_b = lazy_plib_pair
        mlib = MultiPhotonLib([eager_a, lazy_b])
        assert mlib.n_libs == 2
