import torch
import pytest

from photonlib.experimental.dataloader import PhotonLibSampler


# ── Construction ────────────────────────────────────────────

class TestSamplerConstruction:
    def test_basic(self, plib):
        sampler = PhotonLibSampler(plib, batch_size=16, device='cpu')
        assert sampler is not None

    def test_default_device(self, plib):
        sampler = PhotonLibSampler(plib, batch_size=16)
        assert sampler.device is not None

    def test_attributes(self, plib):
        sampler = PhotonLibSampler(
            plib, batch_size=8, shuffle=True, drop_last=True, device='cpu',
        )
        assert sampler.batch_size == 8
        assert sampler.shuffle is True
        assert sampler.drop_last is True


# ── Properties ──────────────────────────────────────────────

class TestSamplerProperties:
    def test_n_voxels(self, sampler, meta):
        expected = torch.tensor(meta.voxel_shape).prod().item()
        assert sampler.n_voxels == expected

    def test_n_pmts(self, sampler, n_channels):
        assert sampler.n_pmts == n_channels

    def test_dataset(self, sampler, plib):
        assert sampler.dataset is sampler.plib

    def test_device(self, sampler):
        assert sampler.device == torch.device('cpu')


# ── Length ──────────────────────────────────────────────────

class TestSamplerLength:
    def test_len_no_drop(self, plib, meta):
        bs = 32
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        sampler = PhotonLibSampler(plib, batch_size=bs, drop_last=False, device='cpu')
        expected = -(-n_voxels // bs)  # ceiling division
        assert len(sampler) == expected

    def test_len_drop_last(self, plib, meta):
        bs = 32
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        sampler = PhotonLibSampler(plib, batch_size=bs, drop_last=True, device='cpu')
        expected = n_voxels // bs
        assert len(sampler) == expected

    def test_len_exact_division(self, plib, meta):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        sampler = PhotonLibSampler(plib, batch_size=1, drop_last=False, device='cpu')
        assert len(sampler) == n_voxels

    def test_len_batch_larger_than_data(self, plib, meta):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        sampler = PhotonLibSampler(
            plib, batch_size=n_voxels + 100, drop_last=False, device='cpu',
        )
        assert len(sampler) == 1

    def test_len_batch_larger_drop_last(self, plib, meta):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        sampler = PhotonLibSampler(
            plib, batch_size=n_voxels + 100, drop_last=True, device='cpu',
        )
        assert len(sampler) == 0


# ── Iteration ───────────────────────────────────────────────

class TestSamplerIteration:
    def test_yields_coord_vis_tuples(self, sampler):
        for coord, vis in sampler:
            assert isinstance(coord, torch.Tensor)
            assert isinstance(vis, torch.Tensor)
            break

    def test_batch_shapes(self, sampler, sampler_batch_size, n_channels, ndim):
        for coord, vis in sampler:
            assert coord.shape[1] == ndim
            assert vis.shape[1] == n_channels
            assert coord.shape[0] == vis.shape[0]
            break

    def test_last_batch_size_no_drop(self, plib, meta):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        sampler = PhotonLibSampler(plib, batch_size=32, drop_last=False, device='cpu')
        total = sum(c.shape[0] for c, v in sampler)
        assert total == n_voxels

    def test_last_batch_size_drop_last(self, plib):
        bs = 32
        sampler = PhotonLibSampler(plib, batch_size=bs, drop_last=True, device='cpu')
        for coord, vis in sampler:
            assert coord.shape[0] == bs

    def test_total_voxels_no_drop(self, sampler, meta):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        total = sum(c.shape[0] for c, v in sampler)
        assert total == n_voxels

    def test_num_batches_matches_len(self, sampler):
        batches = list(sampler)
        assert len(batches) == len(sampler)

    def test_multiple_epochs(self, sampler):
        batches_1 = list(sampler)
        batches_2 = list(sampler)
        assert len(batches_1) == len(batches_2)


# ── Shuffle ─────────────────────────────────────────────────

class TestSamplerShuffle:
    def test_no_shuffle_deterministic(self, plib):
        sampler = PhotonLibSampler(plib, batch_size=16, shuffle=False, device='cpu')
        coords_1 = torch.cat([c for c, v in sampler])
        coords_2 = torch.cat([c for c, v in sampler])
        assert torch.equal(coords_1, coords_2)

    def test_shuffle_changes_order(self, plib):
        sampler = PhotonLibSampler(plib, batch_size=16, shuffle=True, device='cpu')
        coords_1 = torch.cat([c for c, v in sampler])
        all_same = True
        for _ in range(3):
            c = torch.cat([c for c, v in sampler])
            if not torch.equal(c, coords_1):
                all_same = False
                break
        assert not all_same, "Shuffle did not change order across multiple epochs"


# ── Data Correctness ────────────────────────────────────────

class TestSamplerDataCorrectness:
    def test_coords_inside_volume(self, sampler, meta):
        for coord, vis in sampler:
            assert meta.contain(coord).all()
            break

    def test_vis_matches_plib(self, plib, vis):
        sampler = PhotonLibSampler(plib, batch_size=64, shuffle=False, device='cpu')
        for coord, vis_batch in sampler:
            expected = plib.visibility(coord)
            assert torch.allclose(vis_batch, expected, atol=1e-6)
            break

    def test_all_voxels_covered_no_shuffle(self, plib, meta, vis):
        sampler = PhotonLibSampler(
            plib, batch_size=64, shuffle=False, drop_last=False, device='cpu',
        )
        all_vis = torch.cat([v for c, v in sampler])
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        assert all_vis.shape[0] == n_voxels

    def test_no_nan_no_inf(self, sampler):
        for coord, vis in sampler:
            assert not torch.isnan(coord).any()
            assert not torch.isnan(vis).any()
            assert not torch.isinf(coord).any()
            assert not torch.isinf(vis).any()
            break


# ── Device ──────────────────────────────────────────────────

class TestSamplerDevice:
    def test_cpu(self, plib):
        sampler = PhotonLibSampler(plib, batch_size=16, device='cpu')
        assert sampler.device == torch.device('cpu')
        for coord, vis in sampler:
            assert coord.device == torch.device('cpu')
            assert vis.device == torch.device('cpu')
            break

    def test_gpu(self, plib):
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available')
        sampler = PhotonLibSampler(plib, batch_size=16, device='cuda')
        assert sampler.device.type == 'cuda'
        for coord, vis in sampler:
            assert coord.device.type == 'cuda'
            assert vis.device.type == 'cuda'
            break

    def test_plib_moved_to_device(self, plib):
        if not torch.cuda.is_available():
            pytest.skip('CUDA not available')
        sampler = PhotonLibSampler(plib, batch_size=16, device='cuda')
        assert sampler.plib.vis.device.type == 'cuda'
