import torch
import pytest
from torch.utils.data import DataLoader

from photonlib.experimental import (
    PhotonLib, LazyPhotonLib, PhotonLibDataset, PhotonLibSampler,
	create_dataloader,
)
from photonlib.experimental.dataloader import _CollateFn


# ---- Construction --------------------------------------------

class TestDatasetConstruction:
    def test_from_photonlib(self, plib):
        ds = PhotonLibDataset(plib)
        assert ds is not None

    def test_attributes(self, dataset, plib):
        assert dataset.plib is plib
        assert dataset.meta is plib.meta


# ---- Length --------------------------------------------------

class TestDatasetLength:
    def test_len(self, dataset, meta):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        assert len(dataset) == n_voxels


# ---- __getitem__ ---------------------------------------------

class TestDatasetGetItem:
    def test_returns_tuple(self, dataset):
        idx, vis = dataset[0]
        assert isinstance(idx, int)
        assert isinstance(vis, torch.Tensor)

    def test_idx_passthrough(self, dataset):
        for i in [0, 1, 5]:
            idx, _ = dataset[i]
            assert idx == i

    def test_vis_shape(self, dataset, n_channels):
        _, vis = dataset[0]
        assert vis.shape == (n_channels,)

    def test_first_voxel(self, dataset, vis):
        _, vis_out = dataset[0]
        assert torch.allclose(vis_out, vis[0], atol=1e-6)

    def test_last_voxel(self, dataset, meta, vis):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        _, vis_out = dataset[n_voxels - 1]
        assert torch.allclose(vis_out, vis[-1], atol=1e-6)

    def test_random_voxels(self, dataset, vis, meta):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        n_test = min(50, n_voxels)
        for i in torch.randint(0, n_voxels, (n_test,)).tolist():
            _, vis_out = dataset[i]
            assert torch.allclose(vis_out, vis[i], atol=1e-6)

    def test_no_nan_no_inf(self, dataset):
        _, vis = dataset[0]
        assert not torch.isnan(vis).any()
        assert not torch.isinf(vis).any()


# ---- collate_fn ----------------------------------------------

class TestDatasetCollateFn:
    def test_returns_coord_vis(self, dataset, ndim, n_channels):
        collate = _CollateFn(dataset.meta)
        batch = [dataset[i] for i in range(4)]
        coord, vis = collate(batch)
        assert coord.shape == (4, ndim)
        assert vis.shape == (4, n_channels)

    def test_coord_matches_voxel_to_coord(self, dataset, meta):
        collate = _CollateFn(meta)
        batch = [dataset[i] for i in range(4)]
        coord, _ = collate(batch)
        indices = torch.arange(4, dtype=torch.long)
        expected = meta.voxel_to_coord(indices)
        assert torch.allclose(coord, expected, atol=1e-6)

    def test_vis_matches_source(self, dataset, vis):
        collate = _CollateFn(dataset.meta)
        batch = [dataset[i] for i in range(4)]
        _, vis_out = collate(batch)
        expected = vis[:4]
        assert torch.allclose(vis_out, expected, atol=1e-6)

    def test_coords_inside_volume(self, dataset, meta):
        collate = _CollateFn(meta)
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        n_test = min(50, n_voxels)
        batch = [dataset[i] for i in range(n_test)]
        coord, _ = collate(batch)
        assert meta.contain(coord).all()

    def test_picklable(self, dataset):
        """collate_fn must be picklable for num_workers > 0."""
        import pickle
        collate = _CollateFn(dataset.meta)
        restored = pickle.loads(pickle.dumps(collate))
        batch = [dataset[i] for i in range(4)]
        coord1, vis1 = collate(batch)
        coord2, vis2 = restored(batch)
        assert torch.equal(coord1, coord2)
        assert torch.equal(vis1, vis2)


# ---- Raw DataLoader (default collation) ----------------------

class TestDatasetRawDataLoader:
    def test_default_collation_yields_idx_vis(self, dataset):
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        for idx_batch, vis_batch in loader:
            assert idx_batch.dtype in (torch.long, torch.int64)
            assert vis_batch.ndim == 2
            break

    def test_full_epoch(self, dataset, meta):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
        total = sum(v.shape[0] for _, v in loader)
        assert total == n_voxels


# ---- make_dataloader -----------------------------------------

class TestMakeDataLoader:
    def test_returns_dataloader(self, plib):
        loader = PhotonLibDataset.make_dataloader(plib, batch_size=16)
        assert isinstance(loader, DataLoader)

    def test_yields_coord_vis(self, plib, ndim, n_channels):
        loader = PhotonLibDataset.make_dataloader(plib, batch_size=16, shuffle=False)
        for coord, vis in loader:
            assert coord.shape == (16, ndim)
            assert vis.shape == (16, n_channels)
            break

    def test_coords_inside_volume(self, plib, meta):
        loader = PhotonLibDataset.make_dataloader(plib, batch_size=64, shuffle=False)
        for coord, vis in loader:
            assert meta.contain(coord).all()
            break

    def test_vis_matches_plib(self, plib):
        loader = PhotonLibDataset.make_dataloader(plib, batch_size=64, shuffle=False)
        for coord, vis in loader:
            expected = plib.visibility(coord)
            assert torch.allclose(vis, expected, atol=1e-6)
            break

    def test_full_epoch(self, plib, meta):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        loader = PhotonLibDataset.make_dataloader(
            plib, batch_size=64, shuffle=False, drop_last=False,
        )
        total = sum(c.shape[0] for c, v in loader)
        assert total == n_voxels

    def test_drop_last(self, plib):
        loader = PhotonLibDataset.make_dataloader(
            plib, batch_size=32, shuffle=False, drop_last=True,
        )
        for coord, vis in loader:
            assert coord.shape[0] == 32

    def test_shuffle(self, plib):
        loader = PhotonLibDataset.make_dataloader(plib, batch_size=16, shuffle=True)
        batches = list(loader)
        assert len(batches) > 0

    def test_dataset_attribute(self, plib):
        loader = PhotonLibDataset.make_dataloader(plib, batch_size=16)
        assert isinstance(loader.dataset, PhotonLibDataset)
        assert loader.dataset.plib is plib

    def test_num_workers_zero(self, plib):
        loader = PhotonLibDataset.make_dataloader(plib, batch_size=16, num_workers=0)
        for coord, vis in loader:
            assert coord is not None
            break

    def test_custom_collate_not_overridden(self, plib):
        custom = lambda batch: batch
        loader = PhotonLibDataset.make_dataloader(
            plib, batch_size=4, collate_fn=custom,
        )
        assert loader.collate_fn is custom

    def test_no_nan_no_inf(self, plib):
        loader = PhotonLibDataset.make_dataloader(plib, batch_size=64, shuffle=False)
        for coord, vis in loader:
            assert not torch.isnan(coord).any()
            assert not torch.isnan(vis).any()
            assert not torch.isinf(coord).any()
            assert not torch.isinf(vis).any()
            break

    def test_shared_memory_set(self, plib):
        """make_dataloader should move vis to shared memory."""
        PhotonLibDataset.make_dataloader(plib, batch_size=16, num_workers=2)
        assert plib.vis.is_shared()


# ----- Multi-Worker --------------------------------------------

class TestDatasetMultiWorker:
    def test_num_workers_photonlib(self, plib):
        loader = PhotonLibDataset.make_dataloader(
            plib, batch_size=16, num_workers=2,
        )
        for coord, vis in loader:
            assert coord is not None
            break

    def test_num_workers_full_epoch(self, plib, meta):
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        loader = PhotonLibDataset.make_dataloader(
            plib, batch_size=64, shuffle=False, drop_last=False, num_workers=2,
        )
        total = sum(c.shape[0] for c, v in loader)
        assert total == n_voxels

    def test_num_workers_lazy(self, plib_h5):
        """LazyPhotonLib with num_workers > 0 — tests fork/pickle safety."""
        filepath, _, _ = plib_h5
        lplib = LazyPhotonLib(filepath)
        loader = PhotonLibDataset.make_dataloader(
            lplib, batch_size=16, num_workers=2,
        )
        for coord, vis in loader:
            assert coord is not None
            assert not torch.isnan(vis).any()
            break

    def test_num_workers_lazy_full_epoch(self, plib_h5):
        filepath, meta, vis = plib_h5
        n_voxels = torch.tensor(meta.voxel_shape).prod().item()
        lplib = LazyPhotonLib(filepath)
        loader = PhotonLibDataset.make_dataloader(
            lplib, batch_size=64, shuffle=False, drop_last=False, num_workers=2,
        )
        total = sum(c.shape[0] for c, v in loader)
        assert total == n_voxels

    def test_num_workers_lazy_correctness(self, plib_h5):
        """Verify lazy multi-worker results match in-memory PhotonLib."""
        filepath, meta, vis = plib_h5
        lplib = LazyPhotonLib(filepath)
        plib = PhotonLib(meta, vis)

        loader_lazy = PhotonLibDataset.make_dataloader(
            lplib, batch_size=64, shuffle=False, drop_last=False, num_workers=2,
        )
        loader_eager = PhotonLibDataset.make_dataloader(
            plib, batch_size=64, shuffle=False, drop_last=False, num_workers=0,
        )

        for (coord_l, vis_l), (coord_e, vis_e) in zip(loader_lazy, loader_eager):
            assert torch.allclose(coord_l, coord_e, atol=1e-6)
            assert torch.allclose(vis_l, vis_e, atol=1e-6)
            break


# ---- Consistency with PhotonLibSampler -----------------------

class TestDatasetSamplerConsistency:
    def test_matches_sampler_first_batch(self, plib):
        loader = PhotonLibDataset.make_dataloader(
            plib, batch_size=64, shuffle=False, drop_last=False,
        )
        sampler = PhotonLibSampler(
            plib, batch_size=64, shuffle=False, drop_last=False, device='cpu',
        )

        for (coord_dl, vis_dl), (coord_s, vis_s) in zip(loader, sampler):
            assert torch.allclose(coord_dl, coord_s, atol=1e-6)
            assert torch.allclose(vis_dl, vis_s, atol=1e-6)
            break

    def test_matches_sampler_full_epoch(self, plib):
        loader = PhotonLibDataset.make_dataloader(
            plib, batch_size=64, shuffle=False, drop_last=False,
        )
        sampler = PhotonLibSampler(
            plib, batch_size=64, shuffle=False, drop_last=False, device='cpu',
        )

        for (coord_dl, vis_dl), (coord_s, vis_s) in zip(loader, sampler):
            assert torch.allclose(coord_dl, coord_s, atol=1e-6)
            assert torch.allclose(vis_dl, vis_s, atol=1e-6)

    def test_make_dataloader_returns_sampler(self, plib):
        """make_dataloader should return a PhotonLibSampler instance."""
        loader = PhotonLibSampler.make_dataloader(
            plib, batch_size=32, device='cpu'
        )
        assert isinstance(loader, PhotonLibSampler)


# ---- Dataloader factory -----------------------------------------------------
class TestCreateDataloader:

    def test_single(self, plib_cfg):
        loader = create_dataloader("single", plib_cfg, batch_size=32)
        assert isinstance(loader, PhotonLibSampler)
        assert isinstance(loader.plib, PhotonLib)
        assert loader.batch_size == 32

    def test_share(self, plib_cfg):
        loader = create_dataloader("share", plib_cfg, batch_size=32)
        assert isinstance(loader, DataLoader)
        assert isinstance(loader.dataset.plib, PhotonLib)
        assert loader.batch_size == 32

    def test_lazy(self, plib_cfg):
        loader = create_dataloader("lazy", plib_cfg, batch_size=32)
        assert isinstance(loader, DataLoader)
        assert isinstance(loader.dataset.plib, LazyPhotonLib)
        assert loader.batch_size == 32

    def test_invalid(self, plib_cfg):
        with pytest.raises(ValueError, match="Unknown memory strategy"):
            create_dataloader("invalid", plib_cfg)
