import torch
import pytest

from photonlib.experimental import AABox, VoxelMeta, PhotonLib
from photonlib.experimental.multilib import MultiPhotonLib

def pytest_collection_modifyitems(items):
    for item in items:
        if '/experimental/' in str(item.fspath):
            item.add_marker(pytest.mark.experimental)


# -- Shared fixtures -----------------------------------------

@pytest.fixture(params=[2, 3], ids=['2d', '3d'])
def ndim(request):
    return request.param


@pytest.fixture(params=['cpu', 'cuda'], ids=['cpu', 'gpu'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA not available')
    return torch.device(request.param)



# -- Single-volume fixtures ----------------------------------

@pytest.fixture
def ranges(ndim):
    if ndim == 2:
        return torch.tensor([
            [-100.0, 100.0],
            [-200.0, 200.0],
        ])
    return torch.tensor([
        [-100.0, 100.0],
        [-200.0, 200.0],
        [   0.0, 500.0],
    ])


@pytest.fixture
def shape(ndim):
    if ndim == 2:
        return torch.tensor([20, 40], dtype=torch.long)
    return torch.tensor([20, 40, 50], dtype=torch.long)


@pytest.fixture
def meta(ranges, shape):
    return VoxelMeta(ranges, shape)


@pytest.fixture
def n_channels():
    return 10

@pytest.fixture
def vis(meta, n_channels):
    n_voxels = torch.tensor(meta.voxel_shape).prod().item()
    return torch.rand(n_voxels, n_channels)

@pytest.fixture
def plib(meta, vis):
    return PhotonLib(meta, vis)


# -- HDF5 file fixtures (for LazyPhotonLib) ------------------

@pytest.fixture
def plib_h5(tmp_path, meta, vis):
    """Write a single PhotonLib to HDF5 and return (path, meta, vis)."""
    import h5py
    filepath = str(tmp_path / 'photonlib.h5')
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('vis', data=vis.numpy())
        f.create_dataset('min', data=meta.mins.numpy())
        f.create_dataset('max', data=meta.maxs.numpy())
        f.create_dataset('numvox', data=list(meta.voxel_shape))
    return filepath, meta, vis


@pytest.fixture
def plib_h5_grouped(tmp_path, meta_pair, vis_pair):
    """Write two PhotonLibs into HDF5 groups and return (path, metas, vises)."""
    import h5py
    meta_a, meta_b = meta_pair
    vis_a, vis_b = vis_pair

    filepath = str(tmp_path / 'grouped.h5')
    with h5py.File(filepath, 'w') as f:
        for key, m, v in [('vol_a', meta_a, vis_a), ('vol_b', meta_b, vis_b)]:
            g = f.create_group(key)
            g.create_dataset('vis', data=v.numpy())
            g.create_dataset('min', data=m.mins.numpy())
            g.create_dataset('max', data=m.maxs.numpy())
            g.create_dataset('numvox', data=list(m.voxel_shape))

    return filepath, (meta_a, meta_b), (vis_a, vis_b)


# -- Multi-volume fixtures ----------------------------------

@pytest.fixture
def n_pmts_a():
    return 10


@pytest.fixture
def n_pmts_b():
    return 6


@pytest.fixture
def meta_pair(ndim):
    """Two non-overlapping VoxelMeta volumes split along first axis."""
    if ndim == 2:
        ranges_a = torch.tensor([
            [-100.0, 0.0],
            [-200.0, 200.0],
        ])
        ranges_b = torch.tensor([
            [0.0, 100.0],
            [-200.0, 200.0],
        ])
        shape = torch.tensor([10, 40], dtype=torch.long)
    else:
        ranges_a = torch.tensor([
            [-100.0, 0.0],
            [-200.0, 200.0],
            [0.0, 500.0],
        ])
        ranges_b = torch.tensor([
            [0.0, 100.0],
            [-200.0, 200.0],
            [0.0, 500.0],
        ])
        shape = torch.tensor([10, 40, 50], dtype=torch.long)

    meta_a = VoxelMeta(ranges_a, shape)
    meta_b = VoxelMeta(ranges_b, shape)
    return meta_a, meta_b


@pytest.fixture
def vis_pair(meta_pair, n_pmts_a, n_pmts_b):
    meta_a, meta_b = meta_pair
    n_vox_a = torch.tensor(meta_a.voxel_shape).prod().item()
    n_vox_b = torch.tensor(meta_b.voxel_shape).prod().item()
    vis_a = torch.rand(n_vox_a, n_pmts_a)
    vis_b = torch.rand(n_vox_b, n_pmts_b)
    return vis_a, vis_b


@pytest.fixture
def plib_pair(meta_pair, vis_pair):
    meta_a, meta_b = meta_pair
    vis_a, vis_b = vis_pair
    return PhotonLib(meta_a, vis_a), PhotonLib(meta_b, vis_b)


@pytest.fixture
def mlib(plib_pair):
    return MultiPhotonLib(list(plib_pair))


@pytest.fixture
def lazy_plib_pair(plib_h5_grouped):
    """Two LazyPhotonLib instances from grouped HDF5."""
    from photonlib.experimental.photonlib import LazyPhotonLib
    filepath, _, _ = plib_h5_grouped
    a = LazyPhotonLib(filepath, group='vol_a')
    b = LazyPhotonLib(filepath, group='vol_b')
    yield a, b
    a.close()
    b.close()


@pytest.fixture
def mlib_lazy(lazy_plib_pair):
    from photonlib.experimental.multilib import MultiPhotonLib
    return MultiPhotonLib(list(lazy_plib_pair))


# -- Sampler fixtures ----------------------------------------

@pytest.fixture
def sampler_batch_size():
    return 32


@pytest.fixture
def sampler(plib, sampler_batch_size):
    from photonlib.experimental.dataloader import PhotonLibSampler
    return PhotonLibSampler(
        plib, batch_size=sampler_batch_size, shuffle=False, device='cpu',
    )


# ── Dataset fixtures ────────────────────────────────────────

@pytest.fixture
def dataset(plib):
    from photonlib.experimental.dataloader import PhotonLibDataset
    return PhotonLibDataset(plib)
