"""Dataset wrapper for training on photon libraries."""

from __future__ import annotations
from typing import Iterator, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .meta import VoxelMeta
from .photonlib import LazyPhotonLib, PhotonLib

class _CollateFn:
    """Picklable collate functor that converts voxel indices to coordinates."""

    def __init__(self, meta: 'VoxelMeta'):
        self.meta = meta

    def __call__(self, batch):
        indices, vis_list = zip(*batch)
        indices = torch.tensor(indices, dtype=torch.long)
        vis = torch.stack(vis_list)
        coord = self.meta.voxel_to_coord(indices)
        return coord, vis

class PhotonLibDataset(Dataset):
    """
    Training dataset backed by a photon library.

    Each sample is a ``(coord, vis)`` pair, where ``coord``
    is the voxel-centre position and ``vis`` is the
    corresponding visibility vector.  The dataset exposes
    every voxel in the library.

    Parameters
    ----------
    plib : PhotonLib or LazyPhotonLib
        Source photon library (held by reference).

    Attributes
    ----------
    plib : PhotonLib or LazyPhotonLib
        Reference to the underlying library.
    meta : VoxelMeta
        Shortcut to ``plib.meta``.

    Examples
    --------
    ::

        plib = PhotonLib.load("photonlib.h5")
        ds = PhotonLibDataset(plib)
        assert len(ds) == len(plib)
        coord, vis = ds[0]

    Prefer :meth:`make_dataloader` for multi-worker
    usage::

        loader = PhotonLibDataset.make_dataloader(
            plib, batch_size=512, shuffle=True,
            num_workers=4,
        )
        for coord_batch, vis_batch in loader:
            ...
    """

    def __init__(
        self,
        plib: Union[PhotonLib, LazyPhotonLib],
    ) -> None:
        super().__init__()
        self.plib = plib
        self.meta: VoxelMeta = plib.meta

    # ---- Dataset protocol ----------------------------------

    def __len__(self) -> int:
        """Number of samples (= number of voxels).

        Returns
        -------
        int
        """
        return len(self.plib)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch a single sample.

        Parameters
        ----------
        idx : int
            Flat voxel index.

        Returns
        -------
        coord : torch.Tensor of shape (ndim,)
            Voxel-centre coordinates.
        vis : torch.Tensor of shape (n_pmts,)
            Visibility vector.
        """
        vis = self.plib[idx]
        return idx, vis

    # ---- multi-worker safety -------------------------------

    @staticmethod
    def _share_memory(
        plib: Union[PhotonLib, LazyPhotonLib],
    ) -> None:
        """Move large tensors to shared memory.

        Called automatically by :meth:`make_dataloader`
        for in-memory :class:`PhotonLib` backends.

        Parameters
        ----------
        plib : PhotonLib or LazyPhotonLib
            The library whose buffers should be shared.
        """

        if isinstance(plib, PhotonLib) and not plib.vis.is_shared():
            plib.vis.share_memory_()


    # ---- factory -------------------------------------------

    @staticmethod
    def make_dataloader(
        plib: Union[PhotonLib, LazyPhotonLib],
        **kwargs,
    ) -> DataLoader:
        """Create a ready-to-use ``DataLoader``.

        Constructs a :class:`PhotonLibDataset` internally
        and applies the appropriate multi-worker safety
        measures depending on the backend:

        *   :class:`PhotonLib`: tensors are moved to
            shared memory.

        Parameters
        ----------
        plib : PhotonLib or LazyPhotonLib
            Source photon library.
        **kwargs
            Forwarded to
            :class:`~torch.utils.data.DataLoader`.
            Common options: ``batch_size``, ``shuffle``,
            ``num_workers``, ``pin_memory``.

        Returns
        -------
        torch.utils.data.DataLoader

        Examples
        --------
        In-memory::

            plib = PhotonLib.load("photonlib.h5")
            loader = PhotonLibDataset.make_dataloader(
                plib, batch_size=512, shuffle=True,
                num_workers=4,
            )

        Lazy (on-disk)::

            plib = LazyPhotonLib("photonlib.h5")
            loader = PhotonLibDataset.make_dataloader(
                plib, batch_size=256, shuffle=True,
                num_workers=4,
            )
        """
        
        # DataLoader with shuffle guarantees unique indices per batch
        if isinstance(plib, LazyPhotonLib):
            plib.assume_unique = True

        ds = PhotonLibDataset(plib)
        num_workers = kwargs.get("num_workers", 0)

        if 'collate_fn' not in kwargs:
            kwargs['collate_fn'] = _CollateFn(ds.meta)

        if num_workers > 0:
            PhotonLibDataset._share_memory(plib)

        return DataLoader(ds, **kwargs)


class PhotonLibSampler(nn.Module):
    """
    GPU-resident sampler mimicking ``DataLoader`` API.

    When the full visibility table fits in GPU memory, this
    class eliminates CPU–GPU transfer overhead by keeping
    the table on device and generating batches entirely on
    the GPU.

    Supports iteration, length query, and epoch-based
    training in the same style as
    :class:`~torch.utils.data.DataLoader`.

    Parameters
    ----------
    plib : PhotonLib
        Source photon library.  Will be moved to *device*.
    batch_size : int, optional
        Number of voxels per batch.  Default is ``1``.
    shuffle : bool, optional
        Whether to shuffle voxel order at the start of
        each iteration pass.  Default is *True*.
    drop_last : bool, optional
        If *True*, drop the last incomplete batch when
        ``n_voxels`` is not divisible by ``batch_size``.
        Default is *False*.
    device : torch.device or str, optional
        Target device.  Defaults to ``"cuda"`` if
        available, otherwise ``"cpu"``.

    Attributes
    ----------
    plib : PhotonLib
        The on-device photon library (submodule).
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle each pass.
    drop_last : bool
        Whether to drop the last incomplete batch.

    Examples
    --------
    Drop-in replacement for ``DataLoader``::

        plib = PhotonLib.load("photonlib.h5")
        loader = PhotonLibSampler(
            plib, batch_size=4096, shuffle=True,
        )

        for epoch in range(n_epochs):
            for coord, vis in loader:
                pred = model(coord)
                loss = criterion(pred, vis)
                ...
    """

    def __init__(
        self,
        plib: PhotonLib,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        device: Optional[
            Union[torch.device, str]
        ] = None,
    ) -> None:
        super().__init__()

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.plib = plib.to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Pre-compute all voxel-centre coordinates.
        all_ids = torch.arange(
            plib.n_voxels,
            dtype=torch.long,
            device=device,
        )
        coords = self.plib.meta.voxel_to_coord(all_ids)
        self.register_buffer("coords", coords)

    # ---- properties ----------------------------------------

    @property
    def n_voxels(self) -> int:
        """Total number of voxels.

        Returns
        -------
        int
        """
        return self.plib.n_voxels

    @property
    def n_pmts(self) -> int:
        """Number of photo-detectors.

        Returns
        -------
        int
        """
        return self.plib.n_pmts

    @property
    def device(self) -> torch.device:
        """Device where the data resides.

        Returns
        -------
        torch.device
        """
        return self.plib.vis.device

    @property
    def dataset(self) -> PhotonLib:
        """Underlying data source.

        Mirrors ``DataLoader.dataset`` for compatibility
        with training utilities that inspect the loader.

        Returns
        -------
        PhotonLib
        """
        return self.plib

    # ---- DataLoader-like interface -------------------------

    def __len__(self) -> int:
        """Number of batches per epoch.

        Follows the same convention as ``DataLoader``:
        the last (possibly smaller) batch is counted
        unless ``drop_last`` is *True*.

        Returns
        -------
        int
        """
        n = self.n_voxels
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(
        self,
    ) -> Iterator[
        tuple[torch.Tensor, torch.Tensor]
    ]:
        """Iterate over one full epoch of batches.

        Yields
        ------
        coord : torch.Tensor of shape (B, ndim)
            Voxel-centre coordinates.
        vis : torch.Tensor of shape (B, n_pmts)
            Visibility vectors.
        """
        if self.shuffle:
            perm = torch.randperm(
                self.n_voxels, device=self.device
            )
        else:
            perm = torch.arange(
                self.n_voxels, device=self.device
            )

        for start in range(
            0, self.n_voxels, self.batch_size
        ):
            idx = perm[start : start + self.batch_size]

            if self.drop_last and (
                len(idx) < self.batch_size
            ):
                break

            yield self.coords[idx], self.plib.vis[idx]
            
    # ---- factory -------------------------------------------


    @classmethod
    def make_dataloader(cls, plib: PhotonLib, **kwargs):
        """Create a DataLoader using PhotonLibSampler.

        This is a convenience factory method that mirrors the interface of
        PhotonLibDataset.make_dataloader. It is a simple wrapper around
        the PhotonLibSampler constructor.

        Parameters
        ----------
        plib : PhotonLib
            PhotonLib instance to sample from.
        **kwargs
            Additional keyword arguments passed to the PhotonLibSampler constructor.

        Returns
        -------
        PhotonLibSampler
            A PhotonLibSampler instance configured as a DataLoader.
        """
        return cls(plib, **kwargs)

    # ---- repr ----------------------------------------------

    def extra_repr(self) -> str:
        """Return a summary for :func:`repr`.

        Returns
        -------
        str
        """
        return (
            f"n_voxels={self.n_voxels}, "
            f"n_pmts={self.n_pmts}, "
            f"batch_size={self.batch_size}, "
            f"shuffle={self.shuffle}, "
            f"drop_last={self.drop_last}, "
            f"device={self.device}"
        )

# -----------------------------------------------------------------------------
# Single factory function
# -----------------------------------------------------------------------------
def create_dataloader(memory: str, photonlib: dict, **kwargs):

    """Factory function to create a dataloader with the appropriate memory
    strategy.

    Parameters
    ----------
    memory : str
        Memory strategy. One of:
        - 'single': Single device (CPU or GPU) with blocking dataloader.
          Uses PhotonLib and PhotonLibSampler.
        - 'share': Shared CPU memory with multi-process loader and pre-fetching.
          Uses PhotonLib and PhotonLibDataset.
        - 'lazy': Lazy loading with shared CPU memory and multi-process loader.
          Uses LazyPhotonLib and PhotonLibDataset.
    photonlib : dict
        Keyword arguments passed to PhotonLib.load or LazyPhotonLib.load.
    **kwargs
        Additional keyword arguments passed to the corresponding
        make_dataloader class method.

    Returns
    -------
    PhotonLibSampler or DataLoader
        A PhotonLibSampler (for 'single') or a DataLoader (for 'share'/'lazy').

    Raises
    ------
    ValueError
        If `memory` is not one of 'single', 'share', or 'lazy'.
    """
    if memory == 'single':
        plib = PhotonLib.load(**photonlib)
        return PhotonLibSampler.make_dataloader(plib, **kwargs)
    elif memory == 'share':
        plib = PhotonLib.load(**photonlib)
        return PhotonLibDataset.make_dataloader(plib, **kwargs)
    elif memory == 'lazy':
        plib = LazyPhotonLib.load(**photonlib)
        return PhotonLibDataset.make_dataloader(plib, **kwargs)
    else:
        raise ValueError(
            f"Unknown memory strategy '{memory}'. "
            f"Expected one of: 'single', 'share', 'lazy'."
        )
