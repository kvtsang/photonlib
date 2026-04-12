"""Photon visibility library with on-disk random access."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
import weakref

from .meta import VoxelMeta

class PhotonLib(nn.Module):
    """
    Photon visibility lookup table on a voxelised volume.

    The visibility table stores the detection probability for
    each (voxel, PMT) pair.  The table and the voxel metadata
    are registered as buffers / submodules so that the whole
    object follows standard :class:`torch.nn.Module` semantics
    (device placement, ``state_dict``, ``repr``, etc.).

    Parameters
    ----------
    meta : VoxelMeta
        Voxel metadata describing the spatial grid.
    vis : array_like of shape (n_voxels, n_pmts)
        Visibility (detection probability) table.

    Attributes
    ----------
    meta : VoxelMeta
        Voxel metadata (registered submodule).
    vis : torch.Tensor of shape (n_voxels, n_pmts)
        Visibility table (registered buffer).

    Raises
    ------
    ValueError
        If the leading dimension of ``vis`` does not match
        ``meta.n_voxels``, or if ``vis`` is not 2-D.

    Examples
    --------
    >>> meta = VoxelMeta(
    ...     [[0, 10], [0, 20], [0, 30]], [5, 10, 15]
    ... )
    >>> vis = torch.rand(meta.n_voxels, 180)
    >>> plib = PhotonLib(meta, vis)
    >>> plib.n_pmts
    180
    >>> len(plib)
    750
    """

    def __init__(
        self,
        meta: VoxelMeta,
        vis: Union[torch.Tensor, np.ndarray],
    ) -> None:
        super().__init__()

        vis_t = torch.as_tensor(
            vis, dtype=torch.float32
        )
        if vis_t.ndim != 2:
            raise ValueError(
                "vis must be 2-D (n_voxels, n_pmts), "
                f"got ndim={vis_t.ndim}"
            )
        if vis_t.shape[0] != meta.n_voxels:
            raise ValueError(
                f"Leading dim of vis "
                f"({vis_t.shape[0]}) "
                f"!= meta.n_voxels "
                f"({meta.n_voxels})"
            )

        self.meta = meta
        self.register_buffer("vis", vis_t)

    # ---- properties ----------------------------------------

    @property
    def n_voxels(self) -> int:
        """
        Total number of voxels.

        Returns
        -------
        int
        """
        return self.meta.n_voxels

    @property
    def n_pmts(self) -> int:
        """
        Number of photo-detectors (PMTs).

        Returns
        -------
        int
        """
        return self.vis.shape[1]

    # ---- lookup --------------------------------------------

    def __len__(self) -> int:
        """
        Return the total number of voxels.

        Returns
        -------
        int
        """
        return self.n_voxels

    def __getitem__(
        self,
        voxel_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Look up visibility by flat voxel index.

        Parameters
        ----------
        voxel_ids : torch.LongTensor of shape (...)
            Flat (ravelled, C-order) voxel indices.

        Returns
        -------
        torch.Tensor of shape (..., n_pmts)
            Visibility values for the requested voxels.
        """
        return self.vis[voxel_ids]

    def vis_at_voxel(
        self,
        voxel_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Look up visibility by flat voxel index.

        Equivalent to ``self[voxel_ids]``.

        Parameters
        ----------
        voxel_ids : torch.LongTensor of shape (...)
            Flat (ravelled, C-order) voxel indices.

        Returns
        -------
        torch.Tensor of shape (..., n_pmts)
            Visibility values for the requested voxels.

        See Also
        --------
        vis_at_coord : Look up by spatial coordinates.
        """
        return self[voxel_ids]

    def vis_at_coord(
        self,
        pts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Look up visibility by spatial coordinates.

        Coordinates are mapped to voxel indices via
        :meth:`VoxelMeta.coord_to_voxel` and then used to
        index the visibility table.  Points outside the box
        are clamped to the nearest boundary voxel; use
        :meth:`VoxelMeta.contain` beforehand if filtering
        is required.

        Parameters
        ----------
        pts : torch.Tensor of shape (..., ndim)
            Spatial coordinates.

        Returns
        -------
        torch.Tensor of shape (..., n_pmts)
            Visibility values for the voxels containing
            the query points.

        See Also
        --------
        vis_at_voxel : Look up by flat voxel index.
        """
        voxel_ids = self.meta.coord_to_voxel(pts)
        return self[voxel_ids]

    def visibility(
        self,
        pts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Backward compatible function for `vis_at_coord`.
        """

        return self.vis_at_coord(pts)

    
    # ---- class factory ------------------------------------------

    @classmethod
    def load(
        cls,
        file: Union[h5py.File, h5py.Group, str],
        group: Optional[str] = None,
    ) -> "PhotonLib":
        """
        Construct a :class:`PhotonLib` from an HDF5 source.

        The entire visibility table is read into memory.  For
        on-disk random access without loading the full table,
        see :class:`LazyPhotonLib`.

        Parameters
        ----------
        file : h5py.File, h5py.Group, or str
            An open HDF5 handle or a file path.
        group : str, optional
            Group key that contains the photon library data.
            *None* (default) uses the root group.

        Returns
        -------
        PhotonLib

        Notes
        -----
        Expected HDF5 layout (attributes **or** datasets)::

            [group/]
                vis      float[n_voxels, n_pmts]
                min      float[ndim]
                max      float[ndim]
                numvox   int[ndim]
        """
        if isinstance(file, str):
            with h5py.File(file, "r") as f:
                return cls.load(f, group=group)

        grp = (
            file if group is None else file[group]
        )
        meta = VoxelMeta.load(grp)
        vis = _read_dataset(grp, "vis", np.float32)
        return cls(meta, vis)


    # ---- repr ----------------------------------------------

    def extra_repr(self) -> str:
        """
        Return a summary string for :func:`repr`.

        Returns
        -------
        str
        """
        return (
            f"n_voxels={self.n_voxels}, "
            f"n_pmts={self.n_pmts}"
        )


class LazyPhotonLib(nn.Module):
    """
    On-disk photon library with per-query HDF5 reads.

    Unlike :class:`PhotonLib`, which loads the full visibility
    table into memory, this class keeps the HDF5 file open and
    reads only the requested rows on each lookup.  This makes
    it suitable for libraries that are too large to fit in
    memory.

    An optional ``reduce_map`` can be provided to coalesce
    duplicate voxel reads.  When many query points fall into
    the same voxel, only the unique voxels are read from disk
    and the results are expanded back to the original shape.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    group : str, optional
        Group key inside the file.  *None* (default) uses the
        root group.

    Attributes
    ----------
    meta : VoxelMeta
        Voxel metadata (loaded at construction).
    filepath : str
        Path to the source HDF5 file.
    group : str or None
        Group key used for reading.

    Notes
    -----
    The HDF5 file is opened in read-only mode at construction
    and remains open for the lifetime of the object.  Call
    :meth:`close` or use the context manager protocol to
    release the handle.

    Examples
    --------
    >>> with LazyPhotonLib("photonlib.h5") as plib:
    ...     v = plib[torch.tensor([0, 1, 2])]
    """

    #: Default deduplication policy for ``__getitem__``.
    #: Set to *True* to skip deduplication when caller
    #: guarantees unique indices (e.g., batch sampler).
    assume_unique: bool = False

    def __init__(
        self,
        filepath: str,
        group: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.filepath: str = filepath
        self.group: Optional[str] = group

        # file handle
        self._h5 = None
        self._vis_det = None

        # open hdf5 file once to get meta data
        grp = self._grp
        self.meta = VoxelMeta.load(grp)
        self._n_pmts = self.vis_dset.shape[1]

        # close file handle
        # will be opened again for lazy-lading
        self.close()


    # ---- internal helpers ----------------------------------

    @property
    def _grp(self) -> Union[h5py.File, h5py.Group]:
        """
        Resolve the target group in the open file.

        Returns
        -------
        h5py.File or h5py.Group
        """
        f = self._open()
        if self.group is None:
            return f
        return f[self.group]


    # ---- file handling ------------------------------------

    def _open(self) -> None:
        """ Open file on-demand. Return existing handle if opened """
        if self.opened:
            return self._h5

        self._h5 = h5py.File(self.filepath, 'r')
        self._finalizer = weakref.finalize(
            self, self._cleanup, self._h5
        )
        self._vis_dset = self._grp['vis']
        return self._h5
    
    @staticmethod
    def _cleanup(file):
        """ Close file if opened. Safe to call multiple times."""
        if file.id.valid:
            file.close()

    def close(self):
        """ Explicitly close the HDF5 file."""
        if hasattr(self, '_finalizer'):
            self._finalizer()
        self._vis_dset = None
        self._h5 = None


    # ---- pickle --------------------------------------------

    def __getstate__(self):
        """ Explicitly close file handle before pickling. """
        self.close()
        return self.__dict__


    # ---- properties ----------------------------------------

    @property
    def opened(self):
        """ Check if the file is currently open and valid. """
        if self._h5 is not None and self._h5.id.valid:
            return True
        return False

    @property
    def vis_dset(self):
        if self._vis_dset is None:
            self._open()
        return self._vis_dset

    @property
    def n_voxels(self) -> int:
        """
        Total number of voxels.

        Returns
        -------
        int
        """
        return self.meta.n_voxels

    @property
    def n_pmts(self) -> int:
        """
        Number of photo-detectors (PMTs).

        Returns
        -------
        int
        """
        return self._n_pmts


    # ---- disk read -----------------------------------------

    def _read_rows_raw(self, sorted_ids: 'torch.Tensor') -> 'torch.Tensor':
        """
        Read rows from HDF5 by sorted flat indices.
        """
        # your existing h5py read logic here, e.g.:
        ids_np = sorted_ids.numpy()
        vis_np = self.vis_dset[ids_np]
        return torch.from_numpy(vis_np)

    def _read_rows(self, voxel_ids: 'torch.Tensor') -> 'torch.Tensor':
        """
        Read rows with deduplication and reordering.

        Parameters
        ----------
        voxel_ids : torch.Tensor of shape (...)

        Returns
        -------
        torch.Tensor of shape (..., n_pmts)
        """
        orig_shape = voxel_ids.shape
        flat = voxel_ids.ravel()
        unique, inverse = torch.unique(flat, sorted=True, return_inverse=True)
        vis_unique = self._read_rows_raw(unique)
        vis = vis_unique[inverse]
        return vis.reshape(*orig_shape, -1)

    def _read_rows_unique(self, voxel_ids: 'torch.Tensor') -> 'torch.Tensor':
        """
        Read rows assuming all indices are unique.

        Sorts for efficient HDF5 access, then restores
        original order.

        Parameters
        ----------
        voxel_ids : torch.Tensor of shape (...)

        Returns
        -------
        torch.Tensor of shape (..., n_pmts)
        """
        orig_shape = voxel_ids.shape
        flat = voxel_ids.ravel()
        sorted_ids, sort_idx = flat.sort()
        vis_sorted = self._read_rows_raw(sorted_ids)
        vis = torch.empty_like(vis_sorted)
        vis[sort_idx] = vis_sorted
        return vis.reshape(*orig_shape, -1)


    # ---- lookup (mirrors PhotonLib API) --------------------

    def __len__(self) -> int:
        """
        Return the total number of voxels.

        Returns
        -------
        int
        """
        return self.n_voxels

    def __getitem__(self, voxel_ids) -> 'torch.Tensor':
        """
        Look up visibility by flat voxel index.

        Uses :attr:`assume_unique` to decide whether to
        deduplicate.

        Parameters
        ----------
        voxel_ids : int, list, or torch.LongTensor

        Returns
        -------
        torch.Tensor of shape (..., n_pmts)
        """
        squeeze = isinstance(voxel_ids, int)

        if not isinstance(voxel_ids, torch.Tensor):
            voxel_ids = torch.tensor(
                [voxel_ids] if squeeze else voxel_ids,
                dtype=torch.long,
            )

        vis = self.vis_at_voxel(voxel_ids)

        return vis.squeeze(0) if squeeze else vis

    def vis_at_voxel(
        self,
        voxel_ids: 'torch.Tensor',
        unique: 'Optional[bool]' = None,
    ) -> 'torch.Tensor':
        """
        Look up visibility by flat voxel index.

        Parameters
        ----------
        voxel_ids : torch.LongTensor of shape (...)
            Flat voxel indices.
        unique : bool, optional
            If *True*, skip deduplication.
            If *False*, deduplicate.
            If *None* (default), use :attr:`assume_unique`.

        Returns
        -------
        torch.Tensor of shape (..., n_pmts)
        """
        if unique is None:
            unique = self.assume_unique

        if unique:
            return self._read_rows_unique(voxel_ids)
        return self._read_rows(voxel_ids)

    def vis_at_coord(
        self,
        pts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Look up visibility by spatial coordinates.

        Parameters
        ----------
        pts : torch.Tensor of shape (..., ndim)

        Returns
        -------
        torch.Tensor of shape (..., n_pmts)

        See Also
        --------
        vis_at_voxel : Look up by flat voxel index.
        """
        voxel_ids = self.meta.coord_to_voxel(pts)
        return self[voxel_ids]

    def visibility(
        self,
        pts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Backward compatible function for `vis_at_coord`.
        """

        return self.vis_at_coord(pts)

    # ---- fullly load vis -----------------------------------------

    def materialize(self) -> PhotonLib:
        """
        Read the full table and return a :class:`PhotonLib`.

        This is a convenience method for switching from
        lazy to in-memory access.

        Returns
        -------
        PhotonLib
        """
        vis = np.asarray(
            self.vis_dset, dtype=np.float32
        )
        return PhotonLib(self.meta, vis)


    # ---- class factory ------------------------------------------

    @classmethod
    def load(
        cls,
        file: Union[h5py.File, h5py.Group, str],
        group: Optional[str] = None,
    ) -> "LazyPhotonLib":
        """ Class factory. Same interface as PhotonLib. """
        return cls(file, group)


    # ---- repr ----------------------------------------------

    def extra_repr(self) -> str:
        """
        Return a summary string for :func:`repr`.

        Returns
        -------
        str
        """
        return (
            f"filepath='{self.filepath}', "
            f"n_voxels={self.n_voxels}, "
            f"n_pmts={self.n_pmts}"
        )


# ---- module-level helpers ----------------------------------


def _read_dataset(
    grp: Union[h5py.File, h5py.Group],
    key: str,
    dtype,
) -> np.ndarray:
    """
    Read a key from an HDF5 group.

    Tries ``grp[key]`` (dataset) first, then falls back to
    ``grp.attrs[key]`` for backward compatibility.

    Parameters
    ----------
    grp : h5py.File or h5py.Group
        Source group.
    key : str
        Dataset or attribute name.
    dtype : numpy dtype
        Desired output dtype.

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    KeyError
        If *key* is found neither as a dataset nor as an
        attribute.
    """
    if key in grp:
        return np.asarray(grp[key], dtype=dtype)
    if key in grp.attrs:
        return np.asarray(
            grp.attrs[key], dtype=dtype
        )
    raise KeyError(
        f"'{key}' not found in {grp.name}"
    )

