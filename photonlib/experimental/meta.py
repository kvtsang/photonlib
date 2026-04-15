"""Axis-aligned box and voxel metadata as torch.nn.Module."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import h5py
import numpy as np
import torch
import torch.nn as nn


class AABox(nn.Module):
    """
    Axis-aligned bounding box defined by two corners.

    The box is defined by ``ranges`` of shape ``(ndim, 2)`` where
    each row holds ``[low, high]`` for the corresponding axis.

    The corners are registered as **buffers** (not parameters), so
    they automatically follow the module's device / dtype, are
    included in ``state_dict``, but are **not** learnable.

    Parameters
    ----------
    ranges : array_like of shape (ndim, 2)
        Bounding box specification.  Each row is ``[low, high]``
        for the corresponding spatial axis.

    Attributes
    ----------
    ranges : torch.Tensor of shape (ndim, 2)
        Registered buffer holding the bounding box limits.

    Raises
    ------
    ValueError
        If ``ranges`` does not have shape ``(ndim, 2)`` or if any
        ``low >= high``.

    Examples
    --------
    >>> box = AABox([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    >>> box.ndim
    3
    >>> box.contain(torch.tensor([[0.5, 1.0, 1.5]]))
    tensor([True])
    """

    def __init__(
        self,
        ranges: Union[torch.Tensor, np.ndarray, Sequence],
    ) -> None:
        super().__init__()
        ranges_t = torch.as_tensor(ranges, dtype=torch.float32)
        if ranges_t.ndim != 2 or ranges_t.shape[1] != 2:
            raise ValueError(
                "ranges must have shape (ndim, 2), "
                f"got {tuple(ranges_t.shape)}"
            )
        if torch.any(ranges_t[:, 0] >= ranges_t[:, 1]):
            raise ValueError(
                "Each range must satisfy low < high."
            )

        self.register_buffer("ranges", ranges_t)

    # ---- derived properties ----------------------------------------

    @property
    def ndim(self) -> int:
        """
        Number of spatial dimensions.

        Returns
        -------
        int
        """
        return self.ranges.shape[0]

    @property
    def mins(self) -> torch.Tensor:
        """
        Lower corner of the box.

        Returns
        -------
        torch.Tensor of shape (ndim,)
        """
        return self.ranges[:, 0]

    @property
    def maxs(self) -> torch.Tensor:
        """
        Upper corner of the box.

        Returns
        -------
        torch.Tensor of shape (ndim,)
        """
        return self.ranges[:, 1]

    @property
    def lengths(self) -> torch.Tensor:
        """
        Side lengths of the box.

        Returns
        -------
        torch.Tensor of shape (ndim,)
        """
        return self.maxs - self.mins

    # ---- geometric helpers -----------------------------------------

    def contain(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Check whether points lie inside the box.

        The check is element-wise and inclusive of the boundaries.

        Parameters
        ----------
        pts : torch.Tensor of shape (..., ndim)
            Query points.

        Returns
        -------
        torch.Tensor of shape (...,), dtype bool
            ``True`` where the corresponding point is inside the
            box.
        """
        # Half-open: [min, max)
        above = torch.all(pts >= self.mins, dim=-1)
        below = torch.all(pts < self.maxs, dim=-1)
        return above & below

    def norm_coord(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Normalize coordinates to ``[0, 1]`` within the box.

        Parameters
        ----------
        pts : torch.Tensor of shape (..., ndim)
            Absolute coordinates.

        Returns
        -------
        torch.Tensor of shape (..., ndim)
            Normalized coordinates.  Values lie in ``[0, 1]`` for
            points that are inside the box.
        """
        return (pts - self.mins) / self.lengths

    def denorm_coord(
        self,
        pts_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        Map normalized coordinates back to absolute coordinates.

        Parameters
        ----------
        pts_norm : torch.Tensor of shape (..., ndim)
            Coordinates in ``[0, 1]``.

        Returns
        -------
        torch.Tensor of shape (..., ndim)
            Absolute coordinates.
        """
        return pts_norm * self.lengths + self.mins

    # ---- construction helpers --------------------------------------
    @classmethod
    def load(
        cls,
        file: Union[h5py.File, h5py.Group, str],
        group: Optional[str] = None,
    ) -> "AABox":
        """
        Construct an :class:`AABox` from an HDF5 source.

        The function reads ``min`` and ``max`` from the HDF5
        attributes of the target group.  If the attributes are not
        found, it falls back to reading datasets of the same name
        for backward-compatibility.

        Parameters
        ----------
        file : h5py.File, h5py.Group, or str
            An open HDF5 handle or a path to an HDF5 file.
        group : str, optional
            Group key inside the file that contains the metadata
            (e.g. ``"meta"``).  When *None* (default) the root
            group is used.

        Returns
        -------
        AABox
            A new axis-aligned box.

        Notes
        -----
        Expected HDF5 layout (attributes **or** datasets)::

            [group/]
                min   float[ndim]
                max   float[ndim]
        """
        if isinstance(file, str):
            with h5py.File(file, "r") as f:
                return cls.load(f, group=group)

        grp = file if group is None else file[group]

        mins = _read_h5(grp, "min")
        maxs = _read_h5(grp, "max")
        ranges = np.stack([mins, maxs], axis=-1)
        return cls(ranges)


    # ---- hparams --------------------------------------------------

	@classmethod
	def from_hparams(cls, hparams: dict) -> "AABox":
		"""Reconstruct an :class:`AABox` from a ``hparams`` dict.

		Parameters
		----------
		hparams : dict
			Recognised forms:

			* ``{"ranges": [[lo, hi], ...]}`` – construct directly from ranges.
			* ``{"file": "<path>"}`` – load from HDF5 via :meth:`AABox.load`.
			* ``{"file": "<path>", "group": "<group>"}`` – load from a
			  specific HDF5 group.

		Returns
		-------
		AABox

		Raises
		------
		ValueError
			If *hparams* contains neither ``"ranges"`` nor ``"file"``.
		"""
		if "ranges" in hparams:
			return cls(hparams["ranges"])

		if "file" in hparams:
			return cls.load(hparams["file"], group=hparams.get("group", None))

		raise ValueError(
			"AABox hparams must contain either 'ranges' or 'file'. "
			f"Got keys: {list(hparams.keys())}"
		)

    @property
    def hparams(self) -> dict:
        """Return a plain, serialisable dict sufficient to reconstruct this
        :class:`AABox` via :meth:`from_hparams`.

        Returns
        -------
        dict
            ``{"ranges": [[lo, hi], ...]}`` where each sub‐list corresponds
            to one spatial axis.
        """
        return {"ranges": self.ranges.tolist()}


# -----------------------------------------------------------------------------

class VoxelMeta(AABox):
    """Voxelised axis-aligned box.

    Extends :class:`AABox` with a regular grid defined by the
    number of divisions (voxels) along each axis.

    Parameters
    ----------
    ranges : array_like of shape (ndim, 2)
        Bounding box specification passed to :class:`AABox`.
    shape : array_like of int, length ndim
        Number of voxels along each axis.

    Attributes
    ----------
    ranges : torch.Tensor of shape (ndim, 2)
        Inherited bounding box buffer.
    shape_tensor : torch.Tensor of shape (ndim,), dtype long
        Registered buffer with the number of voxels per axis.

    Raises
    ------
    ValueError
        If ``shape`` is not 1-D with length ``ndim``, or if any
        entry is non-positive.

    Examples
    --------
    >>> meta = VoxelMeta(
    ...     [[0, 10], [0, 20], [0, 30]], [10, 20, 30]
    ... )
    >>> meta.n_voxels
    6000
    >>> meta.voxel_size
    tensor([1., 1., 1.])
    """

    def __init__(
        self,
        ranges: Union[
            torch.Tensor, np.ndarray, Sequence
        ],
        shape: Union[
            torch.Tensor, np.ndarray, Sequence[int]
        ],
    ) -> None:
        super().__init__(ranges)

        shape_t = torch.as_tensor(shape, dtype=torch.long)
        if (
            shape_t.ndim != 1
            or shape_t.shape[0] != self.ndim
        ):
            raise ValueError(
                f"shape must be 1-D with length "
                f"ndim={self.ndim}, "
                f"got {tuple(shape_t.shape)}"
            )
        if torch.any(shape_t <= 0):
            raise ValueError(
                "All shape entries must be positive."
            )

        self.register_buffer("shape_tensor", shape_t)

        self.register_buffer(
            "_strides_cache",
            self._compute_strides(shape_t),
        )

    # ---- derived properties ----------------------------------------

    @property
    def voxel_shape(self) -> torch.Size:
        """
        Number of voxels per axis.

        Returns
        -------
        torch.Size of length ndim
        """
        return torch.Size(self.shape_tensor.tolist())

    @property
    def n_voxels(self) -> int:
        """
        Total number of voxels in the grid.

        Returns
        -------
        int
        """
        return int(self.shape_tensor.prod().item())

    @property
    def voxel_size(self) -> torch.Tensor:
        """
        Side lengths of a single voxel.

        Returns
        -------
        torch.Tensor of shape (ndim,)
        """
        return self.lengths / self.shape_tensor.float()

    # ---- strides (for ravelling / unravelling) -----------------
    @staticmethod
    def _compute_strides(shape_t):
        s = shape_t.flip(0).cumprod(0).flip(0)
        strides = torch.ones_like(shape_t)
        strides[:-1] = s[1:]
        return strides

    @property
    def _strides(self):
        return self._strides_cache

    # ---- voxel <-> position conversion -------------------------

    def voxel_to_coord(
        self,
        voxel_ids: torch.Tensor,
    ) -> torch.Tensor:

        """
        Convert coordinates to flat voxel indices.

        Points outside the bounding box are **clamped** to the
        nearest boundary voxel.  No error is raised and no
        sentinel value (e.g. ``-1``) is returned for
        out-of-bounds points.  Use :meth:`AABox.contain` to
        filter or flag such points before calling this method
        if that distinction matters.

        Parameters
        ----------
        pts : torch.Tensor of shape (..., ndim)
            Absolute coordinates.

        Returns
        -------
        torch.LongTensor of shape (...)
            Flat (ravelled, C-order) voxel indices.  Values
            are guaranteed to lie in ``[0, n_voxels - 1]``.

        See Also
        --------
        voxel_to_coord : Inverse operation.
        AABox.contain :
            Check whether points lie inside the box before
            calling this method.

        Examples
        --------
        Points inside the box map to the expected voxel:

        >>> meta = VoxelMeta([[0, 10], [0, 20]], [5, 10])
        >>> meta.coord_to_voxel(
        ...     torch.tensor([[1.0, 1.0]])
        ... )
        tensor([0])

        Points outside are clamped, not rejected:

        >>> meta.coord_to_voxel(
        ...     torch.tensor([[-999.0, -999.0]])
        ... )
        tensor([0])

        Filter out-of-bounds points explicitly:

        >>> pts = torch.tensor([
        ...     [1.0, 1.0],
        ...     [-5.0, 25.0],
        ... ])
        >>> mask = meta.contain(pts)
        >>> vids = meta.coord_to_voxel(pts[mask])
        """

        multi = self._unravel(voxel_ids)
        return (
            self.mins
            + (multi.float() + 0.5) * self.voxel_size
        )

    def coord_to_voxel(
        self,
        pts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert coordinates to flat voxel indices.

        Points are clamped to valid voxel indices.  Use
        :meth:`contain` to check whether points are inside the
        box beforehand if needed.

        Parameters
        ----------
        pts : torch.Tensor of shape (..., ndim)
            Absolute coordinates.

        Returns
        -------
        torch.LongTensor of shape (...)
            Flat (ravelled) voxel indices in C-order.

        See Also
        --------
        voxel_to_coord : Inverse operation.
        """
        frac = (pts - self.mins) / self.voxel_size
        multi = frac.long()
        
        # IDEA: optionally support different out-of-bounds policies
        #       (e.g., 'zero', 'error', 'mask') for use cases where
        #       pre-filtering via meta.contain() is not practical.
        # 2026-04-11 kvt
        multi = multi.clamp(
            min=torch.zeros_like(self.shape_tensor),
            max=self.shape_tensor - 1,
        )
        return self._ravel(multi)

    # ---- multi-index helpers -----------------------------------

    def _ravel(
        self,
        multi_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a multi-index to a flat index (C-order).

        Parameters
        ----------
        multi_index : torch.LongTensor of shape (..., ndim)
            Per-axis voxel indices.

        Returns
        -------
        torch.LongTensor of shape (...)
            Flat voxel index.
        """
        return (multi_index * self._strides).sum(dim=-1)

    def _unravel(
        self,
        flat_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert a flat index to a multi-index (C-order).

        Parameters
        ----------
        flat_index : torch.LongTensor of shape (...)
            Flat voxel index.

        Returns
        -------
        torch.LongTensor of shape (..., ndim)
            Per-axis voxel indices.
        """
        strides = self._strides
        multi = []
        remainder = flat_index
        for i in range(self.ndim):
            multi.append(remainder // strides[i])
            remainder = remainder % strides[i]
        return torch.stack(multi, dim=-1)

    # ---- construction from HDF5 --------------------------------
    @classmethod
    def load(
        cls,
        file: Union[h5py.File, h5py.Group, str],
        group: Optional[str] = None,
    ) -> "VoxelMeta":
        """
        Construct a :class:`VoxelMeta` from an HDF5 source.

        The function reads ``min``, ``max``, and ``numvox`` from the
        HDF5 attributes of the target group.  If any attribute is
        not found, it falls back to reading a dataset of the same
        name for backward-compatibility.

        Parameters
        ----------
        file : h5py.File, h5py.Group, or str
            An open HDF5 handle or a path to an HDF5 file.
        group : str, optional
            Group key inside the file that contains the metadata
            (e.g. ``"meta"``).  When *None* (default) the root
            group is used.

        Returns
        -------
        VoxelMeta
            A new voxelised box.

        Notes
        -----
        Expected HDF5 layout (attributes **or** datasets)::

            [group/]
                min      float[ndim]  -- lower corner
                max      float[ndim]  -- upper corner
                numvox   int[ndim]    -- voxels per axis
        """
        if isinstance(file, str):
            with h5py.File(file, "r") as f:
                return cls.load(f, group=group)

        grp = file if group is None else file[group]

        mins = _read_h5(grp, "min", np.float32)
        maxs = _read_h5(grp, "max", np.float32)
        shape = _read_h5(grp, "numvox", np.int64)
        ranges = np.stack([mins, maxs], axis=-1)
        return cls(ranges, shape)

    # ---- repr --------------------------------------------------

    def extra_repr(self) -> str:
        """
        Return the voxel specification for :func:`repr`.

        Returns
        -------
        str
        """
        base = super().extra_repr()
        return f"{base}, voxel_shape={self.voxel_shape}"

# ---- module-level helpers ----------------------------------
def _read_h5(grp, key, dtype):
    """
    Read key from attrs, falling back to dataset.

    Parameters
    ----------
    grp : h5py.File or h5py.Group
    key : str
    dtype : numpy dtype

    Returns
    -------
    numpy.ndarray

    Raises
    ------
    KeyError
        If key not found as attr or dataset.
    """
    if key in grp.attrs:
        return np.asarray(
            grp.attrs[key], dtype=dtype
        )
    if key in grp:
        return np.asarray(
            grp[key], dtype=dtype
        )
    raise KeyError(
        f"'{key}' not found in {grp.name}"
    )
