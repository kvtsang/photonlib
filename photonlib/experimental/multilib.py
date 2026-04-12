"""Collection of non-overlapping photon libraries."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import h5py
import numpy as np
import torch
import torch.nn as nn

from .meta import AABox, VoxelMeta
from .photonlib import LazyPhotonLib, PhotonLib


class MultiPhotonLib(nn.Module):
    """Union of non-overlapping photon libraries.

    Provides a unified lookup interface over multiple
    :class:`PhotonLib` or :class:`LazyPhotonLib` instances
    whose bounding boxes tile a larger volume without
    overlap.

    Each sub-library owns a contiguous slice along the PMT
    axis.  The total number of PMTs is the sum across all
    sub-libraries.

    Parameters
    ----------
    libs : sequence of PhotonLib or LazyPhotonLib
        The constituent libraries.  Their bounding boxes
        must not overlap.

    Attributes
    ----------
    libs : nn.ModuleList
        The constituent libraries.
    bbox : AABox
        Axis-aligned bounding box enclosing the union of
        all sub-library volumes.

    Raises
    ------
    ValueError
        If ``libs`` is empty or if any pair of bounding
        boxes overlaps.

    Examples
    --------
    ::

        a = PhotonLib.load("part_a.h5")
        b = PhotonLib.load("part_b.h5")
        mlib = MultiPhotonLib([a, b])
        vis = mlib.vis_at_coord(pts)
        assert vis.shape == (len(pts), mlib.n_pmts)
    """

    def __init__(
        self,
        libs: Sequence[
            Union[PhotonLib, LazyPhotonLib]
        ],
    ) -> None:
        super().__init__()

        if len(libs) == 0:
            raise ValueError(
                "At least one library is required."
            )

        self._validate_no_overlap(libs)

        self.libs = nn.ModuleList(libs)

        # PMT slice boundaries: lib i owns
        # pmt_offsets[i] .. pmt_offsets[i+1]
        offsets = [0]
        for lib in libs:
            offsets.append(offsets[-1] + lib.n_pmts)
        self.register_buffer(
            "_pmt_offsets",
            torch.tensor(offsets, dtype=torch.long),
        )

        # Overall bounding box.
        all_mins = torch.stack(
            [lib.meta.mins for lib in libs]
        )
        all_maxs = torch.stack(
            [lib.meta.maxs for lib in libs]
        )
        global_mins = all_mins.min(dim=0).values
        global_maxs = all_maxs.max(dim=0).values
        ranges = torch.stack(
            [global_mins, global_maxs], dim=-1
        )
        self.bbox = AABox(ranges)

    # ---- properties ----------------------------------------

    @property
    def n_libs(self) -> int:
        """Number of sub-libraries.

        Returns
        -------
        int
        """
        return len(self.libs)

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions.

        Returns
        -------
        int
        """
        return self.bbox.ndim

    @property
    def n_pmts(self) -> int:
        """Total number of PMTs across all sub-libraries.

        Returns
        -------
        int
        """
        return int(self._pmt_offsets[-1].item())

    @property
    def n_pmts_per_lib(self) -> tuple[int, ...]:
        """Number of PMTs for each sub-library.

        Returns
        -------
        tuple of int, length :attr:`n_libs`

        Examples
        --------
        >>> mlib.n_pmts_per_lib
        (180, 180, 120)
        >>> mlib.n_pmts
        480
        """
        return tuple(
            lib.n_pmts for lib in self.libs
        )

    def pmt_slice(self, lib_index: int) -> slice:
        """PMT column slice for a sub-library.

        Parameters
        ----------
        lib_index : int
            Index into :attr:`libs`.

        Returns
        -------
        slice
            Column range ``[start, stop)`` in the
            concatenated PMT axis.

        Examples
        --------
        >>> mlib.pmt_slice(0)
        slice(0, 180, None)
        """
        lo = int(self._pmt_offsets[lib_index].item())
        hi = int(
            self._pmt_offsets[lib_index + 1].item()
        )
        return slice(lo, hi)

    # ---- validation ----------------------------------------

    @staticmethod
    def _validate_no_overlap(
        libs: Sequence[
            Union[PhotonLib, LazyPhotonLib]
        ],
    ) -> None:
        """Check that no two bounding boxes overlap.

        Parameters
        ----------
        libs : sequence of PhotonLib or LazyPhotonLib

        Raises
        ------
        ValueError
            If any pair of boxes overlaps.
        """
        n = len(libs)
        for i in range(n):
            for j in range(i + 1, n):
                if _boxes_overlap(
                    libs[i].meta, libs[j].meta
                ):
                    raise ValueError(
                        f"Bounding boxes of lib[{i}] "
                        f"and lib[{j}] overlap."
                    )

    # ---- routing -------------------------------------------

    def _find_lib_indices(
        self,
        pts: torch.Tensor,
    ) -> torch.Tensor:
        """For each point find the sub-library index.

        Parameters
        ----------
        pts : torch.Tensor of shape (N, ndim)

        Returns
        -------
        torch.LongTensor of shape (N,)
            Index into :attr:`libs` for each point.
            ``-1`` if the point is outside all boxes.
        """
        n = pts.shape[0]
        lib_idx = torch.full(
            (n,),
            -1,
            dtype=torch.long,
            device=pts.device,
        )

        for i, lib in enumerate(self.libs):
            mask = lib.meta.contain(pts)
            lib_idx[mask] = i

        return lib_idx

    # ---- lookup --------------------------------------------

    def vis_at_coord(
        self,
        pts: torch.Tensor,
        default: float = 0.0,
    ) -> torch.Tensor:
        """Look up visibility by spatial coordinates.

        Each point is routed to the sub-library whose
        bounding box contains it.  The result is a dense
        tensor covering **all** PMTs across all
        sub-libraries, where each sub-library's
        contribution occupies a contiguous column slice.

        Points outside every sub-library receive the
        *default* fill value.

        Parameters
        ----------
        pts : torch.Tensor of shape (..., ndim)
            Spatial coordinates.
        default : float, optional
            Fill value for points outside all boxes and
            for PMT columns that belong to a different
            sub-library.  Default is ``0.0``.

        Returns
        -------
        torch.Tensor of shape (..., n_pmts)
            Visibility values.  Column layout::

                [lib0_pmts | lib1_pmts | ...]

        Examples
        --------
        >>> vis = mlib.vis_at_coord(pts)
        >>> vis.shape
        torch.Size([100, 480])
        """
        orig_shape = pts.shape[:-1]
        flat_pts = pts.reshape(-1, self.ndim)
        n = flat_pts.shape[0]

        result = torch.full(
            (n, self.n_pmts),
            default,
            dtype=torch.float32,
            device=pts.device,
        )

        lib_idx = self._find_lib_indices(flat_pts)

        for i, lib in enumerate(self.libs):
            mask = lib_idx == i
            if not mask.any():
                continue
            sub_pts = flat_pts[mask]
            col = self.pmt_slice(i)
            result[mask, col] = (
                lib.vis_at_coord(sub_pts)
            )

        return result.reshape(
            *orig_shape, self.n_pmts
        )

    def visibility(
        self,
        pts: torch.Tensor,
        default: float = 0.0
    )-> torch.Tensor:
        """Backward compatible function for `vis_at_coord`"""
        return self.vis_at_coord(pts, default)

    def contain(
        self,
        pts: torch.Tensor,
    ) -> torch.Tensor:
        """Check whether points lie inside any sub-library.

        Parameters
        ----------
        pts : torch.Tensor of shape (..., ndim)

        Returns
        -------
        torch.Tensor of shape (...,), dtype bool
        """
        orig_shape = pts.shape[:-1]
        flat_pts = pts.reshape(-1, self.ndim)

        inside = torch.zeros(
            flat_pts.shape[0],
            dtype=torch.bool,
            device=pts.device,
        )

        for lib in self.libs:
            inside |= lib.meta.contain(flat_pts)

        return inside.reshape(orig_shape)

    # ---- construction from HDF5 ----------------------------

    @classmethod
    def load(
        cls,
        sources: Sequence[
            Union[
                str,
                tuple[str, Sequence[str]],
            ]
        ],
        lazy: bool = False,
    ) -> "MultiPhotonLib":
        """Build from one or more HDF5 sources.

        Parameters
        ----------
        sources : list of str or (str, list of str)
            Each entry specifies one or more sub-libraries:

            **Plain string** — one library from root::

                "part_a.h5"

            **Tuple** — multiple groups from one file::

                ("shared.h5", ["vol_a", "vol_b"])

            The two forms can be mixed.

        lazy : bool, optional
            If *True*, use :class:`LazyPhotonLib`.
            Default is *False*.

        Returns
        -------
        MultiPhotonLib

        Examples
        --------
        One file per library::

            mlib = MultiPhotonLib.load([
                "part_a.h5",
                "part_b.h5",
            ])

        Single file, multiple groups::

            mlib = MultiPhotonLib.load([
                ("detector.h5", ["vol_a", "vol_b"]),
            ])

        Mixed::

            mlib = MultiPhotonLib.load([
                "file0.h5",
                ("file1.h5", ["vol_b", "vol_c"]),
            ])
        """
        pairs = _normalize_sources(sources)

        libs: list[
            Union[PhotonLib, LazyPhotonLib]
        ] = []
        for filepath, group in pairs:
            if lazy:
                lib = LazyPhotonLib(
                    filepath, group=group
                )
            else:
                lib = PhotonLib.load(
                    filepath, group=group
                )
            libs.append(lib)

        return cls(libs)

    # ---- repr ----------------------------------------------

    def extra_repr(self) -> str:
        """Return a summary for :func:`repr`.

        Returns
        -------
        str
        """
        mins = self.bbox.mins.tolist()
        maxs = self.bbox.maxs.tolist()
        return (
            f"n_libs={self.n_libs}, "
            f"n_pmts={self.n_pmts}, "
            f"n_pmts_per_lib="
            f"{self.n_pmts_per_lib}, "
            f"mins={mins}, maxs={maxs}"
        )


# ---- module-level helpers ----------------------------------

def _boxes_overlap(
    a: AABox,
    b: AABox,
) -> bool:
    """Check whether two axis-aligned boxes overlap.

    Two boxes overlap if and only if they overlap on
    **every** axis.  Strict inequality is used so that
    touching faces (shared boundary) are allowed.

    Parameters
    ----------
    a : AABox
    b : AABox

    Returns
    -------
    bool
    """
    return bool(
        torch.all(a.mins < b.maxs)
        and torch.all(b.mins < a.maxs)
    )


def _normalize_sources(
    sources: Sequence[
        Union[
            str,
            tuple[str, Sequence[str]],
        ]
    ],
) -> list[tuple[str, Optional[str]]]:
    """Expand source specs into (file, group) pairs.

    Parameters
    ----------
    sources : list of str or (str, list of str)
        See :meth:`MultiPhotonLib.load`.

    Returns
    -------
    list of (str, str or None)
        Flattened ``(filepath, group)`` pairs.

    Raises
    ------
    TypeError
        If an entry has an unsupported type.
    ValueError
        If the resulting list is empty.
    """
    pairs: list[tuple[str, Optional[str]]] = []

    for entry in sources:
        if isinstance(entry, str):
            pairs.append((entry, None))

        elif isinstance(entry, (tuple, list)):
            if (
                len(entry) != 2
                or not isinstance(entry[0], str)
                or not isinstance(
                    entry[1], (list, tuple)
                )
            ):
                raise TypeError(
                    "Tuple entries must be "
                    "(filepath, [group, ...]), "
                    f"got {entry!r}."
                )

            filepath = entry[0]
            groups = entry[1]

            if len(groups) == 0:
                raise ValueError(
                    f"Empty group list for "
                    f"'{filepath}'."
                )

            for grp in groups:
                if not isinstance(grp, str):
                    raise TypeError(
                        "Group names must be str, "
                        f"got {type(grp)}."
                    )
                pairs.append((filepath, grp))

        else:
            raise TypeError(
                f"Unsupported entry type: "
                f"{type(entry)}."
            )

    if len(pairs) == 0:
        raise ValueError(
            "At least one source is required."
        )

    return pairs
