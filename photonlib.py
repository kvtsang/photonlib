import h5py
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from functools import partial
from contextlib import contextmanager
from scipy.ndimage import sobel

class Meta:
    def __init__(self, shape, ranges, lib=np):
        self.shape = np.copy(shape)
        self.ranges = np.copy(ranges)
        self.lib = lib
       
    def __repr__(self):
        s = 'Meta'
        for i,var in enumerate('xyz'):
            bins = self.shape[i]
            x0, x1 = self.ranges[i]
            s += f' {var}:({x0},{x1},{bins})'
        return s

    @property
    def bins(self):
        output = tuple(
            np.linspace(ranges[0], ranges[1], nbins)
            for ranges, nbins in zip(self.ranges, self.shape+1)
        )

        return output

    @property
    def bin_centers(self):
        centers = tuple((b[1:] + b[:-1]) / 2. for b in self.bins)
        return centers
        
    @property
    def voxel_size(self):
        voxel_size = np.diff(self.ranges).flat / self.shape
        return voxel_size.astype(np.float32)

    @property
    def norm_step_size(self):
        return 2. / self.shape

    @property
    def length(self):
        return np.diff(self.ranges).squeeze()

    def __len__(self):
        return np.product(self.shape)
    
    def _as_type(self, x, device=None):
        if hasattr(self.lib, 'as_tensor'):
            if device is None:
                device = self.device(x)
            return self.lib.as_tensor(x, device=device)
        return np.asarray(x)

    @staticmethod
    def device(x):
        if hasattr(x, 'device'):
            return x.device
        return None

    @classmethod
    def load(cls, fname, lib=np):
        with h5py.File(fname, 'r') as f:
            shape = f['numvox'][:]
            ranges = np.column_stack((f['min'], f['max']))
        return cls(shape.astype(np.int), ranges.astype(np.float32), lib)
    
    def idx_to_voxel(self, idx):
        idx = self._as_type(idx)

        if len(idx.shape) == 1:
            idx = idx[None,:]

        nx, ny = self.shape[:2]
        vox = idx[:,0] + idx[:,1]*nx + idx[:,2]*nx*ny

        return vox.squeeze()
    
    def voxel_to_idx(self, voxel):
        voxel = self._as_type(voxel)
        nx, ny = self.shape[:2]

        if hasattr(self.lib, 'div'):
            fdiv = partial(self.lib.div, rounding_mode='floor')
        else:
            fdiv = self.lib.floor_divide

        idx = self.lib.column_stack([
            voxel % nx, fdiv(voxel, nx) % ny, fdiv(voxel, nx*ny)])

        return idx.squeeze()
    
    def idx_to_coord(self, idx, norm=False):
        device = self.device(idx)
        idx = self._as_type(idx, device)

        if norm:
            shape = self._as_type(self.shape, device)
            coord = (idx+0.5) / shape
            coord *= 2
            coord -= 1
            return coord

        voxel_size = self._as_type(self.voxel_size, device)
        ranges = self._as_type(self.ranges, device)
        coord = (idx+0.5) * voxel_size
        coord += ranges[:, 0]
        return coord

    def voxel_to_coord(self, voxel, norm=False):
        idx = self.voxel_to_idx(voxel)
        return self.idx_to_coord(idx, norm)
    
    def coord_to_idx(self, coord, norm=False):
        # TODO(2021-10-29 kvt) validate coord_to_idx
        # TODO(2021-10-29 kvt) check ranges
        coord = self._as_type(coord)
        device = self.device(coord)

        if norm:
            step = self._as_type(self.norm_step_size, device=device)
            idx = (coord + 1.) / step
        else:
            step = self._as_type(self.voxel_size, device=device)
            ranges = self._as_type(self.ranges, device=device)
            idx = (coord - ranges[:,0]) / step

        idx = self.as_int64(idx)
        idx[idx<0] == 0
        for axis in range(3):
            n = self.shape[axis]
            mask = idx[...,axis] >= n
            idx[mask,axis] = n-1

        return idx

    def as_int64(self, idx):
        if isinstance(idx, np.ndarray):
            idx = idx.astype(self.lib.int64)
        else:
            idx = idx.type(self.lib.int64)
        return idx


    def self_check(self, n=1000):
        prev_lib = self.lib

        meta = self
        self.lib = np
        bins = [np.linspace(meta.ranges[i,0], meta.ranges[i,1], meta.shape[i]+1) 
               for i in range(3)]
        
        for __ in tqdm(range(n)):
            pos = np.random.uniform(meta.ranges[:,0], meta.ranges[:,1])
            idx = np.array([np.digitize(pos[i], bins[i]) - 1 for i in range(3)])
            vox = meta.idx_to_voxel(idx)

            if not np.allclose(idx, meta.voxel_to_idx(vox)):
                raise RunTimeError('voxel_to_idx', pos)

            coord = meta.voxel_to_coord(vox)
            if not np.all(np.abs(coord - pos) < meta.voxel_size):
                raise RunTimeError('voxel_to_coord', pos)

            vox = np.random.randint(low=0, high=len(meta))
            idx = meta.voxel_to_idx(vox)
            pos = meta.voxel_to_coord(pos)

            if not np.allclose(vox, meta.idx_to_voxel(idx)):
                raise RunTimeError('idx_to_voxel', vox)

        self.lib = prev_lib

    @staticmethod
    def select_axis(axis):
        axis_to_num = dict(x=0, y=1, z=2)
        
        if isinstance(axis, str) and axis in axis_to_num:
            axis = axis_to_num[axis]
            
        axis_others = [0, 1, 2]
        if axis not in axis_others:
            raise IndexError(f'unknown axis {axis}')
        axis_others.pop(axis)

        return axis, axis_others

    def idx_at(self, axis, i, device=None):
        axis, axis_others = self.select_axis(axis)
        axis_a, axis_b = axis_others

        grid = [None] * 3
        grid[axis] = i
        grid[axis_a] = np.arange(self.shape[axis_a])
        grid[axis_b] = np.arange(self.shape[axis_b])

        idx = np.column_stack([g.flatten() for g in np.meshgrid(*grid)])
        return self._as_type(idx, device=device)
    
    def check_valid_idx(self, idx, return_components=False):
        idx = self._as_type(idx)
        shape = self._as_type(self.shape)
        mask = (idx >= 0) & (idx < shape)

        if return_components:
            return mask

        return self.lib.all(mask, axis=-1)

    @contextmanager
    def use_lib(self, lib):
        prev_lib = self.lib
        self.lib = lib
        try:
            yield self
        finally:
            self.lib = prev_lib

    def norm_coord(self, pos):
        pos = self._as_type(pos)
        device = self.device(pos)

        ranges = self._as_type(self.ranges, device=device)

        norm_pos = pos - ranges[:,0]
        norm_pos /= self._as_type(self.length, device=device)
        norm_pos *= 2.
        norm_pos -= 1.

        return norm_pos

    def digitize(self, x, axis, norm=False):
        x = self._as_type(x)
        device = self.device(x)
        axis = self.select_axis(axis)[0]
        n = self.shape[axis]

        if norm:
            xmin = -1
            step = self.norm_step_size[axis]
        else:
            xmin = self.ranges[axis, 0]
            step = self.voxel_size[axis]

        idx = self.as_int64((x - xmin) / step)

        # TODO: (2021-10-29 kvt) exception?
        idx[idx<0] = 0
        idx[idx>=n] = n-1

        return idx

class PhotonLib:
    def __init__(
        self, meta, vis, pmt_pos=None, 
        transform=False, eps=1e-7, vmax=1, lib=np
    ):
        self.meta = meta

        if transform:
            print(f'[PhotonLib] transform(vmax={vmax}, eps={eps})')
            self.vis = self.transform(vis, vmax, eps)
        else:
            self.vis = vis

        if pmt_pos is not None:
            self.pmt_pos = pmt_pos
            self.pmt_pos_norm = meta.norm_coord(pmt_pos)
    
    @classmethod
    def load(cls, filepath, pmt_loc=None, lib=np, **kwargs):
        meta = Meta.load(filepath, lib=lib)
        
        print(f'[PhotonLib] loading {filepath}')
        with h5py.File(filepath, 'r') as f:
            vis = f['vis'][:]
        print('[PhotonLib] file loaded')

        pmt_pos = None
        if pmt_loc is not None:
            pmt_pos = PhotonLib.load_pmt_loc(pmt_loc)

        plib = cls(meta, vis, pmt_pos, **kwargs)

        return plib

    @staticmethod
    def load_pmt_loc(fpath):
        df = pd.read_csv(fpath)
        pmt_pos = df[['x', 'y', 'z']].to_numpy()
        return pmt_pos
    

    def view(self, arr):
        shape = list(self.meta.shape[::-1]) + [-1]
        return np.swapaxes(arr.reshape(shape), 0, 2)

    @property
    def vis_view(self):
        return self.view(self.vis)

    def __repr__(self):
        return f'{self.__class__} [:memory:]'
    
    def __len__(self):
        return len(self.vis)
    
    @property
    def n_pmts(self):
        return self.vis.shape[1]
     
    def __getitem__(self, vox_id):    
        return self.vis[vox_id]

    def gradient_on_fly(self, voxel_id):
        with self.meta.use_lib(np):
            idx = self.meta.voxel_to_idx(voxel_id)

        center = np.ones_like(idx)
        center[idx == 0] = 0
        center = tuple(center)

        high = idx + 2
        low = idx - 1
        low[low<0] = 0
        selected = selected = tuple(slice(l,h) for l,h in zip(low, high))

        data = self.vis_view[selected]
        grad = np.column_stack([
            [sobel(data[...,pmt], i)[center] for i in range(3)]
            for pmt in range(self.n_pmts)
        ])

        return grad

    def gradient_from_cache(self, voxel_id):
        if self.grad_cache is None:
            raise RunTimeError('grad_cache not loaded')

        return self.grad_cache[voxel_id]

    def gradient(self, voxel_id):
        if self.grad_cache is not None:
            grad = self.gradient_from_cache(voxel_id)
        else:
            grad = self.gradient_on_fly(voxel_id)

        # convert to dV/dx for comparison with torch.autograd.grad
        # sobel = gaus [1,2,1] (x) gaus [1,2,1] (x) diff [1,0,-1]
        # resacle with a factor of  4x4 (gauss) and 2 (finite diff.)
        # grad /= self.meta.norm_step_size * 32
        return grad

    def grad_view(self, d_axis):
        if self.grad_cache is None:
            raise NotImplementedError('gradient_view requires caching')

        d_axis = self.meta.select_axis(d_axis)[0]
        return self.view(self.grad_cache[:,d_axis])

    @staticmethod
    def transform(x, vmax=1, eps=1e-7, lib=np):
        y0 = np.log10(eps)
        y1 = np.log10(vmax+ eps)

        y = lib.log10(x + eps)
        y -= y0
        y /= (y1 - y0)
        return y

    @staticmethod
    def inv_transform(y, vmax=1, eps=1e-7, lib=np):
        y0 = np.log10(eps)
        y1 = np.log10(vmax + eps)

        x = 10 ** (y * (y1-y0) + y0)
        x -= eps

        return x
