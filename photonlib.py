import h5py
import torch
import numpy as np
from functools import partial
from tqdm import tqdm

class Meta:
    def __init__(self, shape, ranges, lib):
        self.shape = np.copy(shape)
        self.ranges = np.copy(ranges)
        voxel_size = np.diff(ranges).flat / shape
        self.voxel_size = voxel_size.astype(np.float32)
        self.lib = lib
        
    def __repr__(self):
        s = 'Meta'
        for i,var in enumerate('xyz'):
            bins = self.shape[i]
            x0, x1 = self.ranges[i]
            s += f' {var}:({x0},{x1},{bins})'
        return s

    @property
    def max_voxel_id(self):
        return np.product(self.shape)

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
    def load_file(cls, fname, lib=np):
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

    def transform(self, x):
        a = 0.999
        e = 1e-6
        high = np.log(a*(1+e))
        low = np.log(a*e)

        y = self.lib.log(a*(x+e))
        y -= low
        y /= (high - low)
        y *= 2
        y -= 1
        return y

    def inv_transform(self, y):
        a = 0.999
        e = 1e-6
        high = np.log(a*(1+e))
        low = np.log(a*e)

        x = y + 1
        x /= 2
        x *= (high - low)
        x += low
        return self.lib.exp(x)/a - e

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

            vox = np.random.randint(low=0, high=meta.max_voxel_id)
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


class PhotonLib:
    
    def __init__(self, meta, vis):
        self.meta = meta
        self.vis = vis
    
    @classmethod
    def load_file(cls, filepath):
        meta = Meta.load_file(filepath)
        
        print(f'Opening PhotonLib {filepath}')
        with h5py.File(filepath, 'r') as f:
            vis = f['vis'][:]
        print('PhotonLib file closed')
        
        plib = cls(meta, vis.astype(np.float32))
        plib.create_view()
        return plib
    
    def create_view(self):
        shape = list(self.meta.shape[::-1]) + [-1]
        self._vis_view = np.swapaxes(self.vis.reshape(shape), 0, 2)
              
    def __repr__(self):
        return f'{self.__class__} [:memory:]'
    
    def __len__(self):
        return len(self.vis)
    
    def __getitem__(self, vox_id):    
        return self.vis[vox_id]
                
    def view(self, axis, idx, ch=None, return_ranges=False):
        axis_to_num = dict(x=0, y=1, z=2)
        
        if isinstance(axis, str) and axis in axis_to_num:
            axis = axis_to_num[axis]
            
        axis_others = [0, 1, 2]
        if axis not in axis_others:
            raise IndexError(f'unknown axis {axis}')
        axis_others.pop(axis)
        
        mask = [slice(None)] * 4
        mask[axis] = idx

        if ch is None:
            output = [self._vis_view[tuple(mask)].sum(axis=-1)]
        else:
            mask[3] = ch
            output = [self._vis_view[tuple(mask)]]
        
        if return_ranges:
            output.append(self.meta.ranges[axis_others])
        
        if len(output) == 1:
            return output[0]
        return tuple(output)
