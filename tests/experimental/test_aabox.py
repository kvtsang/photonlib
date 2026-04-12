import torch
import pytest
from photonlib.experimental import AABox


class TestAABoxConstruction:
    def test_from_ranges(self, ranges):
        box = AABox(ranges)
        assert box.ndim == ranges.shape[0]
        assert torch.equal(box.ranges, ranges)

    def test_mins_maxs(self, ranges):
        box = AABox(ranges)
        assert torch.equal(box.mins, ranges[:, 0])
        assert torch.equal(box.maxs, ranges[:, 1])

    def test_invalid_ranges_wrong_shape(self):
        with pytest.raises((ValueError, RuntimeError)):
            AABox(torch.tensor([1.0, 2.0, 3.0]))

    def test_invalid_ranges_min_gt_max(self):
        ranges = torch.tensor([[10.0, -10.0], [0.0, 5.0]])
        with pytest.raises(ValueError):
            AABox(ranges)


class TestAABoxContain:
    def test_inside(self, ranges):
        box = AABox(ranges)
        center = ranges.mean(dim=-1).unsqueeze(0)
        assert box.contain(center).all()

    def test_outside_above(self, ranges):
        box = AABox(ranges)
        point = (ranges[:, 1] + 1.0).unsqueeze(0)
        assert not box.contain(point).any()

    def test_outside_below(self, ranges):
        box = AABox(ranges)
        point = (ranges[:, 0] - 1.0).unsqueeze(0)
        assert not box.contain(point).any()

    def test_on_lower_boundary(self, ranges):
        box = AABox(ranges)
        point = ranges[:, 0].unsqueeze(0)
        assert box.contain(point).all()

    def test_on_upper_boundary(self, ranges):
        box = AABox(ranges)
        point = ranges[:, 1].unsqueeze(0)
        result = box.contain(point)
        assert result.shape == (1,)

    def test_batch_all_inside(self, ranges):
        box = AABox(ranges)
        n = 100
        ndim = ranges.shape[0]
        coords = (
            ranges[:, 0].unsqueeze(0)
            + torch.rand(n, ndim) * (ranges[:, 1] - ranges[:, 0]).unsqueeze(0)
        )
        coords = coords * 0.99 + ranges.mean(dim=-1).unsqueeze(0) * 0.01
        assert box.contain(coords).all()

    def test_mixed_in_out(self, ranges):
        box = AABox(ranges)
        center = ranges.mean(dim=-1)
        far = ranges[:, 1] + 100.0
        coords = torch.stack([center, far])
        mask = box.contain(coords)
        assert mask[0] and not mask[1]


class TestAABoxDevice:
    def test_to_device(self, ranges, device):
        box = AABox(ranges).to(device)
        assert box.ranges.device.type == device.type
        assert box.mins.device.type == device.type
        assert box.maxs.device.type == device.type

    def test_dtype_cast(self, ranges):
        box = AABox(ranges).to(dtype=torch.float64)
        assert box.ranges.dtype == torch.float64
