import unittest
import torch
import taichi as ti
from pointnet import PointNet2ClsSSG, PointNet2ClsMSG, PointNet2SegMSG, PointNet2SegSSG, PointNet2PartSegSSG, \
    PointNet2PartSegMSG

ti.init(ti.cuda)


class Test(unittest.TestCase):

    def test_pointnet2_cls_ssg(self):
        x = torch.randn(2, 3, 1024).cuda()
        xyz = x.clone()
        model = PointNet2ClsSSG(3, 40).cuda()
        out = model(x, xyz)
        assert out.shape == (2, 40)

    def test_pointnet2_cls_msg(self):
        x = torch.randn(2, 3, 1024).cuda()
        xyz = x.clone()
        model = PointNet2ClsMSG(3, 40).cuda()
        out = model(x, xyz)
        assert out.shape == (2, 40)

    def test_pointnet2_seg_ssg(self):
        x = torch.randn(2, 3, 1024).cuda()
        xyz = x.clone()
        model = PointNet2SegSSG(3, 40).cuda()
        out = model(x, xyz)
        assert out.shape == (2, 40, 1024)

    def test_pointnet2_seg_msg(self):
        x = torch.randn(2, 3, 1024).cuda()
        xyz = x.clone()
        model = PointNet2SegMSG(3, 40).cuda()
        out = model(x, xyz)
        assert out.shape == (2, 40, 1024)

    def test_pointnet2_part_seg_ssg(self):
        x = torch.randn(2, 3, 1024).cuda()
        xyz = x.clone()
        category = torch.randint(0, 10, (2,)).cuda()
        model = PointNet2PartSegSSG(3, 40).cuda()
        out = model(x, xyz, category)
        assert out.shape == (2, 40, 1024)

    def test_pointnet2_part_seg_msg(self):
        x = torch.randn(2, 3, 1024).cuda()
        xyz = x.clone()
        category = torch.randint(0, 10, (2,)).cuda()
        model = PointNet2PartSegMSG(3, 40).cuda()
        out = model(x, xyz, category)
        assert out.shape == (2, 40, 1024)


if __name__ == '__main__':
    unittest.main()