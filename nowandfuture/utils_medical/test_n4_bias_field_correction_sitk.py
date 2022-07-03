import glob
import os
from unittest import TestCase

from nowandfuture.utils_medical import plot
from nowandfuture.utils_medical.preproccess import n4_bias_field_correction_sitk as n4b
import nowandfuture.utils_medical.preproccess as prp
import numpy as np

path = r"/media/kevin/870A38D039F26F71/Datasets/learn2Reg/NLST"

image_dir = os.path.join(path, "imagesTr")
mask_dir = os.path.join(path, "masksTr")

import nowandfuture.utils_medical.aggregation as agg


def test(root, file_path, file_name: str, is_dir: bool, **kwargs):
    target_dir = kwargs.pop("target_dir")
    filter_tuple = kwargs.pop("filter_tuple")
    target = kwargs.pop("target")

    target = file_name if target == "name" else file_path
    voxel = prp.read_one_voxel(file_path)

    slices_ = prp.take_slice(voxel, 0, rot=[0, 0, 0])
    # plot.slices(slices_)
    prp.write2nii(slices_[1], os.path.join(target_dir, "nifti", file_name))
    prp.write2image((prp.bound_normalized(np.rot90(slices_[1])) * 255).astype(np.uint8), os.path.join(target_dir, "png", file_name.split(".")[0] + ".png"))


# agg.async_process(image_dir, "data/slice2d/imagesTr", test)
agg.async_process(mask_dir, "data/slice2d/masksTr", test)
