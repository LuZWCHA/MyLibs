import utils_medical.preproccess as prp
import numpy as np


def process(patch: np.ndarray):
    patch = np.ones_like(patch)

    return patch

if __name__ == '__main__':
    np_arange = np.arange(0, 11, 1, dtype=np.float32)
    print(np_arange)
    XY = np.meshgrid(np_arange, np_arange)
    print(XY[0])


    print(prp.VoxelWalker.pipeline(XY[0], (2, 2), (2, 2), proc_fun=process, skip_dims=0))

    # f, ret = prp.check_patch_move_step((12, 9), (5, 4), (2, 2))
    # print(f, ret)