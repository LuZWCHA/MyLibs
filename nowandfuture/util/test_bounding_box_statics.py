import glob
import os

from nowandfuture.util import plot
from nowandfuture.util.preproccess import n4_bias_field_correction_sitk as n4b
import nowandfuture.util.preproccess as prp
import numpy as np

# path = 'data'
# images = glob.glob(os.path.join(path, '*.dcm'))
# image_path = images[0]
# print(image_path)
# image = prp.read_dcm_dicom(image_path)
# print(image)
# org_image = prp.convert2numpy(image)[0]
# print(org_image.shape)
# res = n4b(image, convert_numpy=True)
# print(res.shape)
# plot.slices([org_image, res], cmaps=['gray'])

import nowandfuture.util.aggregation as agg

path = r'G:\腹部数据'
to = r'G:\Processed Images'

def test(root, file_path, file_name: str, is_dir: bool, **kwargs):
    target_dir = kwargs.pop("target_dir")
    filter_tuple = kwargs.pop("filter_tuple")
    target = kwargs.pop("target")

    target = file_name if target == "name" else file_path
    rel_path = os.path.relpath(file_path + '.nii.gz', root)

    target_file = os.path.join(target_dir, rel_path)
    if os.path.exists(target_file):
        return
    # resample
    is_mr = 'MR' in file_path
    if is_dir and ('V_' in file_name or 'P_' in file_name):
        print(file_path)
        voxel, meta = prp.read_dcms_dicom(file_path)
        voxel, new_space = prp.resample(voxel, old_spacing=meta)

        print('resample: ', meta, new_space)
        print(voxel.shape)

        # n4_bias_field_correction by SimpleITK
        # if is_mr:
        #     print('n4_bias_field_correction...')
        #     voxel = prp.n4_bias_field_correction_sitk(voxel, convert_numpy=True)

        prp.write2nii(voxel, target_file)



# agg.process_files_match_from_to(path, to, test)


import SimpleITK as sitk
import sys
import os


def command_iteration(method):
    if (method.GetOptimizerIteration() == 0):
        print(f"\tLevel: {method.GetCurrentLevel()}")
        print(f"\tScales: {method.GetOptimizerScales()}")
    print(f"#{method.GetOptimizerIteration()}")
    print(f"\tMetric Value: {method.GetMetricValue():10.5f}")
    print(f"\tLearningRate: {method.GetOptimizerLearningRate():10.5f}")
    if (method.GetOptimizerConvergenceValue() != sys.float_info.max):
        print(f"\tConvergence Value: {method.GetOptimizerConvergenceValue():.5e}")


def command_multiresolution_iteration(method):
    print(f"\tStop Condition: {method.GetOptimizerStopConditionDescription()}")
    print("============= Resolution Change =============")


if len(sys.argv) < 4:
    print("Usage:", sys.argv[0], "<fixedImageFilter> <movingImageFile>",
          "<outputTransformFile>")
    sys.exit(1)
sys.argv[2] = r'D:\迅雷下载\data\P_481\CT_V\45.png'
sys.argv[1] = r'D:\迅雷下载\data\P_1\CT_V\59.png'
# sys.argv[1] = r'D:\迅雷下载\data\P_36\CT_V\16.png'
# sys.argv[2] = r'D:\迅雷下载\data\P_1\CT_V\69.png'
sys.argv[3] = '1.hdf5'

# fixed = sitk.ReadImage(r'D:\Projects\Python\DL\ClockInDemo\data\img-3.png', sitk.sitkFloat32)

fixed = prp.read_image(sys.argv[1]).astype(np.float32)
moving = prp.read_image(sys.argv[2]).astype(np.float32)

fixed = sitk.GetImageFromArray(fixed)
moving = sitk.GetImageFromArray(moving)


# fixed = sitk.ReadImage(sys.argv[1], sitk.sitkFloat32)
#
# moving = sitk.ReadImage(sys.argv[2], sitk.sitkFloat32)

initialTx = sitk.CenteredTransformInitializer(fixed, moving,
                                              sitk.AffineTransform(
                                                  fixed.GetDimension()))

R = sitk.ImageRegistrationMethod()

R.SetShrinkFactorsPerLevel([3, 2, 1])
R.SetSmoothingSigmasPerLevel([2, 1, 1])

R.SetMetricAsJointHistogramMutualInformation(20)
R.MetricUseFixedImageGradientFilterOff()

R.SetOptimizerAsGradientDescent(learningRate=1.0,
                                numberOfIterations=100,
                                estimateLearningRate=R.EachIteration)
R.SetOptimizerScalesFromPhysicalShift()

R.SetInitialTransform(initialTx)

R.SetInterpolator(sitk.sitkLinear)

R.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(R))
R.AddCommand(sitk.sitkMultiResolutionIterationEvent,
             lambda: command_multiresolution_iteration(R))

outTx1 = R.Execute(fixed, moving)

print("-------")
print(outTx1)
print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
print(f" Iteration: {R.GetOptimizerIteration()}")
print(f" Metric value: {R.GetMetricValue()}")

displacementField = sitk.Image(fixed.GetSize(), sitk.sitkVectorFloat64)
displacementField.CopyInformation(fixed)
displacementTx = sitk.DisplacementFieldTransform(displacementField)
del displacementField
displacementTx.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0.0,
                                            varianceForTotalField=1.5)

R.SetMovingInitialTransform(outTx1)
R.SetInitialTransform(displacementTx, inPlace=True)

R.SetMetricAsANTSNeighborhoodCorrelation(4)
R.MetricUseFixedImageGradientFilterOff()

R.SetShrinkFactorsPerLevel([3, 2, 1])
R.SetSmoothingSigmasPerLevel([2, 1, 1])

R.SetOptimizerScalesFromPhysicalShift()
R.SetOptimizerAsGradientDescent(learningRate=1,
                                numberOfIterations=300,
                                estimateLearningRate=R.EachIteration)

R.Execute(fixed, moving)

print("-------")
print(displacementTx)
print(f"Optimizer stop condition: {R.GetOptimizerStopConditionDescription()}")
print(f" Iteration: {R.GetOptimizerIteration()}")
print(f" Metric value: {R.GetMetricValue()}")

# compositeTx = sitk.CompositeTransform([outTx1, displacementTx])
sitk.WriteTransform(displacementTx, sys.argv[3])

# if ("SITK_NOSHOW" not in os.environ):
# sitk.Show(displacementTx.GetDisplacementField(), "Displacement Field")

resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(fixed)
resampler.SetInterpolator(sitk.sitkLinear)
resampler.SetDefaultPixelValue(0)
resampler.SetTransform(displacementTx)

out = resampler.Execute(moving)
simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)

moved,_ = prp.convert2numpy(simg2)
transform = displacementTx.GetDisplacementField()
print(moved, transform)
# prp.write2image(moved, 'moved_v_v.png')
# prp.write2image(prp.convert2numpy(fixed)[0], 'fixed_ctv_p36_16.png')
# prp.write2image(prp.convert2numpy(moving)[0], 'moving_ctv_p1_69.png')
plot.slices([prp.convert2numpy(moving)[0], prp.convert2numpy(fixed)[0], moved], cmaps=['gray'])
# sitk.Show(cimg, "ImageRegistration1 Composition")
