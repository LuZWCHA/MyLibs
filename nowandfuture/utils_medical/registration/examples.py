import ants
import SimpleITK as itk
import numpy as np


# this codes is all from https://antspy.readthedocs.io/en/latest/registration.html, to see more information goto the link.

def ant_base_tutorial(image_path, ):
    # image read
    img = ants.image_read(image_path)
    # read from include data: for example mni, r16...
    img = ants.image_read(ants.get_ants_data('mni'))
    # convert to numpy
    image_array = img.numpy()
    # do some operations...
    image_array += 5

    # copies the image information and just changes the data
    new_img1 = img.new_image_like(image_array)

    # only data copied
    new_img2 = ants.from_numpy(image_array)

    # verbose way to copy information
    new_img3 = ants.from_numpy(image_array, spacing=img.spacing,
                               origin=img.origin, direction=img.direction)

    # index like array

    vals = img[200, :, :]

    img[100, :, :] = 1

    img2 = img.clone()
    # "+" has been overloaded
    img3 = img + img2
    # check the array is same or not
    print(np.allclose(img.numpy() + img2.numpy(), img3.numpy()))

    img = ants.image_read(ants.get_ants_data('r16'))
    mask = ants.get_mask(img)

    img = img.resample_image((64, 64), 1, 0).get_mask().atropos(m='[0.2,1x1]', i='kmeans[3]', x=mask)

    is_same_physical_space = ants.image_physical_space_consistency(img2, img)

    # Segmentation
    img = img.resample_image((64, 64), 1, 0)
    img_seg = ants.atropos(a=img, m='[0.2,1x1]', i='kmeans[3]', x=mask)

    print(img_seg.keys())
    ants.plot(img_seg['segmentation'])

    # Cortical thickness 皮质厚度
    # get 2d data
    img = ants.image_read(ants.get_ants_data('r16'), dimension=2)
    # mask
    mask = ants.get_mask(img).threshold_image(1, 2)  # 1, 2 as the threshold
    # ... see more at 10 minus tutorial
    ###########################################

    # Registration
    fixed = ants.image_read(ants.get_ants_data('r16')).resample_image((64, 64), 1, 0)
    moving = ants.image_read(ants.get_ants_data('r64')).resample_image((64, 64), 1, 0)
    fixed.plot(overlay=moving, title="Before Registration")
    mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='SyN')
    print(mytx)
    warped_moving = mytx['warpedmovout']
    fixed.plot(overlay=warped_moving, title='After Registration')

    # use the transforms output ...
    mywarpedimage = ants.apply_ants_transform(fixed=fixed, moving=moving, transform=mytx['fwdtransforms'])
    mywarpedimage.plot()


def registration_example(fixed, moving, method='SyN', show=False):
    if show:
        fixed.plot(overlay=moving, title="Before Registration")
    mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform=method, reg_iterations=[40, 40, 0],
                             multivariate_extras=[("CC", fixed, moving, 1.5, 4)])
    warped_moving = mytx['warpedmovout']
    if show:
        fixed.plot(overlay=warped_moving, title='After Registration')

    # # use the transforms output ...
    # mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'])
    # mywarpedimage.plot()

    return warped_moving, moving, fixed, mytx['fwdtransforms']


def registration2(fixed_path, moving_path, method='SyNOnly', show=False):
    fixed = ants.image_read(fixed_path)
    moving = ants.image_read(moving_path)
    return registration_example(fixed, moving, method, show)


def registration_np(fixed: np.ndarray, moving: np.ndarray, method='SyNOnly', show=False):
    fixed = ants.from_numpy(fixed)
    moving = ants.from_numpy(moving)
    return registration_example(fixed, moving, method, show)


def ant_transform(moving_image, fixed_image, transforms, interpolation='linear', show=False):
    # use the transforms output ...
    mywarpedimage = ants.apply_transforms(fixed=fixed_image, moving=moving_image, transformlist=transforms, interpolation=interpolation)
    if show:
        mywarpedimage.plot()
    return mywarpedimage


def ant_transform2(image, transforms, interpolation='linear', show=False):
    # use the transforms output ...
    mywarpedimage = ants.apply_ants_transform_to_image(transforms, image, image, interpolation=interpolation)
    if show:
        mywarpedimage.plot()

    return mywarpedimage
