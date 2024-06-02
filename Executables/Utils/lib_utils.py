import SimpleITK as sitk
import numpy as np
import nibabel as nib

interpolated_image_shape = (256, 224, 72)  # FatSegNet dimension
np.int = int

resample = sitk.ResampleImageFilter()
new_img_size = np.array(interpolated_image_shape, dtype=np.int)
new_img_size = [int(s) for s in new_img_size]


def write_list_to_file(file_list, file_path):
    with open(file_path, 'w') as f:
        for item in file_list:

            f.write(item + '\n')


def get_list_from_file(txt_file, t2star_name_replace=False, add_string=None, using_dark_slice_enhanced_input=False):
    file_list = []
    with open(txt_file) as f:
        file_list = f.read().splitlines()
    if t2star_name_replace:
        file_list = [p.replace('T2S', f'T2S_{add_string}') for p in file_list]
    if using_dark_slice_enhanced_input:
        file_list = [p.replace('MRI', 'MRI_Dark_Slice_Enhanced') for p in file_list]
        file_list = [p.replace('.nii.gz', '_Enhanced.nii.gz') for p in file_list]

    return file_list


def read_pred_and_crop(pred_path):
    raw_pred = nib.load(pred_path)
    affine = raw_pred.affine
    cropped_arr = crop_top_and_bottom(raw_pred.get_fdata())

    return cropped_arr, affine


def crop_top_and_bottom(array_img, name='default_name'):

    # Find liver dome
    liver_dome_z = -1
    found_liver_flag = False
    for slice_num in range(array_img.shape[2]-1, -1, -1):
        z_slice = array_img[:, :, slice_num]
        if not found_liver_flag:
            if 3.0 not in z_slice:
                continue
            else:
                found_liver_flag = True
                liver_dome_z = slice_num
                break

    # Truncate labels above 1 slice above Liver
    for slice_num in range(liver_dome_z + 2, array_img.shape[2]):
        z_slice = array_img[:, :, slice_num]
        z_slice[z_slice == 1] = 0  # SAT
        z_slice[z_slice == 2] = 0  # VAT
        z_slice[z_slice == 4] = 0  # Erector
        z_slice[z_slice == 6] = 0  # Bone
        z_slice[z_slice == 7] = 0  # Other
        array_img[:, :, slice_num] = z_slice

    # Find VAT end
    found_vat_end_flag = False
    vat_end_z = 0
    for slice_num in range(array_img.shape[2]):
        z_slice = array_img[:, :, slice_num]
        if not found_vat_end_flag:
            if 2.0 not in z_slice:
                continue
            else:
                found_vat_end_flag = True
                vat_end_z = slice_num
                break
    # Truncate labels below VAT end
    for slice_num in range(vat_end_z-1, -1, -1):
        z_slice = array_img[:, :, slice_num]
        z_slice[z_slice == 1] = 0  # SAT
        z_slice[z_slice == 6] = 0  # Bone
        z_slice[z_slice == 7] = 0  # Other
        array_img[:, :, slice_num] = z_slice

    if (not found_liver_flag) or (not found_vat_end_flag):
        raise ValueError(f'Something is wrong with the predicted mask of {name}')

    return array_img


def interpolate_one_image(img_path, label_keyword='label'):
    raw_img = sitk.ReadImage(img_path)

    raw_img_size = np.array(raw_img.GetSize(), dtype=np.int)
    raw_img_spacing = np.array(raw_img.GetSpacing())

    new_img_spacing = (raw_img_size/new_img_size) * raw_img_spacing

    if label_keyword not in img_path:
        resample.SetInterpolator(sitk.sitkLinear)
    else:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)

    new_image = resample_one_image(raw_img, new_img_size, new_img_spacing)

    return new_image


def resample_one_image(in_image, size, spacing):
    resample.SetSize(size)
    resample.SetOutputSpacing(spacing)
    resample.SetOutputDirection(in_image.GetDirection())
    resample.SetOutputOrigin(in_image.GetOrigin())

    return resample.Execute(in_image)






