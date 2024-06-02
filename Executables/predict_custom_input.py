# Has To DOs
# Predict for custom input (Currently for 3-channel input only, will modify later for 2-channel input)
# Resize images first to 256, 224, 72
# Make data lists of images
# Use the custom dataloader to load the images directly as arrays and predict masks and store them
# Example: python predict_custom_input.py --dataset_name <Dataset_Name> <other flags if requried>

# Most predicted masks already start around the Liver dome, and stop at VAT end, but some not precisely.
# Therefore, there is an additional settable input argument  --vat_start_liver_end
# to truncate the predicted (SAT, VAT, (Erector?), Bone, Other) above liver dome and (SAT, Bone, Other) below VAT end


import argparse
import numpy as np
import torch
from Utils.lib_utils import interpolate_one_image
import SimpleITK as sitk
import os
import time
from Models.UNetModel import UNet3D
from torch.utils.data import DataLoader
from Utils.lib_custom_dataloader import Custom_Dataset
from tqdm import tqdm
import nibabel as nib


# Misc Params -------------------------------------
pin_memory = True
num_workers = 0

# ------------------------------------------------------------------------------------------------

# TO DO: Add flag to exclude segmentations above liver and below VAT (or output both variations)
# Change output folder name accordingly

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,
                    default='Dataset_Name', help='Name of custom dataset')
parser.add_argument('--is_pre_interpol', action="store_true",
                    help='Whether input is already pre-interpolated to (256, 224, 72)')
parser.add_argument('--in_channels', type=int,
                    default=2, help='Choose between 2-channel (Fat, Water), or 3-channel(Fat, Water, T2*) 3DUNet')
parser.add_argument('--gpus', type=int, default=0, help='free gpus')
parser.add_argument('--fat_keyword', type=str, default='Fat_fused', help='Keyword in image name for Fat')
parser.add_argument('--water_keyword', type=str, default='Water_fused', help='Keyword in image name for Water')
parser.add_argument('--t2star_keyword', type=str, default='T2star_fused', help='Keyword in image name for Water')
parser.add_argument('--pdff_keyword', type=str, default='FF_fused', help='Keyword in image name for Water')
# Maybe this will have a use later, for now the code will simply predict
parser.add_argument('--label_keyword', type=str, default='GroundTruth', help='Keyword in image name for Labels')
parser.add_argument('--liver_start_vat_end', action='store_true',
                    help='Whether to crop the the VAT, SAT, and Other Labels above the liver dome and below VAT end')

args = parser.parse_args()
# ---------------------------------------------------------------------------------------------------------------------
in_channels = args.in_channels
model_store_dir = f'../Trained_Models/3DUNet_{in_channels}_Channel'

if in_channels == 2:
    modalities = ['fat', 'water', 'Segmentation']
elif in_channels == 3:
    modalities = ['fat', 'water', 't2star', 'Segmentation']
else:
    raise ValueError('Number of In_Channels Must be 2 or 3')
# ---------------------------------------------------------------------------------------------------------------------
# Comment this line if using single gpu, with gpu number in the arguments
gpus = [args.gpus]
device = gpus[0]

# # Uncomment and modify this code if using multiple gpus
# gpus = [0]
# device = gpus[0]

device = torch.device(device if torch.cuda.is_available() else 'cpu')


# ------------------------------------------------------------------------------------------------
def get_subdirs(main_dir):
    subdir_list = list()

    for root, subdir, files in os.walk(main_dir):
        # if 'Subject' in subdir:
        subdir_list.append(subdir)

    return subdir_list[0]


# Code to make list of subjects (to make list of subject directories) in Dataset_Name/Subjects_Dir
# If the specific subject dirs are immediate subdirectories of Subjects_Dir,
# or if the specific subject dirs are inside a parent subject directory
def get_subject_list(root_dir):
    out_list = list()
    parent_subj_list = get_subdirs(root_dir)
    for parent_subj in parent_subj_list:
        path_to_parent = os.path.join(root_dir, parent_subj)
        for fname in os.listdir(path_to_parent):
            if os.path.isdir(os.path.join(path_to_parent, fname)):
                out_list.append(f'{parent_subj}/{fname}')
            elif os.path.isfile(os.path.join(path_to_parent, fname)):
                out_list.append(parent_subj)
                break
            else:
                raise ValueError(f'Path: {os.path.join(path_to_parent, fname)} is not a file or folder')
    return out_list


def interpolate_and_save_image(in_path, out_path):
    out_img = interpolate_one_image(in_path, label_keyword=args.label_keyword)
    sitk.WriteImage(out_img, out_path)


def interpolate_and_make_data_dict():
    full_subj_dir = input_subjects_dir

    interpolated_subjects_dir = os.path.join(custom_dataset_dir, 'Interpolated_Subjects_Dir')
    if not args.is_pre_interpol:
        print(f'\nInterpolating Input Images Before Running on Network')
        if not os.path.exists(interpolated_subjects_dir):
            os.makedirs(interpolated_subjects_dir)
        subj_list = get_subject_list(full_subj_dir)
    else:
        subj_list = get_subject_list(interpolated_subjects_dir)

    out_dict = dict()

    loop1 = tqdm(subj_list)
    for subj_path_name in loop1:
        # Subject list can be like XXX_YYYY/XXX_YYYY_V1 or just XXX_YYYY_V1
        if '/' in subj_path_name:
            subj_name = subj_path_name.split('/')[1]
        else:
            subj_name = subj_path_name
        loop1.set_description(subj_name)
        out_dict[subj_name] = dict()
        if not args.is_pre_interpol:
            subj_dir = os.path.join(full_subj_dir, subj_path_name)
        else:
            subj_dir = os.path.join(interpolated_subjects_dir, subj_path_name)

        fat_filename = [f for f in os.listdir(subj_dir) if os.path.isfile(os.path.join(subj_dir, f)) and
                        (args.fat_keyword in f and 'nii.gz' in f)][0]
        print(f'\nFat: {fat_filename}')
        water_filename = [f for f in os.listdir(subj_dir) if os.path.isfile(os.path.join(subj_dir, f)) and
                          (args.water_keyword in f and 'nii.gz' in f)][0]
        print(f'Water: {water_filename}')
        t2star_filename = [f for f in os.listdir(subj_dir) if os.path.isfile(os.path.join(subj_dir, f)) and
                           (args.t2star_keyword in f and 'nii.gz' in f)][0]
        print(f'T2Star: {t2star_filename}')
        # Additional conditions for PDFF name checking
        # pdff_regex = re.compile('..._LION_......')
        pdff_filename = [f for f in os.listdir(subj_dir) if os.path.isfile(os.path.join(subj_dir, f)) and
                         ((args.pdff_keyword in f and 'nii.gz' in f) and (('11.nii.gz' in f or '213.nii.gz' in f) or
                                                      ('01_FF' in f or '15.nii.gz' in f)))][0]
        print(f'PDFF: {pdff_filename}\n')

        fat_path = os.path.join(subj_dir, fat_filename)
        water_path = os.path.join(subj_dir, water_filename)
        t2star_path = os.path.join(subj_dir, t2star_filename)
        pdff_path = os.path.join(subj_dir, pdff_filename)

        if not args.is_pre_interpol:
            interpolated_subj_dir = os.path.join(interpolated_subjects_dir, subj_path_name)
            if not os.path.exists(interpolated_subj_dir):
                os.makedirs(interpolated_subj_dir)
            interpolated_fat_path = os.path.join(interpolated_subj_dir, fat_filename)
            interpolated_water_path = os.path.join(interpolated_subj_dir, water_filename)
            interpolated_t2star_path = os.path.join(interpolated_subj_dir, t2star_filename)
            interpolated_pdff_path = os.path.join(interpolated_subj_dir, pdff_filename)

            loop1.set_description(f'{subj_name} Fat')
            if not os.path.isfile(interpolated_fat_path):
                interpolate_and_save_image(fat_path, interpolated_fat_path)
            loop1.set_description(f'{subj_name} Water')
            if not os.path.isfile(interpolated_water_path):
                interpolate_and_save_image(water_path, interpolated_water_path)
            loop1.set_description(f'{subj_name} T2Star')
            if not os.path.isfile(interpolated_t2star_path):
                interpolate_and_save_image(t2star_path, interpolated_t2star_path)
            loop1.set_description(f'{subj_name} PDFF')
            if not os.path.isfile(interpolated_pdff_path):
                interpolate_and_save_image(pdff_path, interpolated_pdff_path)

            out_dict[subj_name]['fat'] = interpolated_fat_path
            out_dict[subj_name]['water'] = interpolated_water_path
            out_dict[subj_name]['t2star'] = interpolated_t2star_path
            out_dict[subj_name]['pdff'] = interpolated_pdff_path
        else:
            out_dict[subj_name]['fat'] = fat_path
            out_dict[subj_name]['water'] = water_path
            out_dict[subj_name]['t2star'] = t2star_path
            out_dict[subj_name]['pdff'] = pdff_path

    return out_dict


# ------------------------------------------------------------------------------------------------
def predict_and_store_masks():
    print('--' * 30)
    print('Loading Custom Dataloader')
    print('--' * 30)

    input_modes = modalities
    input_modes.remove('Segmentation')
    custom_dataset = Custom_Dataset(subject_data_dict, input_channels=input_modes)
    dataset_size = custom_dataset.__len__()
    custom_loader = DataLoader(dataset=custom_dataset, batch_size=1,
                               shuffle=False, pin_memory=pin_memory, num_workers=num_workers)

    print('--' * 30)
    print(f'Custom Dataset: {dataset_size} Subjects')
    print('--' * 30)

    print('--' * 30)
    print(f'Loading Trained 3D UNet Model')
    print('--' * 30)
    print(f'Device: {device}')

    unet = get_unet_model_with_weights()

    print('--' * 30)
    print(f'Loaded 3D UNet Model With Weights')
    print('--' * 30)

    print('--' * 30)
    print(f'Predicting on Custom Data:')
    print('--' * 30)

    print(f'\nElapsed Time: {time.time() - start_time} seconds\n\n')

    unet.eval()
    with torch.no_grad():
        loop = tqdm(custom_loader, total=len(custom_loader), leave=False)
        for custom_input, subj_name, affine_tensor in loop:
            cur_time = time.time()
            s_n = subj_name[0]

            loop.set_description(f'Subject: {s_n}')

            custom_input = custom_input.to(device)
            raw_pred = unet(custom_input)

            tensor_pred = torch.argmax(torch.softmax(raw_pred, dim=1), dim=1).squeeze(0)

            array_pred = tensor_pred.cpu().detach().numpy()
            array_pred = array_pred.astype(np.float32)

            array_affine = affine_tensor.detach().squeeze(0).cpu().numpy()

            # Print Debug
            print(f'\n\nDebug Stats:\n'
                  f'Input Tensor Shape: {custom_input.shape}\n'
                  f'Raw Prediction Shape: {raw_pred.shape}\n'
                  f'Thresholded Pred Shape: {tensor_pred.shape}\n'
                  f'Predicted Array Shape: {array_pred.shape}\n'
                  f'Affine Matrix: {array_affine}')

            write_array_as_image(array_pred, s_n, array_affine)
            print(f'\n\nPrediction Time of Subject: {s_n}: {time.time() - cur_time} seconds\n')


def crop_top_and_bottom(array_img, name):

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


def write_array_as_image(pred_array, name, affine, datatype='Label'):
    # Using Nibabel for writing

    if args.liver_start_vat_end:
        pred_array = crop_top_and_bottom(pred_array, name)

    image = nib.Nifti1Image(pred_array, affine)

    write_path = os.path.join(output_subjects_dir, f'{name}_pred_inter.nii.gz')

    # write_normalized_image(img_pred, write_path)
    image.to_filename(write_path)


def get_unet_model_with_weights():
    unet = UNet3D(in_channels=in_channels).to(device)

    # We did not use dataparallel as all operations were done on 1 gpu
    # if not use_cpu:
    #     unet = nn.DataParallel(unet, device_ids=gpus)

    print('--' * 30)
    print(f'Loading UNet weights')
    print('--' * 30)

    unet.load_state_dict(load_model(checkpoint_path=os.path.join(model_store_dir,
                                                                 f'3dunet_model_{in_channels}_channel.pth.tar')))

    return unet


def load_model(checkpoint_path):
    filename = os.path.basename(checkpoint_path)
    print(f'Loading {filename}')
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint['model_state']


# ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    custom_dataset_dir = os.path.join('../Custom_Datasets', args.dataset_name)
    input_subjects_dir = os.path.join(custom_dataset_dir, 'Subjects_Dir')

    print(f'Input Subjects Dir: {input_subjects_dir}\n')

    output_subjects_dir = os.path.join(custom_dataset_dir, 'Predicted_Masks')
    if not os.path.exists(output_subjects_dir):
        os.makedirs(output_subjects_dir)

    subject_data_dict = interpolate_and_make_data_dict()

    predict_and_store_masks()

    print(f'\nTotal Execution Time: {time.time() - start_time} seconds\n\n')