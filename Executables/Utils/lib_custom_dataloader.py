import torch
import numpy as np
import os
from torch.utils.data import Dataset
from Utils.lib_image_reader import ImageReader


interpolated_image_shape = (256, 224, 72)  # FatSegNet dimension


class Custom_Dataset(Dataset):
    def __init__(self, in_subj_dict, input_channels, image_shape=interpolated_image_shape):
        self.subj_data_dict = in_subj_dict
        self.subj_list = list(self.subj_data_dict.keys())
        print(f'\n\nSubject List: {self.subj_list}\n\n')

        self.image_shape = image_shape
        self.input_channels = input_channels
        self.num_channels = len(input_channels)
        self.model_store_dir = f'../Trained_Models/3DUNet_{self.num_channels}_Channel'

        self.image_reader = ImageReader(self.image_shape, self.num_channels)

        self.mean = 0
        self.std = 1

        self.obtain_norm_stats()

    def __len__(self):
        return len(self.subj_list)

    def _load_data(self, idx):
        subj_name = self.subj_list[idx]
        imgs_arr = [self.subj_data_dict[subj_name][modality] for modality in self.input_channels]

        images = self.image_reader.read_image_set(imgs_arr)
        array_images = [image.get_fdata() for image in images]

        array_images = self.normalize_input(array_images)

        tensor_input = torch.tensor(np.asarray(array_images[:self.num_channels]), dtype=torch.float32)
        affine = torch.tensor(images[0].affine)

        return tensor_input, affine

    def __getitem__(self, index):
        tensor_input, affine = self._load_data(index)

        subj_name = self.subj_list[int(index)]

        return tensor_input, subj_name, affine

    def obtain_norm_stats(self):
        stats_path = os.path.join(self.model_store_dir, 'data_stats.npz')
        stats = np.load(stats_path)
        self.mean, self.std = stats['mean'], stats['std']

    def normalize_input(self, array_images):
        array_images = np.asarray(array_images[:self.num_channels])
        array_images -= self.mean[:, np.newaxis, np.newaxis, np.newaxis]
        array_images /= self.std[:, np.newaxis, np.newaxis, np.newaxis]
        return array_images




