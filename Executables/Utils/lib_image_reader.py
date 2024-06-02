import os
import collections
from Utils.lib_image_utils import NImageUtils


class ImageReader:
    def __init__(self, image_shape, feature_indices=None):
        self.image_shape = image_shape

        if feature_indices is None:
            self.feature_indices = []
        elif not isinstance(feature_indices, collections.Iterable) or isinstance(feature_indices, str):
            self.feature_indices = [feature_indices]

        self.background_val = 0
        self.tolerance = 0.00001
        self.img_utils = NImageUtils()

    def read_image_set(self, subject_file_set):
        # File set: ('img1_FF', 'img1_W',... 'img1_Label')

        image_list = list()
        for index, image_path in enumerate(subject_file_set):
            # Read Single Image
            image_list.append(self.img_utils.read_image(image_path=image_path, image_shape=self.image_shape))
        return image_list
