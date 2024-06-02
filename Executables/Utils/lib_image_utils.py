import nibabel as nib
import numpy as np
import os


class NImageUtils:
    def read_image(self, image_path, image_shape):
        image = nib.load(os.path.abspath(image_path))
        image = self.fix_shape(image)
        return image

    def fix_shape(self, image):  # If Shape of Image is [..., 1]
        if image.shape[-1] == 1:
            image = image.__class__(dataobj=np.squeeze(image.get_fdata()), affine=image.affine)
        return image


