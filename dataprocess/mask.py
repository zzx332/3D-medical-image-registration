import numpy as np
import nibabel as nib
import os
import glob
from argparse import ArgumentParser


def maskseg(image_dir, mask_dir, save_dir):
    images = sorted(glob.glob(os.path.join(image_dir, '*.nii')))
    masks = sorted(glob.glob(os.path.join(mask_dir, '*.nii')))
    for index in range(len(images)):
        img = nib.load(images[index]).get_data()
        img = np.array(img)
        mask = nib.load(masks[index]).get_data()
        mask = np.array(mask)
        assert img.shape==mask.shape, "{},{}".format(images[index], masks[index])
        result = np.multiply(img, mask)
        newvol = nib.Nifti1Image(result, np.eye(4))
        save_file = os.path.join(save_dir, (images[index].split('/')[-1].split('.')[0] + '_seg' + '.nii'))
        nib.save(newvol, save_file)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--image_dir", type=str,
                        dest="image_dir",
                        help="image folder")

    parser.add_argument("--mask_dir", type=str,
                        dest="mask_dir",
                        help="mask folder")

    parser.add_argument("--save_dir", type=str,
                        dest="save_dir", default='/media/sdg/zzx/data',
                        help="data folder")

    args = parser.parse_args()
    maskseg(**vars(args))
      
