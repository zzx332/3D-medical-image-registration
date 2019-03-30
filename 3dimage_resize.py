import numpy as np
import nibabel as nib
import os
import glob
from skimage import transform
from argparse import ArgumentParser


def volresize(image_dir,save_dir):
    images = glob.glob(os.path.join(image_dir, '*.nii'))
    for image in images:
        img = nib.load(image).get_data()
        img = transform.resize(img, (160,192,224))
        newvol = nib.Nifti1Image(img,affine = None)
        save_file = os.path.join(save_dir, (image.split('/')[-1].split('.')[0] + '_r' + '.nii'))
        nib.save(newvol, save_file)

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--image_dir", type=str,
                        help="image")

    parser.add_argument("--save_dir", type=str,
                        dest="save_dir", default='/media/sdg/zzx/data',
                        help="data folder")

    args = parser.parse_args()
    volresize(**vars(args))
