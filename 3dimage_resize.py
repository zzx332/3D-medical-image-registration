import numpy as np
import nibabel as nib
import os
from skimage import transform
from argparse import ArgumentParser


def volresize(image,save_dir):
    img = nib.load(image).get_data()
    img = transform.resize(img, (160,192,224))
    newvol = nib.Nifti1Image(img,affine = None)
    save_file = os.path.join(save_dir, (image.split('/')[-1].split('.')[0] + '_r' + '.nii'))
    nib.save(newvol, save_file)

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--image", type=str,
                        help="image")

    parser.add_argument("--save_dir", type=str,
                        dest="save_dir", default='/media/sdg/zzx/data',
                        help="data folder")

    args = parser.parse_args()
    volresize(**vars(args))
