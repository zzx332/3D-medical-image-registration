import nibabel as nib
import os 
import numpy as np
from argparse import ArgumentParser


def nii2npz(image, save_path):
    img = nib.load(image).get_data()
    img = np.array(img)
    npy_name = save_path +str(image).split('/')[-1].split('.')[0]
    np.savez("{}.npz".format(npy_name), vol = img)


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--image", type=str,
                        help="imagepath")

    parser.add_argument("--save_path", type=str,
                        dest="save_path", default='/media/sdg/zzx/voxelmorph/data/',
                        help="npz save path")                        

    args = parser.parse_args()
    nii2npz(**vars(args))

