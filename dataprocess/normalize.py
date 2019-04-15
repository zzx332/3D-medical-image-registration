import numpy as np
import glob
import nibabel as nib
from skimage import transform
from argparse import ArgumentParser
import os
import cv2
import scipy.ndimage


def volresize(image_dir,save_dir):
    images = sorted(glob.glob(os.path.join(image_dir, '*.nii')))
    for image in images:
        print(image)
        img = nib.load(image).get_data()
        s = len(img[1,1,:])
        print(s,np.max(img),np.min(img))
        x = 224/s        
        img[img>255] = 255
        img[img<0] = 0
        img = img/255
        img = scipy.ndimage.zoom(img, (0.3125,0.375,x),order=0)
        #img = transform.resize(img,(160,192,224))
        #x = np.max(img)
        #y = np.min(img)
        #img = (img-y)/(x-y)
        print(img,np.max(img), np.min(img))
        newvol = nib.Nifti1Image(img,np.eye(4))
        save_file = os.path.join(save_dir, (image.split('/')[-1].split('.')[0] + '_r' + '.nii'))
        nib.save(newvol, save_file)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--image_dir", type=str,
                        dest="image_dir",default='../2018seg/pre',
                        help="image")

    parser.add_argument("--save_dir", type=str,
                        dest="save_dir",default='../atlas/',
                        help="data folder")

    args = parser.parse_args()
    volresize(**vars(args))

