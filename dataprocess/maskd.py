import numpy as np
import nibabel as nib
import os
import glob
from argparse import ArgumentParser
import cv2


def maskseg(image_dir, mask):
    images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    mask = cv2.imread(mask)
    for index in range(len(images)):
        img = cv2.imread(images[index])
        img = cv2.resize(img, (512,512), interpolation = cv2.INTER_CUBIC)
        #print(np.max(img), np.min(img))
        #img = 255 - img
        cv2.imwrite('{}'.format(-index) + '.png',img)
        result = np.multiply(img, mask)
        x = index + 1
        #save_file = os.path.join(save_dir, (images[index].split('/')[-1].split('.')[0] + '_seg' + '.nii'))
        cv2.imwrite('{}'.format(x) + '.png',result)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--image_dir", type=str,
                        dest="image_dir",
                        help="image folder")

    parser.add_argument("--mask", type=str,
                        dest="mask",
                        help="mask folder")

    args = parser.parse_args()
    maskseg(**vars(args))

