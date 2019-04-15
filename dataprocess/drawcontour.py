import numpy as np
import cv2
from argparse import ArgumentParser
from skimage import io


def draw_mask_edge_on_image_cv2(imagel, mask1, color):
    if color=='red':
        color = (0,0,255)
    else:
        color = (255,0,0)
    image = cv2.imread(imagel)
    print(image.shape)

    image = cv2.resize(image, (512,512), interpolation = cv2.INTER_CUBIC)
    mask = cv2.imread(mask1, 0)
   # image = np.array(image)
   # mask = np.array(mask)
    coef = 255 if np.max(image) < 3 else 1
    image = (image * coef).astype(np.float32)
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(image, contours, -1, color, 3)
    #cv2.imwrite(imagel.split('{}'.format(mask1.split('/')[-2]))[0] + mask1.split('/')[-2] + '/' + mask1.split('/')[-1].split('_')[2] + '.png', image)
    cv2.imwrite('1.png',image)
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--imagel", type=str,
                        help="image")

    parser.add_argument("--mask1", type=str,
                        dest="mask1", default='/media/sdg/zzx/data',
                        help="mask1 folder")

    parser.add_argument("--color", type=str,
                        dest="color", default='/media/sdg/zzx/data',
                        help="mask2 folder")

    args = parser.parse_args()
    draw_mask_edge_on_image_cv2(**vars(args))

