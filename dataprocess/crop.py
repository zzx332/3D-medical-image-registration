import numpy as np
import cv2
import glob
import os


#im = cv2.imread('/media/sdg/zzx/data/plot_samples/0/LIDC-IDRI-0266_1.3.6.1.4.1.14519.5.2.1.6279.6001.341557859428950960906150406596_179_low_fake_B.png')
images = sorted(glob.glob(os.path.join('/media/sdg/zzx/data/ct/image', '*.png')))
for index in range(len(images)):
    img = cv2.imread(images[index])
    img = cv2.resize(img, (512,512,3), interpolation = cv2.INTER_CUBIC)
    img = img[260:380,110:230,:]
    cv2.imwrite(images[index], img)


