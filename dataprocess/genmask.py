import numpy as np
import cv2

image = cv2.imread("/media/sdg/zzx/data/plot_samples/0/LIDC-IDRI-0266_1.3.6.1.4.1.14519.5.2.1.6279.6001.341557859428950960906150406596_179.png")


b  = np.array([[[110, 260],  [230,260], [230,380],[110,380]]], dtype = np.int32)

im = np.zeros(image.shape[:2], dtype = "uint8")
cv2.polylines(im, b, 1, 255)
cv2.fillPoly(im, b, 255)

mask = im
mask = (mask)/255
#print(np.max(mask), np.min(mask))
cv2.imwrite('test.png', mask)
