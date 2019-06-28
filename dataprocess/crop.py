# crop and normalization

import glob
import numpy as np
import nibabel as nib

images = sorted(glob.glob('../brain*'))
for image in images:
    vol = nib.load(image).get_data()
    vol = vol/255
    vol = vol[48:-48,31:-33,3:-29]
    newvol = nib.Nifti1Image(vol,affine=None)
    nib.save(newvol, './train_vol'+image.split('n')[-1].split('.')[0]+'.mgz')
