# 3D-medical-image-registration
## 3dimage_resize
resize nii image
## nii2npz 
Convert nii to npz
## voxelmorph
register image of pig liver with voxelmorph(image to image)
## stn
cnn + stn to classificate mnist
## ezdice
parameters:
0 D:\users\zzx\voxelmorph-master\models\ 387
## mask
parameters:
--image_dir D:\users\zzx\data\2017\aft\ --mask_dir D:\users\zzx\data\2017mask\aft --save_dir D:\users\zzx\data\2017\aft\
## crop
normalize .mgz to 0-1 and crop to (160,192,224)
## affine
batch run freesurfer autorecon1
## talairch
batch run freesurfer mri_convert --apply_transform talairach.xfm
