"""
Test models for MICCAI 2018 submission of VoxelMorph.
"""

# py imports
import os
import sys
import glob
import nibabel as nib

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
import keras
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn

# project
sys.path.append('../ext/medipy-lib')
import medipy
import networks
# import util
from medipy.metrics import dice
import datagenerators


def test(gpu_id, model_dir, iter_num, 
         compute_type = 'GPU',  # GPU or CPU
         vol_size=(160,192,224),
         nf_enc=[16,32,32,32],
         nf_dec=[32,32,32,32,16,3],
         save_file=None):
    """
    test via segmetnation propagation
    works by iterating over some iamge files, registering them to atlas,
    propagating the warps, then computing Dice with atlas segmentations
    """  

    # GPU handling
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        # if testing miccai run, should be xy indexing.
        net = networks.miccai2018_net(vol_size, nf_enc, nf_dec, use_miccai_int=False, indexing='ij')  
        net.load_weights(os.path.join(model_dir, str(iter_num) + '.h5'))

        # compose diffeomorphic flow output model
        diff_net = keras.models.Model(net.inputs, net.get_layer('diffflow').output)

        # NN transfer model
        nn_trf_model = networks.nn_trf(vol_size, indexing='ij')

    # if CPU, prepare grid
    if compute_type == 'CPU':
        grid, xx, yy, zz = util.volshape2grid_3d(vol_size, nargout=4)


    # get data
    X_vol = nib.load(r'D:\users\zzx\data\2018nor\pre\01.nii').get_data()
    X_vol = X_vol[np.newaxis, ..., np.newaxis]
    atlas_vol = nib.load(r'D:\users\zzx\data\2018nor\pre\01_a.nii').get_data()
    atlas_vol = atlas_vol[np.newaxis, ..., np.newaxis]

    X_mask = nib.load(r'D:\users\zzx\data\2018mask\pre\pig01_pre_final_r.nii').get_data()
    X_mask = X_mask[np.newaxis, ..., np.newaxis]
    X_mask[X_mask==np.max(X_mask)]=1
    X_mask[X_mask!=1]=0
    atlas_mask = nib.load(r'D:\users\zzx\data\2018mask\aft\pig01_02_final_r.nii').get_data()
    atlas_mask[atlas_mask==np.max(atlas_mask)]=1
    atlas_mask[atlas_mask!=1]=0
    ## feature point
    # X_feapt = np.zeros((160,192,224))
    # X_feapt[128,51,165] = 1
    # X_feapt = X_feapt[np.newaxis,...,np.newaxis]

    # predict transform
    with tf.device(gpu):
        pred = diff_net.predict([X_vol, atlas_vol])

    # Warp segments with flow
    if compute_type == 'CPU':
        flow = pred[0, :, :, :, :]
        warp_seg = util.warp_seg(X_mask, flow, grid=grid, xx=xx, yy=yy, zz=zz)

    else:  # GPU
        warp_mask = nn_trf_model.predict([X_mask, pred])[0,...,0]
        warp_vol = nn_trf_model.predict([X_vol, pred])[0,...,0]
        # pred_point1 = nn_trf_model.predict([X_feapt, pred])[0,...,0]
    print(X_vol.shape)
    # warp_vol = nib.Nifti1Image(warp_vol,np.eye(4))
    warp_vol = nib.Nifti1Image(warp_vol,np.eye(4))
    nib.save(warp_vol,r'D:\users\zzx\data\2018warp\1w.nii')
    # compute Volume Overlap (Dice)
    # X_mask = X_mask[0,...,0]
    # print(X_mask.shape, atlas_mask.shape,pred_point1.shape,np.where(pred_point1 != 0))
    dice_vals = dice(warp_mask, atlas_mask)
    # print('%3d %5.3f %5.3f' % (k, np.mean(dice_vals[:, k]), np.mean(np.mean(dice_vals[:, :k+1]))))
    print(dice_vals)
    if save_file is not None:
        sio.savemat(save_file, {'dice_vals': dice_vals})

if __name__ == "__main__":
    """
    assuming the model is model_dir/iter_num.h5
    python test_miccai2018.py gpu_id model_dir iter_num
    """
    test(sys.argv[1], sys.argv[2], sys.argv[3])
