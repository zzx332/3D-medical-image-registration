#! /bin/bash

# 批量进行talairach

for i in `seq 10 99`;do
mri_convert ./ixi$i/mri/brainmask.mgz --apply_transform ./ixi$i/mri/transforms/talairach.xfm brain$i.mgz
done