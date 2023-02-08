#!/bin/bash
set -e

# Note: code partially-adapted from the
#       'https://github.com/google-research/disentanglement_lib' repo


echo "Downloading dSprites dataset."
if [[ ! -d "dsprites" ]]; then
  mkdir dsprites
  wget -O dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
fi
echo "Downloading dSprites completed!"


echo "Downloading shapes3d dataset."
if [[ ! -d "shapes3d" ]]; then
  mkdir shapes3d
  wget -O shapes3d/3dshapes.h5 https://storage.cloud.google.com/3d-shapes/3dshapes.h5

fi
echo "Downloading shapes3d completed!"