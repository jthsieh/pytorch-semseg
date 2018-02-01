#!/bin/bash


python train.py                 \
  --arch frrnA                  \
  --dataset cityscapes          \
  --name frrnA_epoch90_real \
  --gpus 1                      \
  --img_rows 256                \
  --img_cols 512                \
  --n_epoch 90                  \
  --batch_size 3                \
  --lr_step_size 70             \
  --gamma 0                     \
  --real_synthetic real    \
  --l_rate 0.001

: '
python validate.py         \
  --arch frrnA             \
  --dataset cityscapes     \
  --gpus 2                 \
  --img_rows 256           \
  --img_cols 512           \
  --batch_size 3           \
  --model_path ckpt/frrnA_epoch45_real_synthetic_best_model.pkl
'
