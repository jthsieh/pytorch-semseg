#!/bin/bash

: '
python train.py                 \
  --arch frrnA                  \
  --dataset cityscapes          \
  --name frrnA_epoch45_real_synthetic \
  --gpus 2                      \
  --img_rows 256                \
  --img_cols 512                \
  --n_epoch 45                  \
  --batch_size 3                \
  --lr_step_size 35             \
  --gamma 0                     \
  --real_synthetic real+synthetic    \
  --l_rate 0.001
'

python val.py              \
  --arch frrnA             \
  --dataset cityscapes     \
  --gpus 0                 \
  --img_rows 256           \
  --img_cols 512           \
  --batch_size 3           \
  --resume ckpt/frrnA_epoch45_lr-3_bs3_best_model.pkl

