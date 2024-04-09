export CUDA_VISIBLE_DEVICES=0,1
python train.py --MGPU 2 --channel 48 --batch_size 2 --patch_size 96 --save_prefix 'train_LFRRN_p96/' \
    --path_for_train './data' \
    --epoch 301 --n_steps 60