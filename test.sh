export CUDA_VISIBLE_DEVICES=0
python test.py --model_name 'LFRRN'\
    --save_prefix 'test_LFRRN_p160/' \
    --channel 48 \
    --retrain './pretrained_model/LFRRN_5x5_epoch_298_model.pth'