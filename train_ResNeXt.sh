CUDA_VISIBLE_DEVICES=0 python ResNeXt.py \
--model_name put_your_model_name \
--attention_module cbam_block  \
--reduction_ratio 8 \
--learning_rate 0.1 \
--weight_decay 0.0005 \
--momentum 0.9 \
--batch_size 128 \
--total_epoch 100