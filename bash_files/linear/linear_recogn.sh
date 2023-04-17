python3 main_linear.py \
    --dataset lfwpairs \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --train_dir lfwpairs/train \
    --val_dir lfwpairs/val \
    --max_epochs 30 \
    --gpus 0 \
    --precision 32 \
    --optimizer adam \
    --scheduler step \
    --lr 0.00025 \
    --lr_decay_steps 60 80 \
    --weight_decay 1e-6 \
    --batch_size 128 \
    --num_workers 4 \
    --name lfwpairs-ssl-mult-linear \
    --project semi-supervised-learning \
    --entity alexneakameni \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --wandb \
    --cat_strategy mult \
    --save_checkpoint