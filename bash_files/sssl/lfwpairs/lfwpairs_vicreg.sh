python3 main_continual.py \
    --dataset lfwpairs \
    --encoder resnet18 \
    --data_dir $DATA_DIR \
    --max_epochs 500 \
    --gpus 0 \
    --num_workers 4 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 1.0 \
    --classifier_lr 0.1 \
    --weight_decay 1e-4 \
    --batch_size 128 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --solarization_prob 0.0 0.2 \
    --name vicreg-lfwpairs-ssl \
    --project semi-supervised-learning \
    --entity alexneakameni \
    --save_checkpoint \
    --method vicreg \
    --proj_hidden_dim 2048 \
    --output_dim 2048 \
    --wandb \
    --offline