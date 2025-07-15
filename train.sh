CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7, \
accelerate launch \
    train_Aurora.py \
    --data_root_dir /home/b084020005/era5_tw \
    --output_dir ./post_training_result/testtesttesttesttes \
    --leadtime 1 \
    --rollout_step 1 \
    --train_batch_size 8 \
    --val_batch_size 8 \
    --use_pretrained_weight \
    --epochs 15 \
    --checkpointing_epochs 1 \
    --lr 1e-3 \
    --num_workers 4 \
    --ckpt_prefix AuroraTW \
    --seed 42 \
    --train_start_date_hour "2013-01-01 00:00:00" \
    --train_end_date_hour "2018-12-31 23:00:00" \
    --val_start_date_hour "2022-01-01 00:00:00" \
    --val_end_date_hour "2022-12-31 23:00:00" \
    --surface_variables t2m u10 v10 msl \
    --upper_variables u v t q z \
    --static_variables lsm slt z \
    --levels 1000 925 850 700 500 300 150 50 \
    --latitude 39.75 5 \
    --longitude 100 144.75 \
    --report_to wandb \
