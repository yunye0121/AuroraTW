CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7, \
accelerate launch \
    ar_gen_eval_Aurora.py \
    --data_root_dir /home/b084020005/era5_tw \
    --checkpoint_path "/home/b084020005/AuroraTW/post_training_result/testtesttesttesttes/ckpts/checkpoint-1.ckpt" \
    --batch_size 8 \
    --num_workers 4 \
    --seed 42 \
    --start_date_hour "2023-01-01 00:00:00" \
    --end_date_hour "2023-12-31 23:00:00" \
    --surface_variables t2m u10 v10 msl \
    --upper_variables u v t q z \
    --static_variables lsm slt z \
    --levels 1000 925 850 700 500 300 150 50 \
    --latitude 39.75 5 \
    --longitude 100 144.75 \
    --lead_time 1 \
    --rollout_step 24 \
    --save_lead_time 6 \
