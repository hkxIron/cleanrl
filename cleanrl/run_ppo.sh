
python ppo.py \
    --seed 1 \
    --env-id CartPole-v1 \
    --total-timesteps 500000 \
    --capture_video

# tensorboard 显示
#tensorboard --logdir runs_folder
