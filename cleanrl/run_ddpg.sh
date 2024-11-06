
python ddpg_continuous_action.py \
    --seed 1 \
    --env-id Pendulum-v1 \
    --no-cuda \
    --capture_video

# tensorboard 显示
#tensorboard --logdir runs
