# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

"""
CartPole-v1是一个倒立摆模型,即小车上有个竖立的杆子，目标是通过左右移动滑块保证倒立杆能够尽可能长时间倒立，最长步骤为500步。

模型控制量action=2,是左0、右1两个。
模型状态量observation=4, 为下面四个：
Num Observation Min Max
0 Cart Position -4.8  4.8
1 Cart Velocity -Inf  Inf
2 Pole Angle  -0.418rad 0.418rad
3 Pole Angular Velocity -Inf  Inf

PPO是同策略，on-policy
"""

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""

    seed: int = 1
    """seed of the experiment"""

    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""

    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""

    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""

    wandb_entity: str = None
    """the entity (team) of wandb's project"""

    # 需要先安装 pip install moviepy
    capture_video: bool = False # 传入 --capture_video 设为true,不需要带值, 不传为False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""

    total_timesteps: int = 500000
    """total timesteps of the experiments"""

    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""

    num_envs: int = 4
    """the number of parallel game environments"""

    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""

    anneal_lr: bool = True # 传入 no-anneal_lr 来设置为False
    """Toggle learning rate annealing for policy and value networks"""

    gamma: float = 0.99
    """the discount factor gamma"""

    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""

    num_minibatches: int = 4
    """the number of mini-batches"""

    update_epochs: int = 4
    """the K epochs to update the policy"""

    norm_adv: bool = True
    """Toggles advantages normalization"""

    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""

    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""

    ent_coef: float = 0.01
    """coefficient of the entropy"""

    vf_coef: float = 0.5
    """coefficient of the value function"""

    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""

    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""

    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""

    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std) # 使用正定矩阵初始化
    torch.nn.init.constant_(layer.bias, bias_const) # 使用0初始化bias
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        # critic最后只需要输出一个分数，因此out_dim=1
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        # actor需要输出各个动作的概率，因此out_dim=action_space
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, observation:torch.Tensor):
        """
        计算当前状态的值分数

        observation: [num_envs, action_num]
        critic_value: [num_envs, 1]
        """
        return self.critic(observation)

    def get_action_and_value(self, observation:torch.Tensor, action=None):
        """
        采样一个action并且计算当前状态的值分数
        observation: [num_envs, action_num]
        """
        # observation: [num_envs, action_num]
        # logits:[num_envs, observation_space_num]
        logits = self.actor(observation) # 即进入softmax之前的值，叫logits
        probs = Categorical(logits=logits)
        if action is None: # action不存在，且采样一个新action
            action = probs.sample()

        # action:[num_envs, ]
        # log_prob:[num_envs, ]
        # entropy:[num_envs, ]
        # critic_value:[num_envs, 1]
        entropy = probs.entropy() # 计算probs的熵，即-p*log(p),即我们希望p的分布越出现两端尖峰越好，而不是均匀分布
        critic_value = self.critic(observation)
        log_prob = probs.log_prob(action) # 注意：这里是取当前action的log_prob, 因此shape:[num_envs, ]
        return action, log_prob, entropy, critic_value


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup, 有多个agent env同时进行实验
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device) #  [num_steps, num_envs, observation_space_num]
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device) #   [num_steps, num_envs, single_action_space_num]
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device) #  [num_steps, num_envs]
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device) # [num_steps, num_envs]
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device) # [num_steps, num_envs]
    critic_values = torch.zeros((args.num_steps, args.num_envs)).to(device) # [num_steps, num_envs]

    # TRY NOT TO MODIFY: start the game
    global_step = 0 # 所有环境的step相加
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed) # [num_envs, observation_space_num]
    next_obs = torch.Tensor(next_obs).to(device) # [num_envs, observation_space_num]
    next_done = torch.zeros(args.num_envs).to(device) # [num_envs, ]

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr: # 默认均为退火lr
            # frac从1逐渐降为0, lr慢慢减小
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lr_now = frac * args.learning_rate # lr随iter衰减，直至0
            optimizer.param_groups[0]["lr"] = lr_now # 不同的group可以用不同的lr

        # 让agent与env交互num_steps次，获取state, reward数据
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # next_obs: [num_envs, observation_space_num]
                # action:[num_envs, ]
                # log_prob:[num_envs, ], 当前动作的log_prob
                # entropy:[num_envs, ]
                # critic_value:[num_envs, 1]
                action, log_prob, _, value = agent.get_action_and_value(next_obs)
                critic_values[step] = value.flatten()

            # actions:[num_steps, num_envs, single_action_space_num]
            # log_probs: [num_steps, num_envs]
            actions[step] = action
            logprobs[step] = log_prob

            # TRY NOT TO MODIFY: execute the game and log data.
            # 注意：这里的instant reward来源于按当前策略采样而来的action与环境交互的instant_reward
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # 当前步如果人为中断或到达终点，则next_done=True
            next_done = np.logical_or(terminations, truncations)
            # 计算instant_reward
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # 交互完成后，计算每局的discount reward
        # bootstrap value if not done
        with torch.no_grad():
            # next_obs: [num_envs, observation_space_num]
            # next_value: [num_envs, 1] -> [num_envs, ]
            next_value = agent.get_value(next_obs).reshape(1, -1) # critic对下一状态的评分
            # advantages: [num_steps, num_envs]
            advantages = torch.zeros_like(rewards).to(device)
            """
            Compute generalized advantage estimate. 
            GAE: 在低方差与低偏置中取得平衡
            
            delta(t) = R(t) + gamma * V(t+1) - V(t), 这就是TD_ERROR
            A(t) = delta(t) + gamma * lambda * A(t+1)
            """
            last_gae_lam = 0
            for t in reversed(range(args.num_steps)): # 按时间逆序计算
                if t == args.num_steps - 1: # 最后时间步
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else: # 非最后时间步
                    nextnonterminal = 1.0 - dones[t + 1]
                    # values: [num_steps, num_envs]
                    nextvalues = critic_values[t + 1]
                # 如果下一状态为终结点，则nextvalues=0, 即value无需预测
                # TD_ERROR = R(t+1) + gamma* V(t+1) - V(t)
                # values: [num_steps, num_envs]
                td_delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - critic_values[t]
                # advantages: [num_steps, num_envs]
                # values: [num_steps, num_envs]
                last_gae_lam = td_delta + args.gamma * args.gae_lambda * nextnonterminal * last_gae_lam # 此时last_gea_lam记当的还是next_gae_lam
                advantages[t] = last_gae_lam
            # A(s,a) = Q(s,a) - V(t) = R(t) + gamma*V(t+1) - V(t)
            # =>  Q(s,a) = A(s,a) + V(t), 其中Q(s, a)即为td target 的reward
            discount_rewards = advantages + critic_values

        # flatten the batch
        # b_obs: [num_steps*num_envs, observation_space_num]
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape) # 两个维度concat
        # b_logprobs: [num_steps*num_envs]
        b_logprobs = logprobs.reshape(-1)
        # b_actions: [num_steps*num_envs, single_action_space_num]
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        # b_advantages: [num_steps*num_envs]
        b_advantages = advantages.reshape(-1)
        # b_discount_rewards: [num_steps*num_envs]
        b_discount_rewards = discount_rewards.reshape(-1)
        # b_critic_values: [num_steps*num_envs]
        b_critic_values = critic_values.reshape(-1)

        # Optimizing the policy and value network
        b_index = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs): # 与环境交互了step次后，进行epoch次模型更新
            np.random.shuffle(b_index) # 每次采样部分数据
            for start in range(0, args.batch_size, args.minibatch_size): # minibatch_size:128
                end = start + args.minibatch_size #
                mb_inds = b_index[start:end]

                # 利用更新后的模型重新计算(observation,state)对应的action_prob, critic_value值
                _, new_log_prob, new_entropy, new_critic_value = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                # 重要性采样
                logratio = new_log_prob - b_logprobs[mb_inds] # log[ P(a|s)/P'(a|s) ]
                ratio = logratio.exp() # ratio = P(a|s)/P'(a|s)

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean() # 近似计算kl散度
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv: # 对advantage进行减均值除方差归一化
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = mb_advantages * ratio
                pg_loss2 = mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = - torch.min(pg_loss1, pg_loss2).mean()

                # Value loss
                new_critic_value = new_critic_value.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (new_critic_value - b_discount_rewards[mb_inds]) ** 2
                    # 新老ciritc_values不能差得太远,差得太远时就截断
                    v_clipped = b_critic_values[mb_inds] + torch.clamp(new_critic_value - b_critic_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_discount_rewards[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    # 计算mse loss
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    # 计算mse loss
                    v_loss = 0.5 * ((new_critic_value - b_discount_rewards[mb_inds]) ** 2).mean()

                # 负熵loss, 一般地没有此项, 其为了使熵越大越好，即action的分布越接近越好？不太理解
                entropy_loss = new_entropy.mean()
                # 注意，此处将actor net,critic net网络的loss相加，两个网络参数同时更新
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # 更新模型参数
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) # 进行原地梯度裁剪
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # 查看critic的预测的方差
        y_pred, y_true = b_critic_values.cpu().numpy(), b_discount_rewards.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("step per second:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
