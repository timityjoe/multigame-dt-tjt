# Original implementation:
# https://github.com/etaoxing/multigame-dt
# https://github.com/google-research/google-research/tree/master/multi_game_dt
# Paper
# https://arxiv.org/abs/2205.15241

import functools
import os
import pickle
import random

import gym
import numpy as np
import scipy
import torch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

gym.logger.set_level(gym.logger.ERROR)

from atari.atari_data import get_human_normalized_score
from atari.atari_preprocessing import AtariPreprocessing

from mingpt.multigame_dt import MultiGameDecisionTransformer
from load_pretrained import load_jax_weights

from sequence_environment_wrapper import SequenceEnvironmentWrapper
from batch_rollout1 import batch_rollout1

from loguru import logger
# logger.remove()
# logger.add(sys.stdout, level="INFO")
# logger.add(sys.stdout, level="SUCCESS")
# logger.add(sys.stdout, level="WARNING")

# --- Setup
logger.info("0) Setup Seed & GPU")
seed = 100
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)

# No GPU: take over whatever gpus are on the system
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Hide GPU from tf, since tf.io.encode_jpeg/decode_jpeg seem to cause GPU memory leak.
tf.config.set_visible_devices([], "GPU")

# Use GPU(s)
# device = 'cpu'
# if torch.cuda.is_available():
#     logger.info("GPU available.")
#     device = torch.cuda.current_device()


# --- Create environments

# from https://github.com/facebookresearch/moolib/blob/06e7a3e80c9f52729b4a6159f3fb4fc78986c98e/examples/atari/environment.py
def create_env(env_name, env_id, sticky_actions=False, noop_max=30, terminal_on_life_loss=False):
    logger.info(f"create_env() env_id:{env_id}, env_name:{env_name} sticky_actions:{sticky_actions} noop_max:{noop_max} terminal_on_life_loss:{terminal_on_life_loss}")
    env = gym.make(  # Cf. https://brosa.ca/blog/ale-release-v0.7
    # env = gym.vector.make(  # Cf. https://brosa.ca/blog/ale-release-v0.7
        f"ALE/{env_name}-v5",
        obs_type="grayscale",  # "ram", "rgb", or "grayscale".
        # obs_type="rgb",  # "ram", "rgb", or "grayscale".
        frameskip=1,  # Action repeats. Done in wrapper b/c of noops.
        repeat_action_probability=0.25 if sticky_actions else 0.0,  # Sticky actions.
        max_episode_steps=108000 // 4,
        full_action_space=True,  # Use all actions.

        # Mod by Tim: For rendering purposes
        # render_mode=None,  # None, "human", or "rgb_array".
        # render_mode='human',  # None, "human", or "rgb_array".
        render_mode='rgb_array',  # None, "human", or "rgb_array".
    )

    # Using wrapper from seed_rl in order to do random no-ops _before_ frameskipping.
    # gym.wrappers.AtariPreprocessing doesn't play well with the -v5 versions of the game.
    env = AtariPreprocessing(
        env,
        env_id,
        frame_skip=4,
        terminal_on_life_loss=terminal_on_life_loss,
        screen_size=84,
        max_random_noops=noop_max,  # Max no-ops to apply at the beginning.
    )
    # env = gym.wrappers.FrameStack(env, num_stack=4)  # frame stack done separately
    env = SequenceEnvironmentWrapper(env, num_stack_frames=4, jpeg_obs=True)
    return env

# env_name = "Breakout"
# env_name = "Assault"
# env_name = "Atlantis"
env_name = "Seaquest"

# num_envs = 8
num_envs = 1
# env_fn = lambda: create_env(env_name)
# envs = [env_fn() for _ in range(num_envs)]

envs = [create_env(env_name, env_id=i) for i in range(num_envs)]
logger.info(f"1) Create Env, num_envs: {num_envs}, envs[0]:{envs[0]}")

# --- Create offline RL dataset

# --- Create model
OBSERVATION_SHAPE = (84, 84) # Grayscale scenario, in turn decomposed into 6x6=36patches of 14px by 14px each
PATCH_SHAPE = (14, 14)
NUM_ACTIONS = 18  # Maximum number of actions in the full dataset.
# rew=0: no reward, rew=1: score a point, rew=2: end game rew=3: lose a point
NUM_REWARDS = 4
RETURN_RANGE = [-20, 100]  # A reasonable range of returns identified in the dataset

# See Table 1 of 
# https://arxiv.org/abs/2205.15241
# This is ~200M Params, Training time 8 days on 64 TPUv4
model = MultiGameDecisionTransformer(
    img_size=OBSERVATION_SHAPE,
    patch_size=PATCH_SHAPE,
    num_actions=NUM_ACTIONS,
    num_rewards=NUM_REWARDS,
    return_range=RETURN_RANGE,
    d_model=1280,
    num_layers=10,
    dropout_rate=0.1,
    predict_reward=True,
    single_return_token=True,
    conv_dim=256,
)
logger.info(f"2) Create MultiGameDecisionTransformer model:{model}")
# print(model)

# --- Load pretrained weights
logger.info(f"3) Load pretrained weights")

model_params, model_state = pickle.load(open("checkpoint_38274228.pkl", "rb"))
load_jax_weights(model, model_params)
model = model.to(device=device)

# --- Train model
logger.info(f"4) Train model")
model.train()

# --- Save/Load model weights
# torch.save(model.state_dict(), "model.pth")
# model.load_state_dict(torch.load("model.pth"))

# --- Evaluate model
logger.info(f"5) Evaluate model")
model.eval()
optimal_action_fn = functools.partial(
    model.optimal_action,
    return_range=RETURN_RANGE,
    single_return_token=True,
    opt_weight=0,
    num_samples=128,
    action_temperature=1.0,
    return_temperature=0.75,
    action_top_percentile=50,
    return_top_percentile=None,
    torch_device = device
)

# --- Extract attention map(s)
# logger.info(f"X) Get Attention Map(s)")
# np_attention_map = model.get_attention_map()


# --- Calculate Results
logger.info(f"6) Calculate Results")
task_results = {}
# task_results["rew_sum"] = batch_rollout1(device, envs, model, optimal_action_fn, num_episodes=16, log_interval=100)
task_results["rew_sum"] = batch_rollout1(device, envs, model, optimal_action_fn, num_episodes=160, log_interval=100)
[env.close() for env in envs]

# --- Log metrics
logger.info(f"7) Log metrics")
def print_metrics(metric):
    logger.info(f"    mean: {np.mean(metric):.2f}")
    logger.info(f"    std: {np.std(metric):.2f}")
    logger.info(f"    median: {np.median(metric):.2f}")
    logger.info(f"    iqm: {scipy.stats.trim_mean(metric, proportiontocut=0.25):.2f}")


logger.info("Reward Sum:")
print_metrics(task_results["rew_sum"])

logger.info("-" * 10)

task_results["human_normalized_score"] = [
    get_human_normalized_score(env_name.lower(), score) for score in task_results["rew_sum"]
]
logger.info("Human Normalized Score:")
print_metrics(task_results["human_normalized_score"])
