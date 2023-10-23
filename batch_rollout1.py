
import time
import torch
import numpy as np
import random

from tqdm import tqdm
from loguru import logger
# logger.remove()
# logger.add(sys.stdout, level="INFO")
# logger.add(sys.stdout, level="SUCCESS")
# logger.add(sys.stdout, level="WARNING")


# --- Evaluate model
def batch_rollout1(device, envs, policy_fn, num_episodes, log_interval=None):
    r"""Roll out a batch of environments under a given policy function."""
    num_batch = len(envs)
    num_steps = envs[0].spec.max_episode_steps

    num_epoch = range(num_episodes // num_batch)
    logger.info(f"batch_rollout1() - num_envs:{len(envs)} num_ep:{num_episodes} num_steps:{num_steps} num_batch:{num_batch} num_epoch:{num_epoch} ")
    logger.info(f"   envs[0]:{envs[0]} ")

    assert num_episodes % num_batch == 0

    rng = torch.Generator()
    seeds_list = [random.randint(0, 2**32 - 1) for _ in range(num_episodes)]
    print(f"seeds: {seeds_list}")

    rew_sum_list = []


    # for c in tqdm(range(num_episodes // num_batch)):
    for c in range(num_episodes // num_batch):
        seeds = seeds_list[c * num_batch : (c + 1) * num_batch]
        rng.manual_seed(seeds[0])

        obs_list = [env.reset(seed=seeds[i]) for i, env in enumerate(envs)]
        obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}
        rew_sum = np.zeros(num_batch, dtype=np.float32)
        done = np.zeros(num_batch, dtype=np.int32)
        start = time.perf_counter()

        pbar = tqdm(range(num_steps))

        for t in pbar:
            #-----------------------------------------
            # Mod by Tim: 
            # Render to see whats going on.. 
            # Embedded in atari_preprocessing.py - AtariPreprocessing._fetch_grayscale_observation()
            #-----------------------------------------

            done_prev = done
            obs = {k: torch.tensor(v, device=device) for k, v in obs.items()}
            actions = policy_fn(obs, rng=rng, deterministic=False)

            # Collect step results and stack as a batch.
            step_results = [env.step(act) for env, act in zip(envs, actions.cpu().numpy())]
            obs_list = [result[0] for result in step_results]
            obs = {k: np.stack([obs[k] for obs in obs_list], axis=0) for k in obs_list[0]}
            rew = np.stack([result[1] for result in step_results])
            done = np.stack([result[2] for result in step_results])

            done = np.logical_or(done, done_prev).astype(np.int32)
            rew = rew * (1 - done)
            rew_sum += rew

            if log_interval and t % log_interval == 0:
                elapsed = time.perf_counter() - start
                pbar.set_description(f"Epoch:{c} Step: {t}, FPS: {(num_batch * t / elapsed):.2f}, EnvDone: {done.astype(np.int32)}, Rew_Sum: {rew_sum}")

            # Don't continue if all environments are done.
            if np.all(done):
                logger.info(f"All envs done:{done}! Breaking..")
                break

        rew_sum_list.append(rew_sum)
    return np.concatenate(rew_sum_list)
