# from https://github.com/facebookresearch/moolib/blob/06e7a3e80c9f52729b4a6159f3fb4fc78986c98e/examples/atari/atari_preprocessing.py

# coding=utf-8
# Copyright 2019 The SEED Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A class implementing minimal Atari 2600 preprocessing.

Adapted from Dopamine.
"""


import gym
import numpy as np
from gym.spaces.box import Box

from loguru import logger
# logger.remove()
# logger.add(sys.stdout, level="INFO")
# logger.add(sys.stdout, level="SUCCESS")
# logger.add(sys.stdout, level="WARNING")

# Convert np to rgb
# import time
# from PIL import Image
# from IPython import display
# import matplotlib.pyplot as plt
import cv2


class AtariPreprocessing(gym.Wrapper):
    _rgb_img = None
    _fig = None
    _axarr = None


    """A class implementing image preprocessing for Atari 2600 agents.

    Specifically, this provides the following subset from the JAIR paper
    (Bellemare et al., 2013) and Nature DQN paper (Mnih et al., 2015):

      * Frame skipping (defaults to 4).
      * Terminal signal when a life is lost (off by default).
      * Grayscale and max-pooling of the last two frames.
      * Downsample the screen to a square image (defaults to 84x84).

    More generally, this class follows the preprocessing guidelines set down in
    Machado et al. (2018), "Revisiting the Arcade Learning Environment:
    Evaluation Protocols and Open Problems for General Agents".

    It also provides random starting no-ops, which are used in the Rainbow, Apex
    and R2D2 papers.
    """

    def __init__(self, env, env_id, frame_skip=4, terminal_on_life_loss=False, screen_size=84, max_random_noops=0):
        """Constructor for an Atari 2600 preprocessor.

        Args:
          env: Gym environment whose observations are preprocessed.
          frame_skip: int, the frequency at which the agent experiences the game.
          terminal_on_life_loss: bool, If True, the step() method returns
            is_terminal=True whenever a life is lost. See Mnih et al. 2015.
          screen_size: int, size of a resized Atari 2600 frame.
          max_random_noops: int, maximum number of no-ops to apply at the beginning
            of each episode to reduce determinism. These no-ops are applied at a
            low-level, before frame skipping.

        Raises:
          ValueError: if frame_skip or screen_size are not strictly positive.
        """
        super().__init__(env)
        if frame_skip <= 0:
            raise ValueError("Frame skip should be strictly positive, got {}".format(frame_skip))
        if screen_size <= 0:
            raise ValueError("Target screen size should be strictly positive, got {}".format(screen_size))

        self.terminal_on_life_loss = terminal_on_life_loss
        self.frame_skip = frame_skip
        self.screen_size = screen_size
        self.max_random_noops = max_random_noops

        obs_dims = self.env.observation_space
        # Stores temporary observations used for pooling over two successive
        # frames.
        self.screen_buffer = [
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
            np.empty((obs_dims.shape[0], obs_dims.shape[1]), dtype=np.uint8),
        ]

        self.game_over = False
        self.lives = 0  # Will need to be set by reset().

        # Mod by Tim:
        self.env_id = env_id
        logger.info(f"env_id:{env_id}")

        # Mod by Tim: To render
        # plt.figure()
        # self._fig, self._axarr = plt.subplots(4,1) 


    @property
    def observation_space(self):
        # Return the observation space adjusted to match the shape of the processed
        # observations.
        return Box(low=0, high=255, shape=(self.screen_size, self.screen_size), dtype=np.uint8)

    def apply_random_noops(self):
        """Steps self.env with random no-ops."""
        if self.max_random_noops <= 0:
            return
        # Other no-ops implementations actually always do at least 1 no-op. We
        # follow them.
        no_ops = self.env.np_random.integers(1, self.max_random_noops + 1)
        for _ in range(no_ops):
            _, _, game_over, _ = self.env.step(0)
            if game_over:
                self.env.reset()

    def reset(self, **kwargs):
        """Resets the environment.

        Returns:
          observation: numpy array, the initial observation emitted by the
            environment.
        """
        self.env.reset(**kwargs)
        self.apply_random_noops()

        self.lives = self.env.ale.lives()
        self._fetch_grayscale_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        return self._pool_and_resize()
    
    def render(self, mode):
        """Renders the current screen, before preprocessing.

        This calls the Gym API's render() method.

        Args:
          mode: Mode argument for the environment's render() method.
            Valid values (str) are:
              'rgb_array': returns the raw ALE image.
              'human': renders to display via the Gym renderer.

        Returns:
          if mode='rgb_array': numpy array, the most recent screen.
          if mode='human': bool, whether the rendering was successful.
        """
        # Mod by Tim:
        return self.env.render(mode)


    def step(self, action):
        """Applies the given action in the environment.

        Remarks:

          * If a terminal state (from life loss or episode end) is reached, this may
            execute fewer than self.frame_skip steps in the environment.
          * Furthermore, in this case the returned observation may not contain valid
            image data and should be ignored.

        Args:
          action: The action to be executed.

        Returns:
          observation: numpy array, the observation following the action.
          reward: float, the reward following the action.
          is_terminal: bool, whether the environment has reached a terminal state.
            This is true when a life is lost and terminal_on_life_loss, or when the
            episode is over.
          info: Gym API's info data structure.
        """
        accumulated_reward = 0.0

        for time_step in range(self.frame_skip):
            # We bypass the Gym observation altogether and directly fetch the
            # grayscale image from the ALE. This is a little faster.
            _, reward, game_over, info = self.env.step(action)
            accumulated_reward += reward

            if self.terminal_on_life_loss:
                new_lives = self.env.ale.lives()
                is_terminal = game_over or new_lives < self.lives
                self.lives = new_lives
            else:
                is_terminal = game_over

            if is_terminal:
                break
            # We max-pool over the last two frames, in grayscale.
            elif time_step >= self.frame_skip - 2:
                t = time_step - (self.frame_skip - 2)
                self._fetch_grayscale_observation(self.screen_buffer[t])

        # Mod by Tim: To render
        # Pool the last two observations.
        observation = self._pool_and_resize()
        # logger.info(f"len observation:{len(observation)}") # size 84


        self.game_over = game_over
        return observation, accumulated_reward, is_terminal, info

    def _fetch_grayscale_observation(self, output):
        """Returns the current observation in grayscale.

        The returned observation is stored in 'output'.

        Args:
          output: numpy array, screen buffer to hold the returned observation.

        Returns:
          observation: numpy array, the current observation in grayscale.
        """
        self.env.ale.getScreenGrayscale(output)

        # --------------------------------------------------
        # Mod by Tim: To render RGB
        if self.env_id == 0:
          np_img = self.env.render(mode='rgb_array')
          # logger.info(f"  len(img):{len(np_img)} type(img):{type(np_img)}") # size 210
          # logger.info(f"  type(rgb_img):{type(self._rgb_img)}") # size 210
          #---------------------------------
          cv2.imshow('image', np_img)
          # logger.info(f"_fetch_grayscale_observation() - np_img.shape:{np_img.shape} ") # np_img.shape:(210, 160, 3)
          # waitKey() n milliseconds. 
          # If 0 is passed in the argument it waits till any key is pressed. 
          cv2.waitKey(1) 
        # --------------------------------------------------

        return output

    def _pool_and_resize(self):
        """Transforms two frames into a Nature DQN observation.

        For efficiency, the transformation is done in-place in self.screen_buffer.

        Returns:
          transformed_screen: numpy array, pooled, resized screen.
        """
        # Pool if there are enough screens to do so.
        if self.frame_skip > 1:
            np.maximum(self.screen_buffer[0], self.screen_buffer[1], out=self.screen_buffer[0])

        transformed_image = cv2.resize(
            self.screen_buffer[0], (self.screen_size, self.screen_size), interpolation=cv2.INTER_LINEAR
        )

        np_transformed_image = np.asarray(transformed_image, dtype=np.uint8)
        # --------------------------------------------------
        # Mod by Tim: To render grayscale (FPS 50% faster than RGB)
        # if self.env_id == 0:
        #   # logger.info(f"  len(img):{len(np_transformed_image)} type(img):{type(np_transformed_image)}") # size 84, type numpy.ndarray
        #   #---------------------------------
        #   cv2.imshow('image', np_transformed_image)
        #   cv2.waitKey(1) 
        # --------------------------------------------------


        # return np.asarray(transformed_image, dtype=np.uint8)
        return np_transformed_image
