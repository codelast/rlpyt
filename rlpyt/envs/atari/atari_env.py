import os
from collections import namedtuple

import atari_py
import cv2
import numpy as np

from rlpyt.envs.base import Env, EnvStep
from rlpyt.samplers.collections import TrajInfo
from rlpyt.spaces.int_box import IntBox
from rlpyt.utils.quick_args import save__init__args

W, H = (80, 104)  # Crop two rows, then downsample by 2x (fast, clean image).

EnvInfo = namedtuple("EnvInfo", ["game_score", "traj_done"])


class AtariTrajInfo(TrajInfo):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.GameScore = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self.GameScore += getattr(env_info, "game_score", 0)


class AtariEnv(Env):

    def __init__(self,
                 game="pong",
                 frame_skip=4,  # Frames per step (>=1).
                 num_img_obs=4,  # Number of (past) frames in observation (>=1).
                 clip_reward=True,
                 episodic_lives=True,
                 max_start_noops=30,
                 repeat_action_probability=0.,
                 horizon=27000,  # 在游戏角色没有死的时候，step大于这个值也会被判定为game over
                 ):
        save__init__args(locals(), underscore=True)  # 非常tricky的做法：把局部变量保存到实例的属性中，之后如果找不到self.xxx的定义就在这里面找
        # ALE，即电玩学习环境(Arcade Learning Environment)，它提供了一个关于Atari 2600游戏的数百个游戏环境的接口
        game_path = atari_py.get_game_path(game)
        if not os.path.exists(game_path):
            raise IOError("You asked for game {} but path {} does not exist".format(game, game_path))
        self.ale = atari_py.ALEInterface()
        self.ale.setFloat(b'repeat_action_probability', repeat_action_probability)
        self.ale.loadROM(game_path)

        # Spaces，某些游戏可能不会对控制输入的所有组合都使用所有动作，例如pong游戏仅使用2个方向和射击键，这里要获取当前游戏使用的动作集
        self._action_set = self.ale.getMinimalActionSet()  # 最小的action set(一个numpy array)
        self._action_space = IntBox(low=0, high=len(self._action_set))
        obs_shape = (num_img_obs, H, W)  # H应该是指height，W应该是指width
        self._observation_space = IntBox(low=0, high=255, shape=obs_shape, dtype="uint8")
        self._max_frame = self.ale.getScreenGrayscale()
        self._raw_frame_1 = self._max_frame.copy()
        self._raw_frame_2 = self._max_frame.copy()
        self._obs = np.zeros(shape=obs_shape, dtype="uint8")  # 初始的observation

        # Settings
        self._has_fire = "FIRE" in self.get_action_meanings()
        self._has_up = "UP" in self.get_action_meanings()
        self._horizon = int(horizon)
        self.reset()

    def reset(self, hard=False):
        """
        复位游戏到初始状态。

        :param hard: 不知道干嘛用的。
        :return: 初始的observation，在这里是一个numpy array。
        """
        self.ale.reset_game()
        self._reset_obs()  # 重置observation
        self._life_reset()  # 重置生命值
        for _ in range(np.random.randint(0, self._max_start_noops + 1)):
            self.ale.act(0)
        self._update_obs()  # (don't bother to populate any frame history)
        self._step_counter = 0  # 用于统计走了多少个step的计数器
        return self.get_obs()  # 返回当前的observation，由于上面已经reset过observation了，因此这里返回的就是初始的observation

    def step(self, action):
        """
        在environment中向前走一步。 这个函数在Collector类(例如)的collect_batch()函数中会被调用。
        注意：policy network的前向传播过程不是在这里发生的，而是在agent类(例如DqnAgent)的step()函数里发生(由Collector类的
        collect_batch()函数调用)。environment里的step()，输入的action已经是policy network推断出来的action了，在这里做的工作主要是：
        计算该action带来的reward，判断trajectory是否结束，记录一些统计信息等。

        :param action: 一个标量，其值在 self._action_set 的index范围内。TODO: 确认是否正确？
        :return: 一个 EnvStep 对象，包含observation等数据。
        """
        a = self._action_set[action]  # 从action set(动作集)中取出一个具体的action
        game_score = np.array(0., dtype="float32")  # 游戏分数，其实就是一个标量值。这个函数里算出来的只是当前step的score而不是整个游戏过程的score
        # 可以设置每一个step走游戏的几帧，这里就连续地执行N-1(假设N为帧数)次action
        for _ in range(self._frame_skip - 1):
            game_score += self.ale.act(a)  # 执行一个action，得到一个score，累加到原来已经得到的分数上，这里累加也只是累加本step内的分数
        self._get_screen(1)
        game_score += self.ale.act(a)  # 上面skip的frame，还差一帧，这里补上执行一次action
        lost_life = self._check_life()  # Advances from lost_life state. 看看游戏角色是不是挂了
        if lost_life and self._episodic_lives:
            self._reset_obs()  # Internal reset.
        self._update_obs()
        # 奖励值。当设置了_clip_reward的时候使用-1，0，1作为reward，否则就使用真实的游戏分数作为reward
        reward = np.sign(game_score) if self._clip_reward else game_score
        game_over = self.ale.game_over() or self._step_counter >= self.horizon  # 判断游戏是不是结束了，当horizon达到阈值时也结束
        done = game_over or (self._episodic_lives and lost_life)  # bool类型
        info = EnvInfo(game_score=game_score, traj_done=game_over)  # 当前environment的一些信息，比如游戏分数等
        self._step_counter += 1  # 用于统计走了多少个step的计数器
        return EnvStep(self.get_obs(), reward, done, info)

    def render(self, wait=10, show_full_obs=False):
        """
        图形渲染(在窗口中展示游戏画面)。
        :param wait: 窗口显示的时间(毫秒)，为0或不传都是无限等待。
        :param show_full_obs: TODO
        """
        img = self.get_obs()
        if show_full_obs:
            shape = img.shape
            img = img.reshape(shape[0] * shape[1], shape[2])
        else:
            img = img[-1]
        cv2.imshow(self._game, img)
        cv2.waitKey(wait)

    def get_obs(self):
        return self._obs.copy()

    ###########################################################################
    # Helpers

    def _get_screen(self, frame=1):
        frame = self._raw_frame_1 if frame == 1 else self._raw_frame_2
        self.ale.getScreenGrayscale(frame)

    def _update_obs(self):
        """Max of last two frames; crop two rows; downsample by 2x."""
        self._get_screen(2)
        np.maximum(self._raw_frame_1, self._raw_frame_2, self._max_frame)
        img = cv2.resize(self._max_frame[1:-1], (W, H), cv2.INTER_NEAREST)
        # NOTE: order OLDEST to NEWEST should match use in frame-wise buffer.
        self._obs = np.concatenate([self._obs[1:], img[np.newaxis]])

    def _reset_obs(self):
        self._obs[:] = 0
        self._max_frame[:] = 0
        self._raw_frame_1[:] = 0
        self._raw_frame_2[:] = 0

    def _check_life(self):
        lives = self.ale.lives()
        lost_life = (lives < self._lives) and (lives > 0)
        if lost_life:
            self._life_reset()
        return lost_life

    def _life_reset(self):
        self.ale.act(0)  # (advance from lost life state)
        if self._has_fire:
            # TODO: for sticky actions, make sure fire is actually pressed
            self.ale.act(1)  # (e.g. needed in Breakout, not sure what others)
        if self._has_up:
            self.ale.act(2)  # (not sure if this is necessary, saw it somewhere)
        self._lives = self.ale.lives()

    ###########################################################################
    # Properties

    @property
    def game(self):
        return self._game

    @property
    def frame_skip(self):
        return self._frame_skip

    @property
    def num_img_obs(self):
        return self._num_img_obs

    @property
    def clip_reward(self):
        return self._clip_reward

    @property
    def max_start_noops(self):
        return self._max_start_noops

    @property
    def episodic_lives(self):
        return self._episodic_lives

    @property
    def repeat_action_probability(self):
        return self._repeat_action_probability

    @property
    def horizon(self):
        return self._horizon

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]


ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}

ACTION_INDEX = {v: k for k, v in ACTION_MEANING.items()}
