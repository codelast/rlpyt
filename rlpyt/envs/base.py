"""
environment的基类。如果要开发一个非Atari游戏的强化学习应用，需要自己写一个environment class继承自这个类。
"""
from collections import namedtuple

EnvStep = namedtuple("EnvStep", ["observation", "reward", "done", "env_info"])
EnvInfo = namedtuple("EnvInfo", [])  # Define in env file.
EnvSpaces = namedtuple("EnvSpaces", ["observation", "action"])


class Env:

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a namedtuple containing other diagnostic information from the previous action
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        raise NotImplementedError

    @property
    def action_space(self):
        """
        动作空间，在子类(例如AtariEnv)中会赋值。
        :return: 一个IntBox类型的对象。
        """
        return self._action_space

    @property
    def observation_space(self):
        """
        在子类(例如AtariEnv)中会赋值，一个IntBox类型的对象。
        :return: 一个IntBox类型的对象。
        """
        return self._observation_space

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self.observation_space,  # 调用本类的observation_space()方法
            action=self.action_space,  # 调用本类的action_space()方法
        )

    @property
    def horizon(self):
        """Horizon of the environment, if it has one."""
        raise NotImplementedError

    def close(self):
        """Clean up operation."""
        pass
