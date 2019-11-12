from collections import namedtuple

from rlpyt.utils.collections import namedarraytuple, AttrDict

Samples = namedarraytuple("Samples", ["agent", "env"])

AgentSamples = namedarraytuple("AgentSamples",
                               ["action", "prev_action", "agent_info"])
AgentSamplesBsv = namedarraytuple("AgentSamplesBsv",
                                  ["action", "prev_action", "agent_info", "bootstrap_value"])
EnvSamples = namedarraytuple("EnvSamples",
                             ["observation", "reward", "prev_reward", "done", "env_info"])


class BatchSpec(namedtuple("BatchSpec", "T B")):
    """
    T: int  Number of time steps, >=1.
    B: int  Number of separate trajectory segments (i.e. # env instances), >=1.

    T表示时间步的数量。所谓时间步是指agent与一个environment交互时，会按时间先后顺序不断地步进到下一个state，走一步即一个step。此值>=1。
    B表示独立的trajectory的数量，即environment实例的数量。此值>=1。
    Python允许在定义class的时候，定义一个特殊的__slots__变量来限制该类实例能添加的属性。这里设置为空的tuple表示不允许为该类动态添加绑定属性。
    """
    __slots__ = ()

    @property
    def size(self):
        """
        所有environment实例上的所有时间步的数量总和。
        """
        return self.T * self.B


class TrajInfo(AttrDict):
    """
    Because it inits as an AttrDict, this has the methods of a dictionary,
    e.g. the attributes can be iterated through by traj_info.items()
    Intent: all attributes not starting with underscore "_" will be logged.
    (Can subclass for more fields.)
    Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase.
    """

    _discount = 1  # Leading underscore, but also class attr not in self.__dict__.

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # (for AttrDict behavior)
        self.Length = 0
        self.Return = 0
        self.NonzeroRewards = 0
        self.DiscountedReturn = 0
        self._cur_discount = 1

    def step(self, observation, action, reward, done, agent_info, env_info):
        self.Length += 1
        self.Return += reward
        self.NonzeroRewards += reward != 0
        self.DiscountedReturn += self._cur_discount * reward
        self._cur_discount *= self._discount

    def terminate(self, observation):
        return self
