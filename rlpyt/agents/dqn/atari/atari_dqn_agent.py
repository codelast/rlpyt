"""
这个class是和具体的environment(例如Atari)相关的。其父类 DqnAgent 则抽象到了和具体的environment无关。
为什么要有AtariMixin这个父类？请看 AtariMixin 类的注释。
"""
from rlpyt.agents.dqn.atari.mixin import AtariMixin
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.models.dqn.atari_dqn_model import AtariDqnModel


class AtariDqnAgent(AtariMixin, DqnAgent):

    def __init__(self, ModelCls=AtariDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)
