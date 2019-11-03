import torch

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.agents.dqn.epsilon_greedy import EpsilonGreedyAgentMixin
from rlpyt.distributions.epsilon_greedy import EpsilonGreedy
from rlpyt.models.utils import strip_ddp_state_dict
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple

AgentInfo = namedarraytuple("AgentInfo", "q")


class DqnAgent(EpsilonGreedyAgentMixin, BaseAgent):

    def __call__(self, observation, prev_action, prev_reward):
        """
        __call__使得一个class可以像一个method一样调用，即：假设agent为DqnAgent的一个对象，那么agent(observation, prev_action,
        prev_reward)就等同于调用agent.__call__(observation, prev_action, prev_reward)
        """
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        q = self.model(*model_inputs)  # torch.nn.Module子类的实例
        return q.cpu()  # 将模型的所有参数(parameter)和缓冲区(buffer)都转移到CPU内存

    def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
        """
        初始化agent。
        :param env_spaces:
        :param share_memory:
        :param global_B: 在BatchSpec中，表示独立的trajectory的数量，即environment实例的数量。这里的global_B可能是指所有env的总数
        :param env_ranks:
        :return:
        """
        super().initialize(env_spaces, share_memory, global_B=global_B, env_ranks=env_ranks)
        self.target_model = self.ModelCls(**self.env_model_kwargs, **self.model_kwargs)  # torch.nn.Module的子类
        self.target_model.load_state_dict(self.model.state_dict())  # 加载PyTorch模型，开始的时候target network和main network一致
        self.distribution = EpsilonGreedy(dim=env_spaces.action.n)  # 按ε-greedy方法来探索，n是action的维度
        if env_ranks is not None:
            self.make_vec_eps(global_B, env_ranks)

    def to_device(self, cuda_idx=None):
        """
        指定程序运行在什么设备上。
        TODO: 不明白为什么要指定两次。假设在在父类的to_device()方法里已经指定了一次GPU，紧接着又在后面又指定了一次CPU，这有何意义？

        :param cuda_idx: GPU编号
        """
        super().to_device(cuda_idx)
        self.target_model.to(self.device)  # self.device在初始化的时候已经写死了是CPU，因此这里指定在CPU上运行。

    def state_dict(self):
        """
        返回两个网络的state数据。例如网络的weight，bias等。
        :return: 一个dict
        """
        return dict(model=self.model.state_dict(), target=self.target_model.state_dict())

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        q = self.model(*model_inputs)  # 类型为torch.nn.Module
        q = q.cpu()  # 把模型的所有参数和buffer移到CPU，返回类型为torch.nn.Module
        action = self.distribution.sample(q)  # 选择一个action
        agent_info = AgentInfo(q=q)
        # action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def target(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        target_q = self.target_model(*model_inputs)
        return target_q.cpu()

    def update_target(self):
        self.target_model.load_state_dict(strip_ddp_state_dict(self.model.state_dict()))
