"""
这个class已经抽象到和具体的environment(例如Atari)无关，而它的子类还是有可能和具体的environment相关的。
"""
import torch

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.agents.dqn.epsilon_greedy import EpsilonGreedyAgentMixin
from rlpyt.distributions.epsilon_greedy import EpsilonGreedy
from rlpyt.models.utils import strip_ddp_state_dict
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.models.utils import update_state_dict


AgentInfo = namedarraytuple("AgentInfo", "q")


class DqnAgent(EpsilonGreedyAgentMixin, BaseAgent):

    def __call__(self, observation, prev_action, prev_reward):
        """
        __call__使得一个class可以像一个method一样调用，即：假设agent为DqnAgent的一个对象，那么agent(observation, prev_action,
        prev_reward)就等同于调用agent.__call__(observation, prev_action, prev_reward)
        """
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        q = self.model(*model_inputs)  # torch.nn.Module子类的实例，使用torch.nn.Module里定义的__call__调用，相当于计算模型输出(一个Tensor)
        return q.cpu()  # 将tensor移动到CPU(内存)

    def initialize(self, env_spaces, share_memory=False, global_B=1, env_ranks=None):
        """
        初始化agent。这个函数在Sampler类(例如SerialSampler)中的 initialize() 里会被调用。

        :param env_spaces: 参考 Env.spaces()，类型为 EnvSpaces 这样一个 namedtuple，包含observation space 和 action space两个属性。
        :param share_memory: 为 True 时使得模型参数可以在多进程间共享，为 False 时不共享。
        :param global_B: 在BatchSpec中，表示独立的trajectory的数量，即environment实例的数量。这里的global_B可能是指所有env的总数
        :param env_ranks: TODO:
        :return: TODO:
        """
        super().initialize(env_spaces, share_memory, global_B=global_B, env_ranks=env_ranks)
        self.target_model = self.ModelCls(**self.env_model_kwargs, **self.model_kwargs)  # torch.nn.Module的子类
        self.target_model.load_state_dict(self.model.state_dict())  # 加载PyTorch模型，开始的时候target network和main network一致
        self.distribution = EpsilonGreedy(dim=env_spaces.action.n)  # 按ε-greedy方法来探索，n是action的维度
        if env_ranks is not None:
            self.make_vec_eps(global_B, env_ranks)

    def to_device(self, cuda_idx=None):
        """
        指定把模型数据(parameter和buffer)放在什么设备上(CPU/GPU)。
        父类是指定self.model的数据放在哪个GPU上，在本子类中是指定self.target_model的数据放在哪。

        :param cuda_idx: GPU编号
        """
        super().to_device(cuda_idx)
        self.target_model.to(self.device)  # self.device在初始化的时候已经写死了是CPU，因此这里指定在CPU上运行

    def state_dict(self):
        """
        返回两个网络的state数据。例如网络的weight，bias等。

        :return: 一个dict
        """
        return dict(model=self.model.state_dict(), target=self.target_model.state_dict())

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """
        在environment中走一步。这个函数在Collector类的collect_batch()函数中会被调用。
        这里会发生policy network的前向传播过程(比较耗计算资源的操作)，即根据输入(例如observation)计算下一步要采
        取的action。

        :param observation: 其义自明。
        :param prev_action: 前一个action。
        :param prev_reward: 之前累积的reward。
        :return: 要采取的action(类型为torch.Tensor)，以及agent的信息(例如Q值)
        """
        prev_action = self.distribution.to_onehot(prev_action)  # 返回类型为 torch.Tensor
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)  # 策略网络的输入(torch.Tensor)
        q = self.model(*model_inputs)  # self.model是torch.nn.Module的子类对象，这里是输入特征计算网络的输出，因此会发生NN的forward过程
        q = q.cpu()  # 把tensor移到CPU(内存)，返回torch.Tensor
        action = self.distribution.sample(q)  # 选择一个action(torch.Tensor)
        agent_info = AgentInfo(q=q)
        # action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def target(self, observation, prev_action, prev_reward):
        """
        计算Q值。

        :param observation: 如其名。
        :param prev_action: 前一个action。
        :param prev_reward: 前一个reward。
        :return: CPU(内存)里的Q值对应的Tensor。
        """
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        # 计算Q值，self.target_model是一个torch.nn.Module子类的实例，使用torch.nn.Module里定义的__call__调用，相当于计算模型输出(一个Tensor)
        target_q = self.target_model(*model_inputs)
        return target_q.cpu()  # 将tensor移动到CPU(内存)

    def update_target(self, tau=1):
        """
        更新target network，即把main network的参数拷贝到target network上。
        """
        update_state_dict(self.target_model, self.model.state_dict(), tau)
