import numpy as np

from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import buffer_from_example
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args


class BaseCollector:
    """
    Class that steps through environments, possibly in worker process.
    在environment中步进(step)的类，有可能会在worker进程中运行(意思是也有可能会在主进程中运行)。这里的collector其实就是采样(或者说收集数据)
    用的，但在 rlpyt 里面不叫Sampler，而是叫Collector。在 rlpyt 中，Sampler是另一种概念，相比之下Sampler只会在主进程中运行。要搞清楚这个
    容易混淆的概念。
    这个类是 DecorrelatingStartCollector 类的基类，而 DecorrelatingStartCollector 又是很多其他Collector类的基类，因此这个类也就是
    很多Collector类的基类。
    """

    def __init__(
        self,
        rank,
        envs,
        samples_np,
        batch_T,
        TrajInfoCls,
        agent=None,  # Present or not, depending on collector class.
        sync=None,
        step_buffer_np=None,
        global_B=1,
        env_ranks=None,
    ):
        # 非常tricky的做法：把局部变量保存到实例的属性中，之后如果找不到self.xxx的定义就在这里面找
        save__init__args(locals())

    def start_envs(self):
        """Calls reset() on every env."""
        raise NotImplementedError

    def start_agent(self):
        if getattr(self, "agent", None) is not None:  # Not in GPU collectors.
            self.agent.collector_initialize(
                global_B=self.global_B,  # Args used e.g. for vector epsilon greedy.
                env_ranks=self.env_ranks,
            )
            self.agent.reset()
            self.agent.sample_mode(itr=0)

    def collect_batch(self, agent_inputs, traj_infos):
        """
        收集(即采样)一批数据。

        :param agent_inputs:
        :param traj_infos: TrajInfo类对象组成的一个list，包含trajectory的一些统计信息。
        :return: 由子类(例如 CpuResetCollector)实现。
        """
        raise NotImplementedError

    def reset_if_needed(self, agent_inputs):
        """Reset agent and or env as needed, if doing between batches."""
        pass


class BaseEvalCollector:
    """
    Does not record intermediate data.
    和 evaluation 相关的Collector类的基类。
    """

    def __init__(
        self,
        rank,
        envs,
        TrajInfoCls,
        traj_infos_queue,
        max_T,
        agent=None,
        sync=None,
        step_buffer_np=None,
    ):
        # 非常tricky的做法：把局部变量保存到实例的属性中，之后如果找不到self.xxx的定义就在这里面找
        save__init__args(locals())

    def collect_evaluation(self):
        raise NotImplementedError


class DecorrelatingStartCollector(BaseCollector):
    """
    该类是一个包含[去相关性]特性的collector。在强化学习中，数据之间的相关性(correlation)有时会导致训练出来的效果很差，因此有很多工作都会对
    数据做[去相关性](decorrelation)的操作。
    这个类是其他很多和 evaluation 无关的 Collector class的基类。
    """

    def start_envs(self, max_decorrelation_steps=0):
        """
        Calls reset() on every env and returns agent_inputs buffer.

        这个函数在Sampler类(例如SerialSampler)中的 initialize() 里会被调用，进行诸如收集(采样)第一批数据的工作。
        :param: max_decorrelation_steps: 最大[去相关性]的步数。
        :return 一个 namedarraytuple，包含3个元素(observation，action，reward)，每个元素又分别是一个list；以及trajectory的一些统计
        信息(TrajInfo类对象组成的一个list)。
        """
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]  # 每一个environment都对应一个TrajInfo对象
        observations = list()
        for env in self.envs:  # self.envs是一个environment的list，它是在sampler类(例如SerialSampler)里面实例化的
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, obs in enumerate(observations):
            observation[b] = obs  # numpy array or namedarraytuple
        prev_action = np.stack([env.action_space.null_value() for env in self.envs])
        prev_reward = np.zeros(len(self.envs), dtype="float32")
        if self.rank == 0:
            logger.log("Sampler decorrelating envs, max steps: "
                       f"{max_decorrelation_steps}")

        """
        在所有environment内，依次采样一批数据。按我的理解，这里的decorrelation逻辑是这样的：首先指定一个步数(例如100)，然后对每一个
        environment都走100步来采样，如果不到100步environment就走到头了也没关系，reset之后从头继续走，反正一共走够100步。所有environment
        里的数据混在一起返回，这样做确实起到了decorrelation的作用。
        """
        if max_decorrelation_steps != 0:
            for b, env in enumerate(self.envs):  # 遍历所有environment，b为从0开始的索引值，env为envs里面的每一个environment实例
                n_steps = 1 + int(np.random.rand() * max_decorrelation_steps)  # +1是防止结果为0导致逻辑不通
                for _ in range(n_steps):
                    """
                    关于env.action_space，可参考Env._action_space这个成员变量的值。这里的 env.action_space.sample()，对AtariEnv
                    就是计算IntBox.sample()，即在action space内随机选一个动作的index(并非实际动作)，这里没有直接得到action，而是得到
                    一个action space内的一个index，原因是：在env.step(a)里会根据index获取一个action。另外，这里之所以随机获取action
                    space内的一个index，是因为此时是在Collector类的start_envs()函数中，也就是说此时刚开始从environment里收集数据，
                    因此第一次收集的话，是不知道应该采取什么action的(不像后面已经得到一个network的时候可以根据前面的observation算出一个
                    action)，所以这里就随机选取一个index就好了。
                    """
                    a = env.action_space.sample()
                    o, r, d, info = env.step(a)  # 执行action，得到observation, reward, done(是否完成标志), info(一些统计信息)
                    traj_infos[b].step(o, a, r, d, None, info)  # 更新trajectory的一些统计信息
                    """
                    info是一个namedtuple，取出来的traj_done属性值，是一个bool，表明是否game over了(对Atari游戏来说)，如果没有game 
                    over，还要看是不是已经done了(比如游戏通关了)，所以getattr()的default value设置成了done标志。
                    """
                    if getattr(info, "traj_done", d):
                        o = env.reset()  # 重置environment，回到最初状态
                        traj_infos[b] = self.TrajInfoCls()  # TrajInfo类的对象
                    if d:  # done(比如游戏通关)
                        a = env.action_space.null_value()
                        r = 0
                observation[b] = o
                prev_action[b] = a
                prev_reward[b] = r
        # For action-server samplers. rlpyt有一种并行模式是Parallel-GPU，在这种模式下，会有一个action-server的概念(参考rlpyt论文)
        if hasattr(self, "step_buffer_np") and self.step_buffer_np is not None:
            self.step_buffer_np.observation[:] = observation
            self.step_buffer_np.action[:] = prev_action
            self.step_buffer_np.reward[:] = prev_reward
        return AgentInputs(observation, prev_action, prev_reward), traj_infos
