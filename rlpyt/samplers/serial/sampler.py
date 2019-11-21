"""
这个类抽象到了和具体的environment、agent无关。
"""
from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.buffer import build_samples_buffer
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.samplers.serial.collectors import SerialEvalCollector
from rlpyt.utils.logging import logger


class SerialSampler(BaseSampler):
    """Uses same functionality as ParallelSampler but does not fork worker
    processes; can be easier for debugging (e.g. breakpoint() in master).  Use
    with collectors which sample actions themselves (e.g. under cpu
    category)."""

    def __init__(self, *args, CollectorCls=CpuResetCollector, eval_CollectorCls=SerialEvalCollector, **kwargs):
        # 调用父类BaseSampler的__init__()方法初始化
        super().__init__(*args, CollectorCls=CollectorCls, eval_CollectorCls=eval_CollectorCls, **kwargs)

    def initialize(
        self,
        agent,
        affinity=None,
        seed=None,
        bootstrap_value=False,
        traj_info_kwargs=None,
        rank=0,
        world_size=1,
    ):
        """
        initialize()方法会在runner类(例如MinibatchRlBase)的startup()方法里被调用。
        """
        B = self.batch_spec.B  # 独立的trajectory的数量，即environment实例的数量。此值>=1
        envs = [self.EnvCls(**self.env_kwargs) for _ in range(B)]  # 初始化每一个environment实例，生成一个list
        global_B = B * world_size  # 这里的概念可能是指复制出来的N个environment实例的数量
        env_ranks = list(range(rank * B, (rank + 1) * B))
        """
        由于每一个environment的spaces都是一样的(这里是指action space 和 observation space)，因此只需要拿一个environment的实例出来，
        即 envs[0]，再取其spaces，也就代表了每一个environment的spaces。这里的 .spaces 是被作为一个属性来使用，但实际上它是一个函数，在
        class Env里面，用@property来修饰使之可以用属性的方式调用。总结：envs[0].spaces 得到的是一个namedtuple(EnvSpaces)，其包含两个
        属性：observation space 和 action space。 
        """
        agent.initialize(envs[0].spaces, share_memory=False,
                         global_B=global_B, env_ranks=env_ranks)
        samples_pyt, samples_np, examples = build_samples_buffer(agent, envs[0],
                                                                 self.batch_spec, bootstrap_value, agent_shared=False,
                                                                 env_shared=False, subprocess=False)
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.
        """
        对env_ranks，collector.start_agent()和上面的agent.initialize()都会调用到 EpsilonGreedyAgentMixin.make_vec_eps()，这
        其实是重复执行了一部分逻辑，所以作者会在collector的构造函数这里加上这句注释：Might get applied redundantly to agent.
        """
        collector = self.CollectorCls(
            rank=0,
            envs=envs,
            samples_np=samples_np,
            batch_T=self.batch_spec.T,
            TrajInfoCls=self.TrajInfoCls,
            agent=agent,
            global_B=global_B,
            env_ranks=env_ranks,  # Might get applied redundantly to agent.
        )
        if self.eval_n_envs > 0:  # May do evaluation.
            eval_envs = [self.EnvCls(**self.eval_env_kwargs) for _ in range(self.eval_n_envs)]
            eval_CollectorCls = self.eval_CollectorCls or SerialEvalCollector
            self.eval_collector = eval_CollectorCls(
                envs=eval_envs,
                agent=agent,
                TrajInfoCls=self.TrajInfoCls,
                max_T=self.eval_max_steps // self.eval_n_envs,  # 先做除法再向下取整。计算时间步的数量，这里有点计算"平均"时间步的意思
                max_trajectories=self.eval_max_trajectories,
            )

        """
        收集(即采样)第一批数据(observation，action，reward等)，以及所有environment对应的所有trajectory的信息(含reward等统计值)。这里
        之所以要收集第一批数据并保存到类成员变量 self.agent_inputs 中，是因为现在是Sampler初始化过程，当开始连续收集数据的时候，会在第一批
        数据的基础上step下去，因此就把获取第一批数据的工作放到了这里。
        """
        agent_inputs, traj_infos = collector.start_envs(self.max_decorrelation_steps)
        collector.start_agent()

        self.agent = agent  # self.agent在本类中没有用，但尚未确定在外面使用SerialSampler的时候会不会使用
        self.samples_pyt = samples_pyt  # PyTorch数据格式(即底层是torch.Tensor)的samples
        self.samples_np = samples_np  # numpy数据格式(即底层是numpy array)的samples
        self.collector = collector  # sample收集器
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos  # 一个list
        logger.log("Serial Sampler initialized.")
        return examples

    def obtain_samples(self, itr):
        """
        采样一批数据。这个函数会在Runner类的子类(例如MinibatchRlEval)中被调用。

        :param itr: 第几次迭代
        :return: TODO
        """
        # self.samples_np[:] = 0  # Unnecessary and may take time.
        agent_inputs, traj_infos, completed_infos = self.collector.collect_batch(
            self.agent_inputs, self.traj_infos, itr)  # 采样第itr次
        self.collector.reset_if_needed(agent_inputs)
        # 用每一次collect_batch()得到的新数据替换掉旧数据，下一次collect_batch()的时候就是在新数据的基础上进行的step
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        return self.samples_pyt, completed_infos

    def evaluate_agent(self, itr):
        """
        做evaluation。

        :param itr: 第几次迭代。
        :return: TODO
        """
        return self.eval_collector.collect_evaluation(itr)
