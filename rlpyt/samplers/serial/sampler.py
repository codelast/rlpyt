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
        initialize()方法会在runner类(例如MinibatchRlBase)中的startup()方法里被调用
        """
        B = self.batch_spec.B  # 独立的trajectory的数量，即environment实例的数量。此值>=1
        envs = [self.EnvCls(**self.env_kwargs) for _ in range(B)]  # 初始化每一个environment，生成一个environment的list
        global_B = B * world_size
        env_ranks = list(range(rank * B, (rank + 1) * B))
        agent.initialize(envs[0].spaces, share_memory=False,
                         global_B=global_B, env_ranks=env_ranks)
        samples_pyt, samples_np, examples = build_samples_buffer(agent, envs[0],
                                                                 self.batch_spec, bootstrap_value, agent_shared=False,
                                                                 env_shared=False, subprocess=False)
        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing at init.
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
                max_T=self.eval_max_steps // self.eval_n_envs,
                max_trajectories=self.eval_max_trajectories,
            )

        # 返回(observation, prev_action, prev_reward)以及trajectory的信息，所谓trajectory信息是指reward，非零reward的数量等统计值
        agent_inputs, traj_infos = collector.start_envs(self.max_decorrelation_steps)
        collector.start_agent()

        self.agent = agent  # self.agent在本类中没有用，但尚未确定在外面使用SerialSampler的时候会不会使用
        self.samples_pyt = samples_pyt  # PyTorch格式(即底层是torch.Tensor)的samples
        self.samples_np = samples_np  # numpy格式(即底层是numpy array)的samples
        self.collector = collector  # sample收集器
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        logger.log("Serial Sampler initialized.")
        return examples

    def obtain_samples(self, itr):
        """
        采样一批数据。
        :param itr: 整数，表示第几次迭代
        :return:
        """
        # self.samples_np[:] = 0  # Unnecessary and may take time.
        agent_inputs, traj_infos, completed_infos = self.collector.collect_batch(
            self.agent_inputs, self.traj_infos, itr)  # 采样第itr次
        self.collector.reset_if_needed(agent_inputs)
        self.agent_inputs = agent_inputs
        self.traj_infos = traj_infos
        return self.samples_pyt, completed_infos

    def evaluate_agent(self, itr):
        return self.eval_collector.collect_evaluation(itr)
