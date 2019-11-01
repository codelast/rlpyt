from rlpyt.samplers.collections import BatchSpec, TrajInfo
from rlpyt.utils.quick_args import save__init__args


class BaseSampler:
    """
    Class which interfaces with the Runner, in master process only.
    和Runner对接的类，只在主进程中运行。
    """

    alternating = False

    def __init__(
        self,
        EnvCls,
        env_kwargs,
        batch_T,
        batch_B,
        CollectorCls,
        max_decorrelation_steps=100,
        TrajInfoCls=TrajInfo,
        eval_n_envs=0,  # 0 for no eval setup.
        eval_CollectorCls=None,  # Must supply if doing eval.
        eval_env_kwargs=None,
        eval_max_steps=None,  # int if using evaluation.
        eval_max_trajectories=None,  # Optional earlier cutoff.
    ):
        eval_max_steps = None if eval_max_steps is None else int(eval_max_steps)
        eval_max_trajectories = (None if eval_max_trajectories is None else int(eval_max_trajectories))
        save__init__args(locals())  # 非常tricky的做法：把局部变量保存到实例的属性中，之后如果找不到self.xxx的定义就在这里面找
        self.batch_spec = BatchSpec(batch_T, batch_B)  # 保存batch的信息，包含：时间步的数量，以及environment实例的数量
        # 在SerialSampler初始化时，CollectorCls=CpuResetCollector，且 CpuResetCollector 的 mid_batch_reset 为True
        self.mid_batch_reset = CollectorCls.mid_batch_reset

    def initialize(self, *args, **kwargs):
        raise NotImplementedError

    def obtain_samples(self, itr):
        raise NotImplementedError  # type: Samples

    def evaluate_agent(self, itr):
        raise NotImplementedError

    def shutdown(self):
        pass

    @property
    def batch_size(self):
        """
        所有environment实例上的所有时间步的数量总和。
        """
        return self.batch_spec.size  # For logging at least.
