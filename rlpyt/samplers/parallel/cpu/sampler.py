from rlpyt.samplers.parallel.base import ParallelSamplerBase
from rlpyt.samplers.parallel.cpu.collectors import (CpuResetCollector,
                                                    CpuEvalCollector)


class CpuSampler(ParallelSamplerBase):

    def __init__(self, *args, CollectorCls=CpuResetCollector, eval_CollectorCls=CpuEvalCollector, **kwargs):
        # e.g. or use CpuWaitResetCollector, etc...
        super().__init__(*args, CollectorCls=CollectorCls,
                         eval_CollectorCls=eval_CollectorCls, **kwargs)

    def obtain_samples(self, itr):
        """
        采样一批数据。

        :param itr: 第几次迭代。
        :return: TODO
        """
        self.agent.sync_shared_memory()  # New weights in workers, if needed. 在并行模式下，采样之前需要先同步共享模型。
        return super().obtain_samples(itr)

    def evaluate_agent(self, itr):
        self.agent.sync_shared_memory()
        return super().evaluate_agent(itr)
