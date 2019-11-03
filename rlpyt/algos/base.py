"""
算法类的基类。例如 DQN，DDPG等class都是继承自这个类。
"""

class RlAlgorithm:
    opt_info_fields = ()  # 在子类中会保存和具体算法(例如DQN)相关的一些字段名，例如对DQN来说是"loss", "gradNorm", "tdAbsErr"等
    bootstrap_value = False  # TODO:
    update_counter = 0  # 神经网络的参数更新次数

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples, world_size=1, rank=0):
        raise NotImplementedError

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset, examples, world_size=1):
        """Called instead of initialize() in async runner.
        Should return async replay_buffer using shared memory."""
        raise NotImplementedError

    def optim_initialize(self, rank=0):
        """Called in async runner, and possibly self.initialize()."""
        raise NotImplementedError

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        raise NotImplementedError

    def optim_state_dict(self):
        """If carrying multiple optimizers, overwrite to return dict state_dicts."""
        return self.optimizer.state_dict()

    def load_optim_state_dict(self, state_dict):
        """
        加载优化器对象(Optimizer)的state_dict，它包含了优化器的状态以及被使用的超参数，例如learning rate，momentum等。
        """
        self.optimizer.load_state_dict(state_dict)

    @property
    def batch_size(self):
        """
        _batch_size虽然在父类里没有定义，但是在子类里定义了。
        """
        return self._batch_size  # For logging at least.
