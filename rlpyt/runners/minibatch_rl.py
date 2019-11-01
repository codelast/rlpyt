import math
import time
from collections import deque

import psutil
import torch

from rlpyt.runners.base import BaseRunner
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed, make_seed


class MinibatchRlBase(BaseRunner):
    _eval = False

    def __init__(
        self,
        algo,
        agent,
        sampler,
        n_steps,
        seed=None,
        affinity=None,
        log_interval_steps=1e5,
    ):
        n_steps = int(n_steps)
        log_interval_steps = int(log_interval_steps)
        affinity = dict() if affinity is None else affinity  # CPU亲和性定义
        # 非常tricky的做法：把局部变量保存到实例的属性中。在后面的大量代码中，都会看到很多貌似没有定义过的self.xxx变量，它们就是在这里被定义的
        save__init__args(locals())

    def startup(self):
        p = psutil.Process()

        # 设置CPU亲和性，不支持MacOS
        try:
            if self.affinity.get("master_cpus", None) is not None and self.affinity.get("set_affinity", True):
                p.cpu_affinity(self.affinity["master_cpus"])
            cpu_affin = p.cpu_affinity()
        except AttributeError:
            cpu_affin = "UNAVAILABLE MacOS"
        logger.log(f"Runner {getattr(self, 'rank', '')} master CPU affinity: {cpu_affin}.")

        # 设置线程数
        if self.affinity.get("master_torch_threads", None) is not None:
            torch.set_num_threads(self.affinity["master_torch_threads"])  # 设置用于并行化CPU操作的OpenMP线程数
        logger.log(f"Runner {getattr(self, 'rank', '')} master Torch threads: {torch.get_num_threads()}.")

        # 设置随机数种子
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)

        self.rank = rank = getattr(self, "rank", 0)  # rank是外部传进来的值，如果没有传则默认为1
        self.world_size = world_size = getattr(self, "world_size", 1)  # world_size是外部传进来的值，如果没有传则默认为1

        # 初始化Sampler实例，这里的变量名examples起得不好，可能会让人误解
        examples = self.sampler.initialize(
            agent=self.agent,  # Agent gets initialized in sampler. agent会在Sampler中被初始化，所以要传进去
            affinity=self.affinity,  # CPU亲和性
            seed=self.seed + 1,  # 随机种子
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),  # 此方法里只设置折扣系数
            rank=rank,
            world_size=world_size,
        )
        self.itr_batch_size = self.sampler.batch_spec.size * world_size  # 所有迭代的时间步数
        n_itr = self.get_n_itr()
        self.agent.to_device(self.affinity.get("cuda_idx", None))  # self.affinity 是一个 dict
        if world_size > 1:
            self.agent.data_parallel()

        # 初始化算法(Algorithm)实例
        self.algo.initialize(
            agent=self.agent,
            n_itr=n_itr,
            batch_spec=self.sampler.batch_spec,
            mid_batch_reset=self.sampler.mid_batch_reset,
            examples=examples,
            world_size=world_size,
            rank=rank,
        )

        # 初始化日志参数
        self.initialize_logging()
        return n_itr

    def get_traj_info_kwargs(self):
        """
        函数名起得不好，让人迷惑。
        本函数用于获取算法(Algorithm)实例里设置的某些参数，例如折扣因子(discount factor)。

        :return: 一个dict，里面只含有折扣因子(discount factor)。
        """
        return dict(discount=getattr(self.algo, "discount", 1))  # discount factor默认值为1，即没有折扣

    def get_n_itr(self):
        """
        获取迭代次数。
        """

        # 周期性记录日志的次数。max()保证了记录日志至少1次
        log_interval_itrs = max(self.log_interval_steps // self.itr_batch_size, 1)  # // 即先做除法(/)，然后向下取整(floor)
        n_itr = math.ceil(self.n_steps / self.log_interval_steps) * log_interval_itrs
        self.log_interval_itrs = log_interval_itrs  # 周期性记录日志的次数
        self.n_itr = n_itr
        logger.log(f"Running {n_itr} iterations of minibatch RL.")
        return n_itr

    def initialize_logging(self):
        self._opt_infos = {k: list() for k in self.algo.opt_info_fields}
        self._start_time = self._last_time = time.time()
        self._cum_time = 0.
        self._cum_completed_trajs = 0
        self._last_update_counter = 0

    def shutdown(self):
        logger.log("Training complete.")
        self.pbar.stop()
        self.sampler.shutdown()

    def get_itr_snapshot(self, itr):
        return dict(
            itr=itr,
            cum_steps=itr * self.sampler.batch_size * self.world_size,
            agent_state_dict=self.agent.state_dict(),
            optimizer_state_dict=self.algo.optim_state_dict(),
        )

    def save_itr_snapshot(self, itr):
        logger.log("saving snapshot...")
        params = self.get_itr_snapshot(itr)
        logger.save_itr_params(itr, params)
        logger.log("saved")

    def store_diagnostics(self, itr, traj_infos, opt_info):
        self._cum_completed_trajs += len(traj_infos)
        for k, v in self._opt_infos.items():
            new_v = getattr(opt_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])
        self.pbar.update((itr + 1) % self.log_interval_itrs)  # 更新进度条显示

    def log_diagnostics(self, itr, traj_infos=None, eval_time=0):
        if itr > 0:
            self.pbar.stop()  # 停止更新进度条
        self.save_itr_snapshot(itr)
        new_time = time.time()
        self._cum_time = new_time - self._start_time
        train_time_elapsed = new_time - self._last_time - eval_time
        new_updates = self.algo.update_counter - self._last_update_counter
        new_samples = (self.sampler.batch_size * self.world_size *
                       self.log_interval_itrs)
        updates_per_second = (float('nan') if itr == 0 else
                              new_updates / train_time_elapsed)
        samples_per_second = (float('nan') if itr == 0 else
                              new_samples / train_time_elapsed)
        replay_ratio = (new_updates * self.algo.batch_size * self.world_size /
                        new_samples)
        cum_replay_ratio = (self.algo.batch_size * self.algo.update_counter /
                            ((itr + 1) * self.sampler.batch_size))  # world_size cancels.
        cum_steps = (itr + 1) * self.sampler.batch_size * self.world_size

        if self._eval:
            logger.record_tabular('CumTrainTime',
                                  self._cum_time - self._cum_eval_time)  # Already added new eval_time.
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('CumTime (s)', self._cum_time)
        logger.record_tabular('CumSteps', cum_steps)
        logger.record_tabular('CumCompletedTrajs', self._cum_completed_trajs)
        logger.record_tabular('CumUpdates', self.algo.update_counter)
        logger.record_tabular('StepsPerSecond', samples_per_second)
        logger.record_tabular('UpdatesPerSecond', updates_per_second)
        logger.record_tabular('ReplayRatio', replay_ratio)
        logger.record_tabular('CumReplayRatio', cum_replay_ratio)
        self._log_infos(traj_infos)
        logger.dump_tabular(with_prefix=False)

        self._last_time = new_time
        self._last_update_counter = self.algo.update_counter
        if itr < self.n_itr - 1:
            logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
            self.pbar = ProgBarCounter(self.log_interval_itrs)

    def _log_infos(self, traj_infos=None):
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if not k.startswith("_"):
                    logger.record_tabular_misc_stat(k,
                                                    [info[k] for info in traj_infos])

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)


class MinibatchRl(MinibatchRlBase):
    """Runs RL on minibatches; tracks performance online using learning
    trajectories."""

    def __init__(self, log_traj_window=100, **kwargs):
        super().__init__(**kwargs)
        self.log_traj_window = int(log_traj_window)

    def train(self):
        n_itr = self.startup()
        for itr in range(n_itr):
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)  # Might not be this agent sampling.
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    self.log_diagnostics(itr)
        self.shutdown()

    def initialize_logging(self):
        self._traj_infos = deque(maxlen=self.log_traj_window)
        self._new_completed_trajs = 0
        logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
        super().initialize_logging()
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    def store_diagnostics(self, itr, traj_infos, opt_info):
        self._new_completed_trajs += len(traj_infos)
        self._traj_infos.extend(traj_infos)
        super().store_diagnostics(itr, traj_infos, opt_info)

    def log_diagnostics(self, itr):
        logger.record_tabular('NewCompletedTrajs', self._new_completed_trajs)
        logger.record_tabular('StepsInTrajWindow',
                              sum(info["Length"] for info in self._traj_infos))
        super().log_diagnostics(itr)
        self._new_completed_trajs = 0


class MinibatchRlEval(MinibatchRlBase):
    """Runs RL on minibatches; tracks performance offline using evaluation
    trajectories."""

    _eval = True

    def train(self):
        n_itr = self.startup()  # 调用startup()会导致调用父类的__init__()方法，从而会把外面的algo，agent，sampler传进去
        with logger.prefix(f"itr #0 "):
            eval_traj_infos, eval_time = self.evaluate_agent(0)
            self.log_diagnostics(0, eval_traj_infos, eval_time)
        for itr in range(n_itr):
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    eval_traj_infos, eval_time = self.evaluate_agent(itr)
                    self.log_diagnostics(itr, eval_traj_infos, eval_time)
        self.shutdown()

    def evaluate_agent(self, itr):
        if itr > 0:
            self.pbar.stop()  # 停止进度条
        logger.log("Evaluating agent...")
        self.agent.eval_mode(itr)  # Might be agent in sampler.
        eval_time = -time.time()
        traj_infos = self.sampler.evaluate_agent(itr)
        eval_time += time.time()
        logger.log("Evaluation runs complete.")
        return traj_infos, eval_time

    def initialize_logging(self):
        super().initialize_logging()
        self._cum_eval_time = 0

    def log_diagnostics(self, itr, eval_traj_infos, eval_time):
        if not eval_traj_infos:
            logger.log("WARNING: had no complete trajectories in eval.")
        steps_in_eval = sum([info["Length"] for info in eval_traj_infos])
        logger.record_tabular('StepsInEval', steps_in_eval)
        logger.record_tabular('TrajsInEval', len(eval_traj_infos))
        self._cum_eval_time += eval_time
        logger.record_tabular('CumEvalTime', self._cum_eval_time)
        super().log_diagnostics(itr, eval_traj_infos, eval_time)
