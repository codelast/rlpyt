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
    _eval = False  # 该标志设置为False的时候，不会记录很多额外的统计信息到日志里

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
        """
        一些初始化工作。
        """
        p = psutil.Process()

        # 设置CPU亲和性(MacOS不支持)
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

        """
        ￿batch_spec.size的实现参见 BatchSpec 类，它表示的所有的environment实例上的所有时间步的数量总和。这里又乘了一个没有说明含义的
        world_size，用一个不正经的比喻，我猜这里有大概是"平行宇宙"的概念(美剧闪电侠)，在"当前宇宙"内的发生的采样，它是算在 batch_spec.size 
        内的，而像这样的场景，我们可以把它复制很多个出来，用所有这些创造出来的集合来训练RL模型。
        """
        self.itr_batch_size = self.sampler.batch_spec.size * world_size  # 所有迭代的时间步数
        n_itr = self.get_n_itr()  # 计算模型训练的迭代次数。在这里，迭代次数并不是直接指定的，而是经过一个复杂的方法计算出来的
        self.agent.to_device(self.affinity.get("cuda_idx", None))  # 在指定的设备上运行程序
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
        [函数名起得不好，让人迷惑]
        用于获取算法(Algorithm)实例里设置的某些参数，例如折扣因子(discount factor)。

        :return: 一个dict，里面只含有折扣因子(discount factor)。
        """
        return dict(discount=getattr(self.algo, "discount", 1))  # discount factor默认值为1，即没有折扣

    def get_n_itr(self):
        """
        获取训练模型的迭代次数。
        """
        # 周期性记录日志的次数。max()保证了记录日志至少1次
        log_interval_itrs = max(self.log_interval_steps // self.itr_batch_size, 1)  # // 先做除法(/)，然后向下取整(floor)
        n_itr = math.ceil(self.n_steps / self.log_interval_steps) * log_interval_itrs
        self.log_interval_itrs = log_interval_itrs  # 周期性记录日志的次数
        self.n_itr = n_itr
        logger.log(f"Running {n_itr} iterations of minibatch RL.")
        return n_itr

    def initialize_logging(self):
        """
        初始化日志记录相关的参数。
        """
        self._opt_infos = {k: list() for k in self.algo.opt_info_fields}
        self._start_time = self._last_time = time.time()
        self._cum_time = 0.
        self._cum_completed_trajs = 0
        self._last_update_counter = 0

    def shutdown(self):
        """
        模型训练完成之后的清理工作。
        """
        logger.log("Training complete.")
        self.pbar.stop()  # 停止更新进度条
        self.sampler.shutdown()  # 一些清理工作

    def get_itr_snapshot(self, itr):
        """
        获取指定的某次迭代里的一些数据。
        :param itr: 当前是第几次迭代
        :return: 一个dict，其包含了指定的某次迭代的数据。
        """
        return dict(
            itr=itr,
            cum_steps=itr * self.sampler.batch_size * self.world_size,
            agent_state_dict=self.agent.state_dict(),  # 模型的状态，例如模型参数，模型的持久化buffer
            optimizer_state_dict=self.algo.optim_state_dict(),  # optimizer的状态
        )

    def save_itr_snapshot(self, itr):
        """
        保存指定的某次迭代的快照数据到日志中。所谓快照数据是指模型参数等，保存到日志文件有利于debug问题。
        :param itr: 第几次迭代。
        """
        logger.log("saving snapshot...")
        params = self.get_itr_snapshot(itr)  # 获取第iter次迭代的快照数据
        logger.save_itr_params(itr, params)  # 保存第iter次迭代的快照数据
        logger.log("saved")

    def store_diagnostics(self, itr, traj_infos, opt_info):
        """
        更新/保存诊断信息。此函数不写日志，只更新内存中的一些统计数据。
        :param itr: 第几次迭代。
        """
        self._cum_completed_trajs += len(traj_infos)
        for k, v in self._opt_infos.items():
            new_v = getattr(opt_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])
        self.pbar.update((itr + 1) % self.log_interval_itrs)  # 更新进度条显示

    def log_diagnostics(self, itr, traj_infos=None, eval_time=0):
        """
        记录诊断信息(写日志)。
        :param itr: 第几次迭代。
        """
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

        # 写一些额外的统计信息到日志里
        if self._eval:
            logger.record_tabular('CumTrainTime', self._cum_time - self._cum_eval_time)  # Already added new eval_time.
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
        logger.dump_tabular(with_prefix=False)  # 写日志文件

        self._last_time = new_time
        self._last_update_counter = self.algo.update_counter
        if itr < self.n_itr - 1:
            logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
            self.pbar = ProgBarCounter(self.log_interval_itrs)  # 进度条

    def _log_infos(self, traj_infos=None):
        """
        记录trajectory的信息。
        :param traj_infos:
        :return:
        """
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if not k.startswith("_"):
                    logger.record_tabular_misc_stat(k, [info[k] for info in traj_infos])

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)


class MinibatchRl(MinibatchRlBase):
    """
    Runs RL on minibatches; tracks performance online using learning trajectories.
    按mini-batch来训练模型，但不对模型做evaluation。
    """

    def __init__(self, log_traj_window=100, **kwargs):
        super().__init__(**kwargs)
        self.log_traj_window = int(log_traj_window)

    def train(self):
        n_itr = self.startup()  # 调用startup()会导致调用父类的__init__()方法，从而会把外面的algo，agent，sampler传进去
        for itr in range(n_itr):  # 重复训练N轮
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)  # Might not be this agent sampling. # 设置成采样模式
                samples, traj_infos = self.sampler.obtain_samples(itr)  # 采样一批数据
                self.agent.train_mode(itr)  # 把神经网络module设置成训练模式，传进入的迭代次数其实没用
                opt_info = self.algo.optimize_agent(itr, samples)  # 训练模型，反向传播之类的工作就是在这里面做的
                self.store_diagnostics(itr, traj_infos, opt_info)  # 更新内存中的一些统计数据
                if (itr + 1) % self.log_interval_itrs == 0:
                    self.log_diagnostics(itr)  # 记录诊断信息(写日志)
        self.shutdown()

    def initialize_logging(self):
        """
        初始化日志记录相关的参数。
        """
        self._traj_infos = deque(maxlen=self.log_traj_window)
        self._new_completed_trajs = 0
        logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
        super().initialize_logging()
        self.pbar = ProgBarCounter(self.log_interval_itrs)  # 进度条

    def store_diagnostics(self, itr, traj_infos, opt_info):
        """
        更新/保存诊断信息。此函数不写日志，只更新内存中的一些统计数据。
        :param itr: 第几次迭代。
        """
        self._new_completed_trajs += len(traj_infos)
        self._traj_infos.extend(traj_infos)
        super().store_diagnostics(itr, traj_infos, opt_info)

    def log_diagnostics(self, itr):
        """
        记录诊断信息。
        :param itr: 第几次迭代。
        """
        logger.record_tabular('NewCompletedTrajs', self._new_completed_trajs)
        logger.record_tabular('StepsInTrajWindow',
                              sum(info["Length"] for info in self._traj_infos))
        super().log_diagnostics(itr)
        self._new_completed_trajs = 0


class MinibatchRlEval(MinibatchRlBase):
    """
    Runs RL on minibatches; tracks performance offline using evaluation trajectories.
    按mini-batch来训练模型，并且每隔一定的周期就evaluate一次模型。
    """

    _eval = True  # 设置为True的时候，在记录诊断信息(log_diagnostics)的时候，会把很多额外的统计信息记录到日志里

    def train(self):
        n_itr = self.startup()  # 调用startup()会导致调用父类的__init__()方法，从而会把外面的algo，agent，sampler传进去
        with logger.prefix(f"itr #0 "):
            eval_traj_infos, eval_time = self.evaluate_agent(0)  # 开始训练模型之前先evaluate一次
            self.log_diagnostics(0, eval_traj_infos, eval_time)  # 记录诊断信息(写日志)
        for itr in range(n_itr):  # 重复训练N轮
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)  # 设置成采样模式
                samples, traj_infos = self.sampler.obtain_samples(itr)  # 采样一批数据
                self.agent.train_mode(itr)  # 把神经网络module设置成训练模式，传进入的迭代次数其实没用
                opt_info = self.algo.optimize_agent(itr, samples)  # 训练模型，反向传播之类的工作就是在这里面做的
                self.store_diagnostics(itr, traj_infos, opt_info)  # 更新内存中的一些统计数据
                if (itr + 1) % self.log_interval_itrs == 0:  # 每迭代到记录一次日志的步数
                    eval_traj_infos, eval_time = self.evaluate_agent(itr)  # 评估模型
                    self.log_diagnostics(itr, eval_traj_infos, eval_time)  # 记录诊断信息(写日志)
        self.shutdown()  # 模型训练完成后的清理工作

    def evaluate_agent(self, itr):
        """
        评估模型。
        :param itr: 第几次迭代。
        :return: 一个tuple，包含trajectory的信息以及evaluation所消耗的时间。
        """
        if itr > 0:
            self.pbar.stop()  # 停止进度条
        logger.log("Evaluating agent...")
        self.agent.eval_mode(itr)  # Might be agent in sampler. 设置成evaluation模式
        eval_time = -time.time()  # 当前时间取负值
        traj_infos = self.sampler.evaluate_agent(itr)  # 真正开始做evaluation的地方
        eval_time += time.time()  # 经过这么一计算，现在eval_time变成了上一条语句的执行消耗时间
        logger.log("Evaluation runs complete.")
        return traj_infos, eval_time

    def initialize_logging(self):
        """
        初始化日志记录相关的参数。
        """
        super().initialize_logging()
        self._cum_eval_time: float = 0

    def log_diagnostics(self, itr, eval_traj_infos, eval_time: float):
        """
        [此函数设计得不好，和父类的同名函数签名不一致]
        记录诊断信息。此函数会写日志。
        :param itr: 第几次迭代。
        :param eval_traj_infos:
        :param eval_time:
        :return:
        """
        if not eval_traj_infos:
            logger.log("WARNING: had no complete trajectories in eval.")
        steps_in_eval = sum([info["Length"] for info in eval_traj_infos])
        logger.record_tabular('StepsInEval', steps_in_eval)
        logger.record_tabular('TrajsInEval', len(eval_traj_infos))
        self._cum_eval_time += eval_time  # 累积的evaluation消耗时间
        logger.record_tabular('CumEvalTime', self._cum_eval_time)
        super().log_diagnostics(itr, eval_traj_infos, eval_time)
