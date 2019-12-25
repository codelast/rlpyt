
import multiprocessing as mp
import ctypes
import time

from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.buffer import build_samples_buffer
from rlpyt.samplers.parallel.worker import sampling_process
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.synchronize import drain_queue


EVAL_TRAJ_CHECK = 0.1  # seconds.


class ParallelSamplerBase(BaseSampler):

    gpu = False

    ######################################
    # API
    ######################################

    def initialize(
            self,
            agent,
            affinity,
            seed,
            bootstrap_value=False,
            traj_info_kwargs=None,
            world_size=1,
            rank=0,
            worker_process=None,
            ):
        """
        initialize()函数在runner类(例如MinibatchRlBase)中会被调用。
        """
        n_envs_list = self._get_n_envs_list(affinity=affinity)  # 用户设置的worker数不一定与environment数相匹配，这里会重新调整
        self.n_worker = n_worker = len(n_envs_list)  # 经过调整之后的worker数
        B = self.batch_spec.B  # environment实例的数量
        global_B = B * world_size  # "平行宇宙"概念下的environment实例的数量
        env_ranks = list(range(rank * B, (rank + 1) * B))  # 含义可参考：https://www.codelast.com/?p=10932
        self.world_size = world_size
        self.rank = rank

        if self.eval_n_envs > 0:  # 在example_*.py中传入的参数
            self.eval_n_envs_per = max(1, self.eval_n_envs // n_worker)  # 计算每个worker至少承载几个evaluation的environment(至少1)
            self.eval_n_envs = eval_n_envs = self.eval_n_envs_per * n_worker  # 保证至少有"worker数量"个eval environment实例
            logger.log(f"Total parallel evaluation envs: {eval_n_envs}.")
            self.eval_max_T = eval_max_T = int(self.eval_max_steps // eval_n_envs)

        env = self.EnvCls(**self.env_kwargs)  # 实例化environment，参数env_kwargs是example_*.py传入父类BaseSampler的
        self._agent_init(agent, env, global_B=global_B,
            env_ranks=env_ranks)
        examples = self._build_buffers(env, bootstrap_value)
        env.close()  # 在environment类的父类Env中定义了这个空方法
        del env

        self._build_parallel_ctrl(n_worker)

        if traj_info_kwargs:
            for k, v in traj_info_kwargs.items():
                setattr(self.TrajInfoCls, "_" + k, v)  # Avoid passing every init.

        common_kwargs = self._assemble_common_kwargs(affinity, global_B)
        workers_kwargs = self._assemble_workers_kwargs(affinity, seed, n_envs_list)

        # 创建一批子进程
        target = sampling_process if worker_process is None else worker_process
        self.workers = [mp.Process(target=target,
            kwargs=dict(common_kwargs=common_kwargs, worker_kwargs=w_kwargs))
            for w_kwargs in workers_kwargs]
        # 启动子进程
        for w in self.workers:
            w.start()

        self.ctrl.barrier_out.wait()  # Wait for workers ready (e.g. decorrelate).
        return examples  # e.g. In case useful to build replay buffer.

    def obtain_samples(self, itr):
        """
        采样一批数据。

        :param itr: 第几次迭代。
        :return: TODO
        """
        self.ctrl.itr.value = itr
        self.ctrl.barrier_in.wait()
        # Workers step environments and sample actions here.
        self.ctrl.barrier_out.wait()
        traj_infos = drain_queue(self.traj_infos_queue)
        return self.samples_pyt, traj_infos

    def evaluate_agent(self, itr):
        """
        评估模型。

        :param itr: 第几次迭代。
        :return: trajectory的统计信息。
        """
        self.ctrl.itr.value = itr
        self.ctrl.do_eval.value = True
        self.sync.stop_eval.value = False
        self.ctrl.barrier_in.wait()
        traj_infos = list()
        if self.eval_max_trajectories is not None:
            while True:
                time.sleep(EVAL_TRAJ_CHECK)
                traj_infos.extend(drain_queue(self.eval_traj_infos_queue,
                    guard_sentinel=True))
                if len(traj_infos) >= self.eval_max_trajectories:
                    self.sync.stop_eval.value = True
                    logger.log("Evaluation reached max num trajectories "
                        f"({self.eval_max_trajectories}).")
                    break  # Stop possibly before workers reach max_T.
                if self.ctrl.barrier_out.parties - self.ctrl.barrier_out.n_waiting == 1:
                    logger.log("Evaluation reached max num time steps "
                        f"({self.eval_max_T}).")
                    break  # Workers reached max_T.
        self.ctrl.barrier_out.wait()
        traj_infos.extend(drain_queue(self.eval_traj_infos_queue,
            n_sentinel=self.n_worker))
        self.ctrl.do_eval.value = False
        return traj_infos

    def shutdown(self):
        """
        结束时的清理工作。
        """
        self.ctrl.quit.value = True
        self.ctrl.barrier_in.wait()
        for w in self.workers:
            w.join()

    ######################################
    # Helpers
    ######################################

    def _get_n_envs_list(self, affinity=None, n_worker=None, B=None):
        """
        根据environment实例的数量(所谓的"B")，以及用户设定的用于采样的worker的数量(n_worker)，来计算得到一个list，这个list的元素的总数，
        就是最终的worker的数量；而这个list里的每个元素的值，分别是每个worker承载的environment实例的数量。

        :param affinity: 一个字典(dict)，包含硬件亲和性定义。
        :param n_worker: 用户设定的用于采样的worker的数量。
        :param B: environment实例的数量。
        :return 一个list，其含义如上所述。
        """
        B = self.batch_spec.B if B is None else B  # 参考BatchSpec类，可以认为B是environment实例的数量
        n_worker = len(affinity["workers_cpus"]) if n_worker is None else n_worker  # worker的数量(不超过物理CPU数否则在别处报错)
        """
        当environment实例的数量<worker的数量时，例如有8个worker(即8个物理CPU)，5个environment实例，每一个物理CPU运行一个environment，
        那么此时会有3个物理CPU多余，此时就会把worker的数量设置成和environment实例数量一样，使得每个CPU都刚好运行一个environment实例。
        """
        if B < n_worker:
            logger.log(f"WARNING: requested fewer envs ({B}) than available worker "
                f"processes ({n_worker}). Using fewer workers (but maybe better to "
                "increase sampler's `batch_B`.")
            n_worker = B
        n_envs_list = [B // n_worker] * n_worker
        """
        当environment实例的数量不是worker数量的整数倍时，每个worker被分配到的environment实例的数量是不均等的。
        """
        if not B % n_worker == 0:
            logger.log("WARNING: unequal number of envs per process, from "
                f"batch_B {self.batch_spec.B} and n_worker {n_worker} "
                "(possible suboptimal speed).")
            for b in range(B % n_worker):
                n_envs_list[b] += 1
        return n_envs_list

    def _agent_init(self, agent, env, global_B=1, env_ranks=None):
        """
        初始化agent。

        :param agent: agent的一个实例。
        :param env: environment的一个实例。
        :param global_B: 含义可参考：https://www.codelast.com/?p=10883 以及 https://www.codelast.com/?p=10932
        :param env_ranks: 含义可参考：https://www.codelast.com/?p=10932
        """
        agent.initialize(env.spaces, share_memory=True,
            global_B=global_B, env_ranks=env_ranks)
        self.agent = agent

    def _build_buffers(self, env, bootstrap_value):
        self.samples_pyt, self.samples_np, examples = build_samples_buffer(
            self.agent, env, self.batch_spec, bootstrap_value,
            agent_shared=True, env_shared=True, subprocess=True)
        return examples

    def _build_parallel_ctrl(self, n_worker):
        """
        创建用于控制并行训练过程的一些数据结构。

        multiprocessing.RawValue：不存在lock的多进程间共享值。
        multiprocessing.Barrier：一种简单的同步原语，用于固定数目的进程相互等待。当所有进程都调用wait以后，所有进程会同时开始执行。
        multiprocessing.Queue：用于多进程间数据传递的消息队列。

        :param n_worker: 真正的worker数(不一定等于用户设置的那个原始值)。
        """
        self.ctrl = AttrDict(
            quit=mp.RawValue(ctypes.c_bool, False),
            barrier_in=mp.Barrier(n_worker + 1),
            barrier_out=mp.Barrier(n_worker + 1),
            do_eval=mp.RawValue(ctypes.c_bool, False),
            itr=mp.RawValue(ctypes.c_long, 0),
        )
        self.traj_infos_queue = mp.Queue()
        self.eval_traj_infos_queue = mp.Queue()
        # RawValue(typecode_or_type, *args) 返回从共享内存中分配的ctypes对象，这里为bool类型的对象
        self.sync = AttrDict(stop_eval=mp.RawValue(ctypes.c_bool, False))

    def _assemble_common_kwargs(self, affinity, global_B=1):
        """
        创建对各个worker都相同的通用(common)参数字典。

        :param affinity: 亲和性定义，一个字典(dict)。
        :param global_B: 含义可参考：https://www.codelast.com/?p=10883 以及 https://www.codelast.com/?p=10932
        :return: 一个参数字典(dict)。
        """
        common_kwargs = dict(
            EnvCls=self.EnvCls,
            env_kwargs=self.env_kwargs,
            agent=self.agent,
            batch_T=self.batch_spec.T,
            CollectorCls=self.CollectorCls,
            TrajInfoCls=self.TrajInfoCls,
            traj_infos_queue=self.traj_infos_queue,
            ctrl=self.ctrl,
            max_decorrelation_steps=self.max_decorrelation_steps,
            torch_threads=affinity.get("worker_torch_threads", 1),
            global_B=global_B,
        )
        if self.eval_n_envs > 0:
            common_kwargs.update(dict(
                eval_n_envs=self.eval_n_envs_per,
                eval_CollectorCls=self.eval_CollectorCls,
                eval_env_kwargs=self.eval_env_kwargs,
                eval_max_T=self.eval_max_T,
                eval_traj_infos_queue=self.eval_traj_infos_queue,
                )
            )
        return common_kwargs

    def _assemble_workers_kwargs(self, affinity, seed, n_envs_list):
        """
        由于每个worker使用的CPU、承载的environment实例的数量等情况是不相同的，因此它们需要的初始化参数也不同，此函数为每个worker创建不同的
        参数字典。

        :param affinity: 亲和性定义，一个字典(dict)。
        :param seed: 种子，一个整数值。
        :param n_envs_list: 这个list的元素的总数是最终的worker的数量；而这个list里的每个元素的值，分别是每个worker承载的environment
        实例的数量。
        :return: 一个参数字典(dict)。
        """
        workers_kwargs = list()
        i_env = 0
        g_env = sum(n_envs_list) * self.rank
        for rank in range(len(n_envs_list)):
            n_envs = n_envs_list[rank]  # 当前worker承载的environment实例的数量
            slice_B = slice(i_env, i_env + n_envs)
            env_ranks = list(range(g_env, g_env + n_envs))
            worker_kwargs = dict(
                rank=rank,
                env_ranks=env_ranks,
                seed=seed + rank,
                cpus=(affinity["workers_cpus"][rank]
                    if affinity.get("set_affinity", True) else None),
                n_envs=n_envs,
                samples_np=self.samples_np[:, slice_B],
                sync=self.sync,  # Only for eval, on CPU.
            )
            i_env += n_envs
            g_env += n_envs
            workers_kwargs.append(worker_kwargs)
        return workers_kwargs
