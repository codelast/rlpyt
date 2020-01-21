
import psutil
import time
import torch

from rlpyt.utils.collections import AttrDict
from rlpyt.utils.logging import logger
from rlpyt.utils.seed import set_seed


def initialize_worker(rank, seed=None, cpu=None, torch_threads=None):
    """
    初始化采样用的worker。

    :param rank: 采样进程的标识序号。
    :param seed: 种子，一个整数值。
    :param cpu: CPU序号，例如 0, 1, 2 等等。
    :param torch_threads: CPU并发执行的线程数
    """
    log_str = f"Sampler rank {rank} initialized"
    cpu = [cpu] if isinstance(cpu, int) else cpu
    p = psutil.Process()
    try:
        if cpu is not None:
            p.cpu_affinity(cpu)  # 设置CPU亲和性(MacOS不支持)
        cpu_affin = p.cpu_affinity()
    except AttributeError:
        cpu_affin = "UNAVAILABLE MacOS"
    log_str += f", CPU affinity {cpu_affin}"
    torch_threads = (1 if torch_threads is None and cpu is not None else
        torch_threads)  # Default to 1 to avoid possible MKL hang.
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)  # 设置CPU并发执行的线程数
    log_str += f", Torch threads {torch.get_num_threads()}"
    if seed is not None:
        set_seed(seed)
        time.sleep(0.3)  # (so the printing from set_seed is not intermixed)
        log_str += f", Seed {seed}"
    logger.log(log_str)


def sampling_process(common_kwargs, worker_kwargs):
    """
    Arguments fed from the Sampler class in master process.

    采样进程函数。

    :param common_kwargs: 各个worker通用的参数列表。
    :param worker_kwargs: 各个worker可能不同的参数列表。
    """
    c, w = AttrDict(**common_kwargs), AttrDict(**worker_kwargs)
    initialize_worker(w.rank, w.seed, w.cpus, c.torch_threads)
    # 初始化用于training的environment实例和collector实例
    envs = [c.EnvCls(**c.env_kwargs) for _ in range(w.n_envs)]
    collector = c.CollectorCls(
        rank=w.rank,
        envs=envs,
        samples_np=w.samples_np,
        batch_T=c.batch_T,
        TrajInfoCls=c.TrajInfoCls,
        agent=c.get("agent", None),  # Optional depending on parallel setup.
        sync=w.get("sync", None),
        step_buffer_np=w.get("step_buffer_np", None),
        global_B=c.get("global_B", 1),
        env_ranks=w.get("env_ranks", None),
    )
    agent_inputs, traj_infos = collector.start_envs(c.max_decorrelation_steps)  # 这里会做收集(采样)第一批数据的工作
    collector.start_agent()  # collector的初始化

    # 初始化用于evaluation的environment实例和collector实例
    if c.get("eval_n_envs", 0) > 0:
        eval_envs = [c.EnvCls(**c.eval_env_kwargs) for _ in range(c.eval_n_envs)]
        eval_collector = c.eval_CollectorCls(
            rank=w.rank,
            envs=eval_envs,
            TrajInfoCls=c.TrajInfoCls,
            traj_infos_queue=c.eval_traj_infos_queue,
            max_T=c.eval_max_T,
            agent=c.get("agent", None),
            sync=w.get("sync", None),
            step_buffer_np=w.get("eval_step_buffer_np", None),
        )
    else:
        eval_envs = list()

    ctrl = c.ctrl  # 用于控制多个worker进程同时运行时能正确运作的控制器
    ctrl.barrier_out.wait()  # 每个worker都有一个wait()，加上ParallelSamplerBase.initialize()中的一个wait()，刚好n_worker+1个
    while True:
        collector.reset_if_needed(agent_inputs)  # Outside barrier?
        ctrl.barrier_in.wait()
        if ctrl.quit.value:  # 在主进程中set了这个值为True时，所有worker进程会退出采样
            break
        if ctrl.do_eval.value:  # 在主进程的evaluate_agent()函数里set了这个值为True时，这里才会收集evaluation用的数据
            eval_collector.collect_evaluation(ctrl.itr.value)  # Traj_infos to queue inside.
        else:  # 不是做evaluation
            agent_inputs, traj_infos, completed_infos = collector.collect_batch(
                agent_inputs, traj_infos, ctrl.itr.value)
            for info in completed_infos:
                c.traj_infos_queue.put(info)  # 向所有worker进程共享的队列塞入当前worker的统计数据
        ctrl.barrier_out.wait()

    # 清理environment
    for env in envs + eval_envs:
        env.close()
