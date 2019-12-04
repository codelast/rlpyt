import multiprocessing as mp

import numpy as np

from rlpyt.agents.base import AgentInputs
from rlpyt.samplers.collections import (Samples, AgentSamples, AgentSamplesBsv,
                                        EnvSamples)
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer


def build_samples_buffer(agent, env, batch_spec, bootstrap_value=False,
                         agent_shared=True, env_shared=True, subprocess=True, examples=None):
    """Recommended to step/reset agent and env in subprocess, so it doesn't
    affect settings in master before forking workers (e.g. torch num_threads
    (MKL) may be set at first forward computation.)

    :param agent: 一个Agent类的对象。
    :param env: 一个environment类的对象。
    :param batch_spec: 一个BatchSpec类的对象。
    """
    if examples is None:
        if subprocess:  # 创建子进程
            mgr = mp.Manager()  # Manager模块用于资源共享
            examples = mgr.dict()  # Examples pickled back to master. 可以被子进程共享的全局变量
            w = mp.Process(target=get_example_outputs,
                           args=(agent, env, examples, subprocess))  # 创建worker进程，此进程执行的是target指定的函数，参数由args指定
            w.start()
            w.join()
        else:
            examples = dict()
            get_example_outputs(agent, env, examples)  # examples会在get_example_outputs()函数中被更新，所以没有返回值

    T, B = batch_spec  # time step数，以及environment实例数
    all_action = buffer_from_example(examples["action"], (T + 1, B), agent_shared)
    action = all_action[1:]
    prev_action = all_action[:-1]  # Writing to action will populate prev_action.
    agent_info = buffer_from_example(examples["agent_info"], (T, B), agent_shared)
    agent_buffer = AgentSamples(
        action=action,
        prev_action=prev_action,
        agent_info=agent_info,
    )
    if bootstrap_value:
        bv = buffer_from_example(examples["agent_info"].value, (1, B), agent_shared)
        agent_buffer = AgentSamplesBsv(*agent_buffer, bootstrap_value=bv)

    observation = buffer_from_example(examples["observation"], (T, B), env_shared)
    all_reward = buffer_from_example(examples["reward"], (T + 1, B), env_shared)
    reward = all_reward[1:]
    prev_reward = all_reward[:-1]  # Writing to reward will populate prev_reward.
    done = buffer_from_example(examples["done"], (T, B), env_shared)
    env_info = buffer_from_example(examples["env_info"], (T, B), env_shared)
    env_buffer = EnvSamples(
        observation=observation,
        reward=reward,
        prev_reward=prev_reward,
        done=done,
        env_info=env_info,
    )
    samples_np = Samples(agent=agent_buffer, env=env_buffer)
    samples_pyt = torchify_buffer(samples_np)
    return samples_pyt, samples_np, examples


def get_example_outputs(agent, env, examples, subprocess=False):
    """Do this in a sub-process to avoid setup conflict in master/workers (e.g. MKL).
    在一个重置的environment中(从头开始)，随机采取一个action，把得到的observation，reward等数据记录下来，保存到输入的examples里返回。
    注意：虽然输入的examples看上去名字是复数，但实际上，返回的并不是在environment中走多步的结果，而仅仅是走一步产生的结果。这个变量命名不好，
    我认为叫"example"更合理。

    :param agent: 一个agent类的对象。
    :param env: 一个environment类的对象。
    :param examples: 同时作为input和output。输入的有可能是一个空的dict，输出的是经过填充过的内容。
    :param subprocess: 是否是在子进程中执行。
    :return: 没有返回值，但需要返回的数据放在了输入的examples变量中返回。
    """
    if subprocess:  # i.e. in subprocess.
        import torch
        torch.set_num_threads(1)  # Some fix to prevent MKL hang.
    o = env.reset()  # 重置environment，从头开始
    a = env.action_space.sample()  # 随机选择action space内的一个index
    o, r, d, env_info = env.step(a)  # 根据选择的action(的index)，在environment中步进(step)
    r = np.asarray(r, dtype="float32")  # Must match torch float dtype here. 把reward转成float32类型
    agent.reset()
    agent_inputs = torchify_buffer(AgentInputs(o, a, r))
    a, agent_info = agent.step(*agent_inputs)  # 星号把agent_inputs这一个tuple展开成step()函数所需的3个参数输入
    if "prev_rnn_state" in agent_info:
        # Agent leaves B dimension in, strip it: [B,N,H] --> [N,H]
        agent_info = agent_info._replace(prev_rnn_state=agent_info.prev_rnn_state[0])
    examples["observation"] = o
    examples["reward"] = r  # 只含有一个数的NumPy array
    examples["done"] = d  # bool
    examples["env_info"] = env_info  # EnvInfo类型的对象
    examples["action"] = a  # OK to put torch tensor here, could numpify.
    examples["agent_info"] = agent_info
