import numpy as np

from rlpyt.agents.base import AgentInputs
from rlpyt.samplers.collectors import (DecorrelatingStartCollector, BaseEvalCollector)
from rlpyt.utils.buffer import (torchify_buffer, numpify_buffer, buffer_from_example, buffer_method)


class CpuResetCollector(DecorrelatingStartCollector):
    mid_batch_reset = True

    def collect_batch(self, agent_inputs, traj_infos, itr):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        """
        收集(即采样)一批数据。这里面会发生推断action的过程(NN的前向传播)。
        整个过程中，会发生几种step(步进)事件 ：在agent中step()，在environment中step()，在trajectory中step()。

        :param agent_inputs:
        :param traj_infos: TrajInfo类对象组成的一个list，包含trajectory的一些统计信息。
        :param itr: 第几次迭代。
        :return: AgentInputs, list(TrajInfo对象), list(TrajInfo对象)
        """
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        observation, action, reward = agent_inputs  # 右式：一个namedarraytuple，参见 rlpyt/agents/base.py 中的 AgentInputs
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_inputs)
        agent_buf.prev_action[0] = action  # Leading prev_action.
        env_buf.prev_reward[0] = reward
        self.agent.sample_mode(itr)
        for t in range(self.batch_T):  # batch_T：每个sampler迭代有多少个step
            env_buf.observation[t] = observation
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)  # 根据输入选择一个action，策略网络的前向传播过程在这里发生
            action = numpify_buffer(act_pyt)  # action由torch.Tensor转换成numpy array格式
            for b, env in enumerate(self.envs):
                # Environment inputs and outputs are numpy arrays.
                o, r, d, env_info = env.step(action[b])  # 计算reward，统计environment信息等
                traj_infos[b].step(observation[b], action[b], r, d, agent_info[b], env_info)  # 统计trajectory的信息
                if getattr(env_info, "traj_done", d):  # EnvInfo里包含 traj_done 的情况，有可能是游戏玩得烂一下子就game over了
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()  # TrajInfo类的对象
                    o = env.reset()
                if d:  # done标志。对游戏来说，done的情况包含一局游戏game over，也包含没有剩余的生命了(TODO:确认是否正确？)
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
                env_buf.done[t, b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = action
            env_buf.reward[t] = reward
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(obs_pyt, act_pyt, rew_pyt)

        return AgentInputs(observation, action, reward), traj_infos, completed_infos


class CpuWaitResetCollector(DecorrelatingStartCollector):
    mid_batch_reset = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.need_reset = np.zeros(len(self.envs), dtype=np.bool)
        self.done = np.zeros(len(self.envs), dtype=np.bool)
        self.temp_observation = buffer_method(
            self.samples_np.env.observation[0, :len(self.envs)], "copy")

    def collect_batch(self, agent_inputs, traj_infos, itr):
        # Numpy arrays can be written to from numpy arrays or torch tensors
        # (whereas torch tensors can only be written to from torch tensors).
        agent_buf, env_buf = self.samples_np.agent, self.samples_np.env
        completed_infos = list()
        observation, action, reward = agent_inputs
        b = np.where(self.done)[0]
        observation[b] = self.temp_observation[b]
        self.done[:] = False  # Did resets between batches.
        obs_pyt, act_pyt, rew_pyt = torchify_buffer(agent_inputs)
        agent_buf.prev_action[0] = action  # Leading prev_action.
        env_buf.prev_reward[0] = reward
        self.agent.sample_mode(itr)
        for t in range(self.batch_T):
            env_buf.observation[t] = observation
            # Agent inputs and outputs are torch tensors.
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                if self.done[b]:
                    action[b] = 0  # Record blank.
                    reward[b] = 0
                    if agent_info:
                        agent_info[b] = 0
                    # Leave self.done[b] = True, record that.
                    continue
                # Environment inputs and outputs are numpy arrays.
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, d, agent_info[b],
                                   env_info)
                if getattr(env_info, "traj_done", d):
                    completed_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    self.need_reset[b] = True
                if d:
                    self.temp_observation[b] = o
                    o = 0  # Record blank.
                observation[b] = o
                reward[b] = r
                self.done[b] = d
                if env_info:
                    env_buf.env_info[t, b] = env_info
            agent_buf.action[t] = action
            env_buf.reward[t] = reward
            env_buf.done[t] = self.done
            if agent_info:
                agent_buf.agent_info[t] = agent_info

        if "bootstrap_value" in agent_buf:
            # agent.value() should not advance rnn state.
            agent_buf.bootstrap_value[:] = self.agent.value(obs_pyt, act_pyt, rew_pyt)

        return AgentInputs(observation, action, reward), traj_infos, completed_infos

    def reset_if_needed(self, agent_inputs):
        for b in np.where(self.need_reset)[0]:
            agent_inputs[b] = 0
            agent_inputs.observation[b] = self.envs[b].reset()
            self.agent.reset_one(idx=b)
        self.need_reset[:] = False


class CpuEvalCollector(BaseEvalCollector):

    def collect_evaluation(self, itr):
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        observations = list()
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(self.envs[0].action_space.null_value(),
                                     len(self.envs))
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        self.agent.eval_mode(itr)
        for t in range(self.max_T):
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)
            for b, env in enumerate(self.envs):
                o, r, d, env_info = env.step(action[b])
                traj_infos[b].step(observation[b], action[b], r, d,
                                   agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    self.traj_infos_queue.put(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    action[b] = 0  # Next prev_action.
                    r = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
            if self.sync.stop_eval.value:
                break
        self.traj_infos_queue.put(None)  # End sentinel.
