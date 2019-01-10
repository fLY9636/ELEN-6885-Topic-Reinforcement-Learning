# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys

from utils.accum_trainer import AccumTrainer
from environment import GridWorldEnvironment as Environment
from network import DRLNetwork

from constants import ACTION_SIZE
from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import VERBOSE
from constants import EPSILON
from constants import ACTION_SIZE
from constants import MEMORY_SIZE
from constants import DQN_BATCH_SIZE
from constants import DQN_REPLACE_TARGET_ITER
from constants import REWARD_DECAY

class ADQN_Thread(object):
  def __init__(self, thread_index, global_network, initial_learning_rate,
               learning_rate_input, grad_applier, max_global_time_step,
               device, network_scope="network", scene_scope="scene",
               task_scope="task"):
    
    self.thread_index = thread_index                                        # Number the thread

    self._set_local_network(device, network_scope, scene_scope, task_scope) # Set local network

    self.sync = self.local_network.sync_from(global_network)                # Synthesize from the global network

    self.learning_rate_input = learning_rate_input                          # Set learning rate

    self.max_global_time_step = max_global_time_step                        # Set maximum of global time step
    
    self._set_trainer_optimizer(device, global_network, grad_applier)                     # Set trainer
    
    self._set_environment(initial_learning_rate)                            # Set environment

    self.memory_size = MEMORY_SIZE # memory size for replay buffer

    self.memory = np.zeros((self.memory_size, 2048 * 4 * 2 + 2))  # initialize zero memory [s, a, r, s_]

    self.replace_target_iter = DQN_REPLACE_TARGET_ITER

    self.batch_size = DQN_BATCH_SIZE

    self.gamma = REWARD_DECAY


  # Create local network
  def _set_local_network(self, device, network_scope, scene_scope, task_scope):
    self.local_network = DRLNetwork(action_size=ACTION_SIZE, device=device, network_scope=network_scope,
                           scene_scopes=[scene_scope])
    
    self.network_scope = network_scope
    self.scene_scope = scene_scope
    self.task_scope = task_scope
    self.scopes = [network_scope, scene_scope, task_scope]
    self.local_network.prepare_loss(ENTROPY_BETA, self.scopes)

  # Set trainer and optimizer
  # Set Actor-Critic gradient and optimizer
  # Use the accumulated trainer from Zhu
  def _set_trainer_optimizer(self, device, global_network, grad_applier):
    self.trainer = AccumTrainer(device)
    self.trainer.prepare_minimize(self.local_network.total_loss,
                                  self.local_network.get_vars())

    self.accum_gradients = self.trainer.accumulate_gradients()
    self.reset_gradients = self.trainer.reset_gradients()

    accum_grad_names = [self._local_var_name(x) for x in self.trainer.get_accum_grad_list()]
    global_net_vars = [x for x in global_network.get_vars() if self._get_accum_grad_name(x) in accum_grad_names]

    self.apply_gradients = grad_applier.apply_gradients(global_net_vars, self.trainer.get_accum_grad_list() )

  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])

  def _get_accum_grad_name(self, var):
    return self._local_var_name(var).replace(':','_') + '_accum_grad:0'

  # Set environments
  def _set_environment(self, initial_learning_rate):
    self.env = None
    self.local_t = 0
    self.initial_learning_rate = initial_learning_rate
    self.episode_reward = 0
    self.episode_length = 0


  def choose_action(self, actions_value):
    # epsilon-greedy
    if np.random.uniform() < EPSILON:
      action = np.argmax(actions_value)
    else:
      action = np.random.randint(0, ACTION_SIZE)
    return action

  # Take LOCAL_T_MAX step in one process
  def process(self, sess, global_t, summary_writer, summary_op, summary_placeholders):
    #print("start process")

    if self.env is None:
      # lazy evaluation
      time.sleep(self.thread_index*1.0)
      self.env = Environment({
        'scene_name': self.scene_scope,
        'terminal_state_id': int(self.task_scope)
      })
    start_local_t = self.local_t

    # Reset accmulated gradient variables
    sess.run(self.reset_gradients)
    # Obtain shared parameters from global 
    sess.run( self.sync )

    # t_max times loop
    for i in range(LOCAL_T_MAX):
      old_s_t = self.env.s_t
      actions_value = self.local_network.run_DQN(sess, self.env.s_t, self.env.target, self.scopes)
      action = self.choose_action(actions_value)

      if VERBOSE and (self.thread_index == 0) and (self.local_t % 1000) == 0:
        sys.stdout.write("%s:" % self.scene_scope)
        sys.stdout.write("Pi = {0} V = {1}\n".format(actions_value, action))

      # process game
      self.env.step(action)

      # receive game result
      reward = self.env.reward
      terminal = self.env.terminal

      # ad-hoc reward for navigation
      # reward = 10.0 if terminal else -0.01
      if self.episode_length > 5e3: terminal = True

      self.episode_reward += reward
      self.episode_length += 1

      """
      print("Local t: {0:d}".format(self.local_t))
      print("Reward: {0:f}".format(reward))
      print("Episode reward: {0:f}".format(self.episode_reward))
      print("Episode length: {0:d}".format(self.episode_length))
      """

      self.local_t += 1

      # store transition to replay buffer
      self.store_transition(old_s_t, action, reward, self.env.s_t)

      if terminal:
        sys.stdout.write("#Thread: %d \n time %d | thread #%d | scene %s | target #%s\n%s %s episode reward = %.3f\n%s %s episode length = %d\n%s %s \n" % (self.thread_index, global_t, self.thread_index, self.scene_scope, self.task_scope, self.scene_scope, self.task_scope, self.episode_reward, self.scene_scope, self.task_scope, self.episode_length, self.scene_scope, self.task_scope))

        summary_values = {
          "episode_reward_input": self.episode_reward,
          "episode_length_input": float(self.episode_length),
          "learning_rate_input": self._anneal_learning_rate(global_t)
        }

        self._record_score(sess, summary_writer, summary_op, summary_placeholders,
                           summary_values, global_t)
        self.episode_reward = 0
        self.episode_length = 0
        self.env.reset()

        break

    # update target network
    if self.local_t % self.replace_target_iter == 0:
      sess.run(self.local_network.replace_target_op)
      # print('\ntarget_params_replaced\n')

    # sample batch memory from all memory
    if self.memory_counter > self.memory_size:
      sample_index = np.random.choice(self.memory_size, size=self.batch_size)
    else:
      sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
    batch_memory = self.memory[sample_index, :]

    batch_memory_s_ = np.reshape(batch_memory[:, -2048*4:], (-1, 2048, 4))
    batch_memory_s = np.reshape(batch_memory[:, :2048*4], (-1, 2048, 4))
    batch_memory_t = np.reshape(np.tile(self.env.target, [self.batch_size, 1]), (-1, 2048, 4))

    q_next, q_eval = sess.run(
      [self.local_network.q_next, self.local_network.q_eval],
      feed_dict={
        self.local_network.s_: batch_memory_s_,  # fixed params
        self.local_network.s: batch_memory_s,  # newest params
        self.local_network.t: batch_memory_t
      })

    # change q_target w.r.t q_eval's action
    q_target = q_eval.copy()

    batch_index = np.arange(self.batch_size, dtype=np.int32)
    eval_act_index = batch_memory[:, 2048*4].astype(int)
    reward = batch_memory[:, 2048*4 + 1]

    key_eval = self.network_scope + '/' + self.scene_scope + '/eval'
    if terminal:
      q_target[key_eval][batch_index, eval_act_index] = reward
    else:
      key_target = self.network_scope + '/'+ self.scene_scope + '/target'
      q_target[key_eval][batch_index, eval_act_index] = reward + self.gamma * np.max(q_next[key_target], axis=1)

    for idx in batch_index:
      # train eval network
      sess.run(self.accum_gradients,
               feed_dict={
                 self.local_network.s: [batch_memory_s[idx]],
                 self.local_network.t: [batch_memory_t[idx]],
                 self.local_network.q_target: [q_target[key_eval][idx]]})

      cur_learning_rate = self._anneal_learning_rate(global_t)

      # update global network
      sess.run( self.apply_gradients,
                feed_dict = { self.learning_rate_input: cur_learning_rate } )

    if VERBOSE and (self.thread_index == 0) and (self.local_t % 100) == 0:
      sys.stdout.write("#Thread-%d-%s-Local timestep-%d\n" % (self.thread_index, self.scene_scope, self.local_t))

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t


  def _record_score(self, sess, writer, summary_op, placeholders, values, global_t):
    feed_dict = {}
    for k in placeholders:
      feed_dict[placeholders[k]] = values[k]
    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    if VERBOSE: sys.stdout.write('writing to summary writer at time %d\n' % (global_t))
    writer.add_summary(summary_str, global_t)
    # writer.flush()


  def _anneal_learning_rate(self, global_time_step):
    time_step_to_go = max(self.max_global_time_step - global_time_step, 0.0)
    learning_rate = self.initial_learning_rate * time_step_to_go / self.max_global_time_step
    return learning_rate

  def store_transition(self, s, a, r, s_):
    if not hasattr(self, 'memory_counter'):
      self.memory_counter = 0

    transition = np.hstack((np.reshape(s, -1), [a, r], np.reshape(s_,-1)))

    # replace the old memory with new memory
    index = self.memory_counter % self.memory_size
    self.memory[index, :] = transition

    self.memory_counter += 1
