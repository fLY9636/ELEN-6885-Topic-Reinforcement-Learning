#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import sys
import pickle

from network import DRLNetwork
from A3C_Thread import A3C_Thread
from environment import GridWorldEnvironment as Environment

from constants import CHECKPOINT_DIR
from constants import NUM_EVAL_EPISODES
from constants import VERBOSE

from constants import TASK_TYPE
from constants import TASK_LIST

class Evaluate(object):
  def __init__(self, global_step):

    device = "/cpu:0" # use CPU for display tool
    network_scope = TASK_TYPE
    list_of_tasks = TASK_LIST
    scene_scopes = list_of_tasks.keys()

    global_network = DRLNetwork(action_size= 4,
                                          device=device,
                                          network_scope=network_scope,
                                          scene_scopes=scene_scopes)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()

    # Read network from checkpoint file
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)

    if checkpoint and checkpoint.model_checkpoint_path:
      saver.restore(sess, checkpoint.model_checkpoint_path)
      print("checkpoint loaded: {}".format(checkpoint.model_checkpoint_path))
    else:
      print("Could not find old checkpoint")
    
    # Read checkpoint directly, 'meta' file
    # saver.restore(sess, checkpoint.model_checkpoint_path)


    scene_stats = dict()
    self.results = dict()
    for scene_scope in scene_scopes:

      scene_stats[scene_scope] = []
      for task_scope in list_of_tasks[scene_scope]:

        env = Environment({
          'scene_name': scene_scope,
          'terminal_state_id': int(task_scope)
        })
        ep_rewards = []
        ep_lengths = []
        ep_collisions = []

        scopes = [network_scope, scene_scope, task_scope]

        for i_episode in range(NUM_EVAL_EPISODES):

          env.reset()
          terminal = False
          ep_reward = 0
          ep_collision = 0
          ep_t = 0

          while not terminal:

            pi_values = global_network.run_policy(sess, env.s_t, env.target, scopes)
            pi_values = np.array(pi_values)/np.sum(pi_values)
            action = np.random.choice(np.arange(len(pi_values)), p = pi_values)
            env.step(action)

            terminal = env.terminal
            if ep_t == 10000: break
            if env.collided: ep_collision += 1
            ep_reward += env.reward
            ep_t += 1

          ep_lengths.append(ep_t)
          ep_rewards.append(ep_reward)
          ep_collisions.append(ep_collision)
          if VERBOSE: print("episode #{} ends after {} steps".format(i_episode, ep_t))

        print('evaluation: %s %s' % (scene_scope, task_scope))
        print('mean episode reward: %.2f' % np.mean(ep_rewards))
        print('mean episode length: %.2f' % np.mean(ep_lengths))
        print('mean episode collision: %.2f' % np.mean(ep_collisions))

        scene_stats[scene_scope].extend(ep_lengths)

        
    print('\nResults (average trajectory length):')    
    for scene_scope in scene_stats:
      self.results[scene_scope] = np.mean(scene_stats[scene_scope])
      print('%s: %.2f steps'%(scene_scope, self.results[scene_scope]))

    with open("./Evaluation/result_%d.txt"%global_step, 'wb') as fp:
        pickle.dump(self.results, fp)
    

  def get_result(self):
    return self.results

if __name__ == '__main__':
  evalution = Evaluate(10)
  evalution.get_result()