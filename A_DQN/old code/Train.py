#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import threading
import numpy as np
import signal
import random

from A3C_thread import A3C_Thread
from network import DRLNetwork
from utils.rmsprop_applier import RMSPropApplier

from constants import ACTION_SIZE
from constants import NUM_THREADS
from constants import LR_ALPHA_LOW
from constants import LR_ALPHA_HIGH
from constants import LR_ALPHA_LOG_RATE
from constants import MAX_TIME_STEP
from constants import CHECKPOINT_DIR
from constants import LOG_FILE
from constants import RMSP_EPSILON
from constants import RMSP_ALPHA
from constants import GRAD_NORM_CLIP
from constants import USE_GPU
from constants import TASK_TYPE
from constants import TASK_LIST
      

if __name__ == '__main__':
  
  # -*- Initialize parameters -*-
  device = "/gpu:0" if USE_GPU else "/cpu:0"
  network_scope = TASK_TYPE
  list_of_tasks = TASK_LIST
  scene_scopes = list_of_tasks.keys()
  global_t = 0
  stop_requested = False

  if not os.path.exists(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)
  
  # Initialize learning rate
  log_lo = np.log(LR_ALPHA_LOW)
  log_hi = np.log(LR_ALPHA_HIGH)
  v = log_lo * (1-LR_ALPHA_LOG_RATE) + log_hi * LR_ALPHA_LOG_RATE
  initial_learning_rate = np.exp(v)

  # Create global network
  global_network = DRLNetwork(action_size = ACTION_SIZE, device = device,
                            network_scope = network_scope, scene_scopes = scene_scopes)
  
  # Initialize scene-task specific branch
  branches = []
  for scene in scene_scopes:
    for task in list_of_tasks[scene]:
      branches.append((scene, task))

  NUM_TASKS = len(branches)
  assert NUM_THREADS >= NUM_TASKS, \
    "Not enough threads for multitasking: at least {} threads needed.".format(NUM_TASKS)
  
  # Create gradient applier
  learning_rate_input = tf.placeholder("float")
  grad_applier = RMSPropApplier(learning_rate = learning_rate_input,
                                decay = RMSP_ALPHA,
                                momentum = 0.0,
                                epsilon = RMSP_EPSILON,
                                clip_norm = GRAD_NORM_CLIP,
                                device = device)

  # Create training thread
  # Each thread is training for one target in one scene
  training_threads = []
  for i in range(NUM_THREADS):
    scene, task = branches[i%NUM_TASKS]
    training_thread = A3C_Thread(i, global_network, initial_learning_rate,
                                        learning_rate_input,
                                        grad_applier, MAX_TIME_STEP,
                                        device = device,
                                        network_scope = "thread-%d"%(i+1),
                                        scene_scope = scene,
                                        task_scope = task)
    training_threads.append(training_thread)

  # Run tf network
  sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                          allow_soft_placement=True))

  init = tf.global_variables_initializer()
  sess.run(init)

  def create_summary():
    summary_op = dict()
    summary_placeholders = dict()
    for i in range(NUM_THREADS):
        scene, task = branches[i%NUM_TASKS]
        key = scene + "-" + task

        # summary for tensorboard
        episode_reward_input = tf.placeholder("float")
        episode_length_input = tf.placeholder("float")
        episode_max_q_input  = tf.placeholder("float")

        scalar_summaries = [
        tf.summary.scalar(key+"/Episode Reward", episode_reward_input),
        tf.summary.scalar(key+"/Episode Length", episode_length_input),
        tf.summary.scalar(key+"/Episode Max Q", episode_max_q_input)
        ]

        summary_op[key] = tf.summary.merge(scalar_summaries)
        summary_placeholders[key] = {
        "episode_reward_input": episode_reward_input,
        "episode_length_input": episode_length_input,
        "episode_max_q_input": episode_max_q_input,
        "learning_rate_input": learning_rate_input
        }
    return summary_op, summary_placeholders

  def train_function(parallel_index):
    global global_t
    training_thread = training_threads[parallel_index]
    last_global_t = 0

    scene, task = branches[parallel_index % NUM_TASKS]
    key = scene + "-" + task

    while global_t < MAX_TIME_STEP and not stop_requested:
      diff_global_t = training_thread.process(sess, global_t, summary_writer,
                                              summary_op[key], summary_placeholders[key])
      global_t += diff_global_t
      # periodically save checkpoints to disk
      if parallel_index == 0 and global_t - last_global_t > 1000000:
        print('Save checkpoint at timestamp %d' % global_t)
        saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)
        last_global_t = global_t

  def signal_handler(signal, frame):
    global stop_requested
    print('You pressed Ctrl+C!')
    stop_requested = True
  # create tensorboard summaries
  summary_op, summary_placeholders = create_summary()
  summary_writer = tf.summary.FileWriter(LOG_FILE, sess.graph)

  # init or load checkpoint with saver
  # if you don't need to be able to resume training, use the next line instead.
  # it will result in a much smaller checkpoint file.
  saver = tf.train.Saver(max_to_keep=10, var_list=global_network.get_vars())
  # saver = tf.train.Saver(max_to_keep=10)

  checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
  if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print("checkpoint loaded: {}".format(checkpoint.model_checkpoint_path))
    tokens = checkpoint.model_checkpoint_path.split("-")
    # set global step
    global_t = int(tokens[1])
    print(">>> global step set: {}".format(global_t))
  else:
    print("Could not find old checkpoint")
  train_threads = []
  for i in range(NUM_THREADS):
    train_threads.append(threading.Thread(target= train_function, args=(i,)))

  signal.signal(signal.SIGINT, signal_handler)

  # start each training thread
  for t in train_threads:
    t.start()

  print('Press Ctrl+C to stop.')
  signal.pause()

  # wait for all threads to finish
  for t in train_threads:
    t.join()

  print('Now saving data. Please wait.')
  saver.save(sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = global_t)
  summary_writer.close()

  
