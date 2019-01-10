#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import threading
import numpy as np
import signal
import random

from A3C_Thread import A3C_Thread
from network import SingleDSNetwork
from utils.rmsprop_applier import RMSPropApplier
from evaluate import Evaluate

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

class Train(object):
    def __init__(self):
        if not os.path.exists(CHECKPOINT_DIR):
            os.mkdir(CHECKPOINT_DIR)

        self.device = "/gpu:0" if USE_GPU else "/cpu:0"
        self.network_scope = TASK_TYPE
        self.list_of_tasks = TASK_LIST
        self.scene_scopes = self.list_of_tasks.keys()
        self.global_t = 0
        self.stop_requested = False

        

        self.initial_learning_rate = self.log_uniform(LR_ALPHA_LOW,
                                            LR_ALPHA_HIGH,
                                            LR_ALPHA_LOG_RATE)

        self.global_network = SingleDSNetwork(action_size = ACTION_SIZE,
                                                device = self.device,
                                                network_scope = self.network_scope)

        self.branches = []
        for scene in self.scene_scopes:
            for task in self.list_of_tasks[scene]:
                self.branches.append((scene, task))

        self.NUM_TASKS = len(self.branches)
        assert NUM_THREADS >= self.NUM_TASKS, \
            "Not enough threads for multitasking: at least {} threads needed.".format(self.NUM_TASKS)

        self.learning_rate_input = tf.placeholder("float")
        self.grad_applier = RMSPropApplier(learning_rate = self.learning_rate_input,
                                        decay = RMSP_ALPHA,
                                        momentum = 0.0,
                                        epsilon = RMSP_EPSILON,
                                        clip_norm = GRAD_NORM_CLIP,
                                        device = self.device)

        # instantiate each training thread
        # each thread is training for one target in one scene
        self.training_threads = []
        for i in range(NUM_THREADS):
            scene, task = self.branches[i%self.NUM_TASKS]
            training_thread = A3C_Thread(i, self.global_network, self.initial_learning_rate,
                                                self.learning_rate_input,
                                                self.grad_applier, MAX_TIME_STEP,
                                                device = self.device,
                                                network_scope = "thread-%d"%(i+1),
                                                scene_scope = scene,
                                                task_scope = task)
            self.training_threads.append(training_thread)
    def log_uniform(self, lo, hi, rate):
        log_lo = np.log(lo)
        log_hi = np.log(hi)
        v = log_lo * (1-rate) + log_hi * rate
        return np.exp(v)
    
    def train(self):
        # prepare session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,
                                                allow_soft_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

        # create tensorboard summaries
        self.create_summary()
        self.summary_writer = tf.summary.FileWriter(LOG_FILE, self.sess.graph)

        # init or load checkpoint with saver
        # if you don't need to be able to resume training, use the next line instead.
        # it will result in a much smaller checkpoint file.
        # self.saver = tf.train.Saver(max_to_keep=10, var_list=self.global_network.get_vars())
        self.saver = tf.train.Saver(max_to_keep=10)

        self.checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if self.checkpoint and self.checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, self.checkpoint.model_checkpoint_path)
            print("checkpoint loaded: {}".format(self.checkpoint.model_checkpoint_path))
            tokens = self.checkpoint.model_checkpoint_path.split("-")
            # set global step
            self.global_t = int(tokens[1])
            print(">>> global step set: {}".format(self.global_t))
        else:
            print("Could not find old checkpoint")

        train_threads = []
        for i in range(NUM_THREADS):
            train_threads.append(threading.Thread(target=self.train_function, args=(i,)))

        signal.signal(signal.SIGINT, self.signal_handler)

        # start each training thread
        for t in train_threads:
            t.start()

        print('Press Ctrl+C to stop.')
        signal.pause()

        # wait for all threads to finish
        for t in train_threads:
            t.join()

        print('Now saving data. Please wait.')
        self.saver.save(self.sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = self.global_t)
        self.summary_writer.close()

    def create_summary(self):
        self.summary_op = dict()
        self.summary_placeholders = dict()
        for i in range(NUM_THREADS):
            scene, task = self.branches[i%self.NUM_TASKS]
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

            self.summary_op[key] = tf.summary.merge(scalar_summaries)
            self.summary_placeholders[key] = {
            "episode_reward_input": episode_reward_input,
            "episode_length_input": episode_length_input,
            "episode_max_q_input": episode_max_q_input,
            "learning_rate_input": self.learning_rate_input
            }
    def train_function(self, parallel_index):
        training_thread = self.training_threads[parallel_index]
        last_global_t = 0

        scene, task = self.branches[parallel_index % self.NUM_TASKS]
        key = scene + "-" + task
        while self.global_t < MAX_TIME_STEP and not self.stop_requested:
            diff_global_t = training_thread.process(self.sess, self.global_t, self.summary_writer,
                                                self.summary_op[key], self.summary_placeholders[key])
            self.global_t += diff_global_t
            # periodically save checkpoints to disk
            if parallel_index == 0 and self.global_t - last_global_t > 1000000:
                print('Save checkpoint at timestamp %d' % self.global_t)
                self.saver.save(self.sess, CHECKPOINT_DIR + '/' + 'checkpoint', global_step = self.global_t)
                last_global_t = self.global_t
                

    def signal_handler(self, signal, frame):
        print('You pressed Ctrl+C!')
        self.stop_requested = True

    

if __name__ == '__main__':

    machine = Train()
    machine.train()
