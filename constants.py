# -*- coding: utf-8 -*-


EPSILON = 0.1 # epsilon parameter
CHECKPOINT_DIR = 'checkpoints'
LOG_FILE = 'logs'
INITIAL_ALPHA = 0.1
INITIAL_GAMMA = 1
#REWARD = -1
NUM_TRA_EPISODES = 15000
NUM_TRA_EPISODES_DQN = 2000

PARALLEL_SIZE = 20 # parallel thread size
ACTION_SIZE = 4 # action size


ENTROPY_BETA = 0.01 # entropy regurarlization constant
MAX_TIME_STEP = 10.0 * 10**6 # 10 million frames
GRAD_NORM_CLIP = 40.0 # gradient norm clipping
USE_GPU = True # To use GPU, set True
VERBOSE = True

SCREEN_WIDTH = 84
SCREEN_HEIGHT = 84
HISTORY_LENGTH = 4

NUM_EVAL_EPISODES = 100 # number of episodes for evaluation

TASK_TYPE = 'navigation' # no need to change
# keys are scene names, and values are a list of location ids (navigation targets)
TASK_LIST = {
  'bathroom_02'    : ['26']
}
