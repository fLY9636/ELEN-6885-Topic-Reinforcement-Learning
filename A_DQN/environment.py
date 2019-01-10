# This file load the scene dumps and construct grid world environment according to scences.

import numpy as np
import random
import h5py

from constants import ACTION_SIZE
from constants import REWARD_COLLIDE
from constants import REWARD_MOVE
from constants import REWARD_TERMINAL
from constants import HISTORY_LENGTH

class GridWorldEnvironment(object):

  def __init__(self, config=dict()):

    # Load scene and read h5 file
    self.scene_name = config.get('scene_name', 'bedroom_04')
    self.file_path = config.get('h5_file_path', 'data/%s.h5'%self.scene_name)
    self.h5_file = h5py.File(self.file_path, 'r')
    self.locations   = self.h5_file['location'][()]
    self.rotations   = self.h5_file['rotation'][()]
    self.id_terminal_state = config.get('id_terminal_state', 0)
    
    # Set terminals
    self.n_locations = self.locations.shape[0]
    self.terminals = np.zeros(self.n_locations)
    self.terminals[self.id_terminal_state] = 1
    self.terminal_states, = np.where(self.terminals)

    # Transition graph
    self.transition_graph = self.h5_file['graph'][()]
    self.shortest_path_distances = self.h5_file['shortest_path_distance'][()]

    
    self.random_start        = config.get('random_start', True)
    self.n_feat_per_locaiton = config.get('n_feat_per_locaiton', 1) # 1 for no sampling

    # Set initial state, with 4 history length
    self.history_length = HISTORY_LENGTH
    self.s_t      = np.zeros([2048, self.history_length])
    self.s_target = self._initialize_state(self.id_terminal_state)

    self.reset()

  # reset environment
  # The method is called when the class object is created
  def reset(self):
    # randomize initial state
    while True:
      k = random.randrange(self.n_locations)
      min_d = np.inf
      # check if target is reachable
      for t_state in self.terminal_states:
        dist = self.shortest_path_distances[k][t_state]
        min_d = min(min_d, dist)
      if min_d > 0: break

    # reset parameters
    self.current_state_id = k
    self.s_t = self._initialize_state(self.current_state_id)
    self.reward   = 0
    self.collided = False
    self.terminal = False

  # Step according to action and update s_t
  def step(self, action):
    assert not self.terminal, 'arrived terminal, can not move '
    k = self.current_state_id
    # Judge if the agent will collide 
    if self.transition_graph[k][action] == -1:
      self.collided = True
      self.terminal = False
    else:
      self.collided = False
      self.current_state_id = self.transition_graph[k][action]
      # Update s_t
      self.s_t = np.append(self.s_t[:,1:], self.state, axis=1)

      if self.terminals[self.current_state_id]:
        self.terminal = True
      else:
        self.terminal = False

    self.reward = self._reward(self.terminal, self.collided)


  # create initial state with 4 repeats of the same feature
  def _initialize_state(self, state_id):
    k = random.randrange(self.n_feat_per_locaiton)
    f = self.h5_file['resnet_feature'][state_id][k][:,np.newaxis]
    return np.tile(f, (1, self.history_length))

  # return the reward of the action
  def _reward(self, terminal, collided):
    if terminal: return REWARD_TERMINAL
    return REWARD_COLLIDE if collided else REWARD_MOVE


  # properties: return the information of current state, including locations, view...
  # coordinates
  @property
  def x(self):
    return self.locations[self.current_state_id][0]

  @property
  def y(self):
    return self.locations[self.current_state_id][1]

  @property
  def r(self):
    return self.rotations[self.current_state_id]

    # target
  @property
  def target(self):
    return self.s_target

  # action
  @property
  def action_size(self):
    return ACTION_SIZE 
  @property
  def action_definitions(self):
    action_vocab = ["MoveForward", "RotateRight", "RotateLeft", "MoveBackward"]
    return action_vocab[:ACTION_SIZE]
  
  # image view
  @property
  def observation(self):
    return self.h5_file['observation'][self.current_state_id]

  # current feature
  @property
  def state(self):
    # read from hdf5 cache
    k = random.randrange(self.n_feat_per_locaiton)
    return self.h5_file['resnet_feature'][self.current_state_id][k][:,np.newaxis]
  

  

if __name__ == "__main__":
  scene_name = 'bedroom_04'

  env = GridWorldEnvironment({
    'random_start': True,
    'scene_name': scene_name,
    'h5_file_path': 'data/%s.h5'%scene_name
  })
