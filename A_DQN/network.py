

import tensorflow as tf
import numpy as np

from constants import ACTION_SIZE

# Deep Reinforcement learning network
class DRLNetwork(object):
  """
    Implementation of the target-driven deep siamese actor-critic network from [Zhu et al., ICRA 2017]
    We use tf.variable_scope() to define domains for parameter sharing
  """
  # -*- Construction of networks -*-
  def __init__(self,
               action_size = ACTION_SIZE,
               device="/cpu:0",
               network_scope="network",
               scene_scopes=["scene"]):
    self._device = device
    self._action_size = action_size
    self._network_scope = network_scope
    self._scene_scopes = scene_scopes
    self._network()

    t_params = tf.get_collection('target_net_params')
    e_params = tf.get_collection('eval_net_params')
    self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    
  # Construct networks
  def _network(self):
    # Define the names of domains
    network_scope = self._network_scope
    scene_scopes = self._scene_scopes
    action_size = self._action_size

    # Create dictionary for each variable
    # eval network
    self.W_fc1 = dict()
    self.b_fc1 = dict()

    self.W_fc2 = dict()
    self.b_fc2 = dict()

    self.w1 = dict()
    self.b1 = dict()

    self.w2 = dict()
    self.b2 = dict()

    self.q_eval = dict()

    # target network
    self.W_fc1_ = dict()
    self.b_fc1_ = dict()

    self.W_fc2_ = dict()
    self.b_fc2_ = dict()

    self.W_fc3_ = dict()
    self.b_fc3_ = dict()

    self.w1_ = dict()
    self.b1_ = dict()

    self.w2_ = dict()
    self.b2_ = dict()

    self.q_next = dict()

    with tf.device(self._device):

      # state (input)
      self.s = tf.placeholder("float", [None, 2048, 4])

      # target (input)
      self.t = tf.placeholder("float", [None, 2048, 4])

      # next state (input)
      self.s_ = tf.placeholder("float", [None, 2048, 4])

      with tf.variable_scope(network_scope):
        # network key
        key = network_scope

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
          key_eval = key + '/eval'

          # flatten input
          self.s_flat = tf.reshape(self.s, [-1, 8192])
          self.t_flat = tf.reshape(self.t, [-1, 8192])

          # shared siamese layer
          self.W_fc1[key_eval] = self._fc_weight_variable([8192, 512])
          self.b_fc1[key_eval] = self._fc_bias_variable([512], 8192)

          h_s_flat = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1[key_eval]) + self.b_fc1[key_eval])
          h_t_flat = tf.nn.relu(tf.matmul(self.t_flat, self.W_fc1[key_eval]) + self.b_fc1[key_eval])
          h_fc1 = tf.concat(values=[h_s_flat, h_t_flat], axis=1)

          # shared fusion layer
          self.W_fc2[key_eval] = self._fc_weight_variable([1024, 512])
          self.b_fc2[key_eval] = self._fc_bias_variable([512], 1024)
          h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2[key_eval]) + self.b_fc2[key_eval])

          for scene_scope in scene_scopes:
            # scene-specific key
            key_eval = self._get_key([network_scope, scene_scope]) + '/eval'

            with tf.variable_scope(scene_scope):
              self.q_target = tf.placeholder(tf.float32, [None, action_size], name='Q_target')

              c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                          tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

              with tf.variable_scope('l1'):
                self.w1[key_eval] = tf.get_variable('w1', [512, n_l1], initializer=w_initializer, collections=c_names)
                self.b1[key_eval] = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(h_fc2, self.w1[key_eval]) + self.b1[key_eval])

              with tf.variable_scope('l2'):
                self.w2[key_eval] = tf.get_variable('w2', [n_l1, action_size], initializer=w_initializer, collections=c_names)
                self.b2[key_eval] = tf.get_variable('b2', [1, action_size], initializer=b_initializer, collections=c_names)
                self.q_eval[key_eval] = tf.matmul(l1, self.w2[key_eval]) + self.b2[key_eval]


        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
          key_target = key + '/target'

          # flatten input
          self.s_flat_ = tf.reshape(self.s_, [-1, 8192])
          self.t_flat_ = tf.reshape(self.t, [-1, 8192])

          # shared siamese layer
          self.W_fc1_[key_target] = self._fc_weight_variable([8192, 512])
          self.b_fc1_[key_target] = self._fc_bias_variable([512], 8192)

          h_s_flat_ = tf.nn.relu(tf.matmul(self.s_flat_, self.W_fc1_[key_target]) + self.b_fc1_[key_target])
          h_t_flat_ = tf.nn.relu(tf.matmul(self.t_flat_, self.W_fc1_[key_target]) + self.b_fc1_[key_target])
          h_fc1_ = tf.concat(values=[h_s_flat_, h_t_flat_], axis=1)

          # shared fusion layer
          self.W_fc2_[key_target] = self._fc_weight_variable([1024, 512])
          self.b_fc2_[key_target] = self._fc_bias_variable([512], 1024)
          h_fc2_ = tf.nn.relu(tf.matmul(h_fc1_, self.W_fc2_[key_target]) + self.b_fc2_[key_target])

          for scene_scope in scene_scopes:
            # scene-specific key
            key_target = self._get_key([network_scope, scene_scope]) + '/target'

            with tf.variable_scope(scene_scope):

              c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

              with tf.variable_scope('l1'):
                self.w1_[key_target] = tf.get_variable('w1', [512, n_l1], initializer=w_initializer, collections=c_names)
                self.b1_[key_target] = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1_ = tf.nn.relu(tf.matmul(h_fc2_, self.w1_[key_target]) + self.b1_[key_target])

              with tf.variable_scope('l2'):
                self.w2_[key_target] = tf.get_variable('w2', [n_l1, action_size], initializer=w_initializer, collections=c_names)
                self.b2_[key_target] = tf.get_variable('b2', [1, action_size], initializer=b_initializer, collections=c_names)
                self.q_next[key_target] = tf.matmul(l1_, self.w2_[key_target]) + self.b2_[key_target]






  # Compute the loss funtion in tf operation
  def prepare_loss(self, entropy_beta, scopes):
    k_scopes = scopes[:2]
    k_scopes.append('eval')
    k = self._get_key(k_scopes)

    with tf.device(self._device):
      """
      # input for loss function
      self.a = tf.placeholder("float", [None, self._action_size])       # action, input for policy
      self.td = tf.placeholder("float", [None])                         # temporary difference, input for policy
      self.r = tf.placeholder("float", [None])                          # R (input for value)
      # calculation of loss function 
      log_pi = tf.log(tf.clip_by_value(self.pi[scope_key], 1e-20, 1.0)) # log pi
      entropy = -tf.reduce_sum(self.pi[scope_key] * log_pi, axis=1)     # entropy of policy
      policy_loss = - tf.reduce_sum(tf.reduce_sum(tf.multiply(log_pi, self.a), axis=1) * self.td + entropy * entropy_beta) # policy loss (output)
      value_loss = 0.5 * tf.nn.l2_loss(self.r - self.v[scope_key])      # value loss (output) # learning rate for critic is half of actor's
      # Total loss
      self.total_loss = policy_loss + value_loss  
      """
      self.total_loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval[k]))


  # -*- Initialization of variables -*-
  def _fc_weight_variable(self, shape, name='W_fc'):
    input_channels = shape[0]
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _fc_bias_variable(self, shape, input_channels, name='b_fc'):
    d = 1.0 / np.sqrt(input_channels)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_weight_variable(self, shape, name='W_conv'):
    w = shape[0]
    h = shape[1]
    input_channels = shape[2]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv_bias_variable(self, shape, w, h, input_channels, name='b_conv'):
    d = 1.0 / np.sqrt(input_channels * w * h)
    initial = tf.random_uniform(shape, minval=-d, maxval=d)
    return tf.Variable(initial, name=name)

  def _conv2d(self, x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

  def _get_key(self, scopes):
    return '/'.join(scopes)


  # -*- Run tf operations  -*-
  def run_DQN(self, sess, state, target, scopes):
    k_scopes = scopes[:2]
    k_scopes.append('eval')
    k = self._get_key(k_scopes)
    actions_value = sess.run(self.q_eval[k], feed_dict={self.s: [state], self.t: [target]})
    return actions_value



  # -*- Components for A3C -*-
  # Get the shared parameters from global network
  def sync_from(self, src_netowrk, name=None):
    src_vars = src_netowrk.get_vars()
    dst_vars = self.get_vars()

    local_src_var_names = [self._local_var_name(x) for x in src_vars]
    local_dst_var_names = [self._local_var_name(x) for x in dst_vars]

    # keep only variables from both src and dst
    src_vars = [x for x in src_vars
      if self._local_var_name(x) in local_dst_var_names]
    dst_vars = [x for x in dst_vars
      if self._local_var_name(x) in local_src_var_names]

    sync_ops = []

    with tf.device(self._device):
      with tf.name_scope(name, "ActorCriticNetwork", []) as name:
        for(src_var, dst_var) in zip(src_vars, dst_vars):
          sync_op = tf.assign(dst_var, src_var)
          sync_ops.append(sync_op)

        return tf.group(*sync_ops, name=name)

  # get local variable name
  # remove network scope from the name
  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])

  def get_vars(self):
    var_list = [
      self.W_fc1, self.b_fc1,
      self.W_fc2, self.b_fc2,

      self.w1, self.b1,
      self.w2, self.b2
    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs
