

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
    

    
  # Construct networks
  def _network(self):
    # Define the names of domains
    network_scope = self._network_scope
    scene_scopes = self._scene_scopes
    action_size = self._action_size

    # Create dictionary for each variable
    self.pi = dict()
    self.v = dict()

    self.W_fc1 = dict()
    self.b_fc1 = dict()

    self.W_fc2 = dict()
    self.b_fc2 = dict()

    self.W_fc3 = dict()
    self.b_fc3 = dict()

    self.W_policy = dict()
    self.b_policy = dict()

    self.W_value = dict()
    self.b_value = dict()

    with tf.device(self._device):

      # state (input)
      self.s = tf.placeholder("float", [None, 2048, 4])

      # target (input)
      self.t = tf.placeholder("float", [None, 2048, 4])

      with tf.variable_scope(network_scope):
        # network key
        key = network_scope

        # flatten input
        self.s_flat = tf.reshape(self.s, [-1, 8192])
        self.t_flat = tf.reshape(self.t, [-1, 8192])

        # shared siamese layer
        self.W_fc1[key] = self._fc_weight_variable([8192, 512])
        self.b_fc1[key] = self._fc_bias_variable([512], 8192)

        h_s_flat = tf.nn.relu(tf.matmul(self.s_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_t_flat = tf.nn.relu(tf.matmul(self.t_flat, self.W_fc1[key]) + self.b_fc1[key])
        h_fc1 = tf.concat(values=[h_s_flat, h_t_flat], axis=1)

        # shared fusion layer
        self.W_fc2[key] = self._fc_weight_variable([1024, 512])
        self.b_fc2[key] = self._fc_bias_variable([512], 1024)
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, self.W_fc2[key]) + self.b_fc2[key])

        for scene_scope in scene_scopes:
          # scene-specific key
          key = self._get_key([network_scope, scene_scope])

          with tf.variable_scope(scene_scope):

            # scene-specific adaptation layer
            self.W_fc3[key] = self._fc_weight_variable([512, 512])
            self.b_fc3[key] = self._fc_bias_variable([512], 512)
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2, self.W_fc3[key]) + self.b_fc3[key])

            # weight for policy output layer
            self.W_policy[key] = self._fc_weight_variable([512, action_size])
            self.b_policy[key] = self._fc_bias_variable([action_size], 512)

            # policy (output)
            pi_ = tf.matmul(h_fc3, self.W_policy[key]) + self.b_policy[key]
            self.pi[key] = tf.nn.softmax(pi_)

            # weight for value output layer
            self.W_value[key] = self._fc_weight_variable([512, 1])
            self.b_value[key] = self._fc_bias_variable([1], 512)

            # value (output)
            v_ = tf.matmul(h_fc3, self.W_value[key]) + self.b_value[key]
            self.v[key] = tf.reshape(v_, [-1])


  # Compute the loss funtion in tf operation
  def prepare_loss(self, entropy_beta, scopes):
    scope_key = self._get_key(scopes[:-1])

    with tf.device(self._device):
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
      self.total_loss = policy_loss + value_loss                        # gradienet of policy and value are summed up


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
  def run_policy_and_value(self, sess, state, target, scopes):
    k = self._get_key(scopes[:2])
    pi_out, v_out = sess.run( [self.pi[k], self.v[k]], feed_dict = {self.s : [state], self.t: [target]} )
    return (pi_out[0], v_out[0])

  def run_policy(self, sess, state, target, scopes):
    k = self._get_key(scopes[:2])
    pi_out = sess.run( self.pi[k], feed_dict = {self.s : [state], self.t: [target]} )
    return pi_out[0]

  def run_value(self, sess, state, target, scopes):
    k = self._get_key(scopes[:2])
    v_out = sess.run( self.v[k], feed_dict = {self.s : [state], self.t: [target]} )
    return v_out[0]
  

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
  def _local_var_name(self, var):
    return '/'.join(var.name.split('/')[1:])

  def get_vars(self):
    var_list = [
      self.W_fc1, self.b_fc1,
      self.W_fc2, self.b_fc2,
      self.W_fc3, self.b_fc3,
      self.W_policy, self.b_policy,
      self.W_value, self.b_value
    ]
    vs = []
    for v in var_list:
      vs.extend(v.values())
    return vs
