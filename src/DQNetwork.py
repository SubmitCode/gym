import tensorflow as tf
import numpy as np


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        
        
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions")
            
            
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")
            
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                          filters = 32,
                                          kernel_size = [8,8],
                                          strides = [4,4],
                                          padding = "VALID",
                                          kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                          name = "conv1")
            
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            
            
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                          filters = 64,
                                          kernel_size = [4,4],
                                          strides = [2,2],
                                          padding = "VALID",
                                          kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                          name = "conv2")
            
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")
            
            
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                          filters = 64,
                                          kernel_size = [3,3],
                                          strides = [2,2],
                                          padding = "VALID",
                                          kernel_initializer = tf.contrib.layers.xavier_initializer_conv2d(),
                                          name = "conv3")
            
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            
            self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            
            self.fc = tf.layers.dense(inputs = self.flatten,
                                     units = 512,
                                     activation = tf.nn.elu,
                                     kernel_initializer = tf.contrib.layers.xavier_initializer(),
                                     name = "fc1")
            
            self.output = tf.layers.dense(inputs = self.fc,
                                        units = self.action_size,
                                        activation = None)
            
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_))
            
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))
            
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss) 

    def predict_action(self, explore_start, explore_stop, decay_rate, decay_step, state, actions, possible_actions):
        ## EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        ## First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
        
        if (explore_probability > exp_exp_tradeoff):
            # Make a random action (exploration)
            choice = np.random.randint(possible_actions)-1
            action = possible_actions[choice]
            
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = tf.sess.run(self.output, feed_dict = {self.inputs_: state.reshape((1, *state.shape))})
            
            # Take the biggest Q value (= the best action)
            choice = np.argmax(Qs)
            action = possible_actions[choice]
                    
                    
        return action, explore_probability