import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.initializers import RandomUniform
import numpy as np


ENABLE_ACTIONS = [1,2,3,4,5]
N_ACTIONS = len(ENABLE_ACTIONS)

class ActorNetwork(tf.keras.Model):

    ACTION_RANGE = 2.0

    def __init__(self, action_space):

        super(ActorNetwork, self).__init__()

        self.action_space = action_space

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

        self.dense1 = kl.Dense(20, activation="relu")
        #print(self.dense1)

        #self.bn1 = kl.BatchNormalization()

        self.dense2 = kl.Dense(20, activation="relu")

        #self.bn2 = kl.BatchNormalization()

        self.actions = kl.Dense(self.action_space, activation="linear")

    def call(self, state, training=True):
        
        #print(state)
        #x = np.reshape(state, 20)
        #print(x)

        x = self.dense1(state)
        #print(state)
        #print(x)

        #x = self.bn1(x, training=training)

        x = self.dense2(x)

        #x = self.bn2(x, training=training)

        actions = self.actions(x)

        #actions = actions * self.ACTION_RANGE

        return actions

    def sample_action(self, state, noise=None):
        """ノイズつきアクションのサンプリング
        """

        self.enable_actions = ENABLE_ACTIONS
        
        state1 = np.atleast_2d(state).astype(np.float32)
        #print(state)

        action = self(state1, training=False).numpy()[0]
        #print(action)

        self.action1 = np.clip(action, 0, 6).astype(np.float32)
        #print(self.action1)

        self.action1 = self.enable_actions[np.argmax(self.call(state1))]

        
        
        #print(state)
        #print(action)
        return self.action1


class CriticNetwork(tf.keras.Model):

    def __init__(self):

        super(CriticNetwork, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

        self.dense1 = kl.Dense(20, activation="relu")

        self.bn1 = kl.BatchNormalization()

        self.dense2 = kl.Dense(20, activation="relu")

        self.bn2 = kl.BatchNormalization()

        self.values = kl.Dense(1)

    def call(self, state, action1, training=True):

        #print(state)
        #print(action1)

        #y = np.reshape(state, 20)
        #print(y)

        a = tf.concat([action1], 1)

        x = tf.concat([state, a], 1)
        #print(x)

        x = self.dense1(x)

        #x = self.bn1(x, training=training)

        x = self.dense2(x)

        #x = self.bn2(x, training=training)
        #print(x)
        values = self.values(x)

        #print(values)
        return values