import tensorflow as tf
import numpy as np

class CategoricalSampler(tf.keras.Model):
  def call(self, logits):
    return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

class Actor(tf.keras.Model):
    def __init__(self,
                 action_space_size: int,
                 name: str = None):
        super(Actor, self).__init__(name=name)
        self.hidden_1 = tf.keras.layers.Dense(128, activation='relu')
        self.hidden_2 = tf.keras.layers.Dense(128, activation='relu')

        self.logits = tf.keras.layers.Dense(action_space_size)


    def call(self, inputs):
        x = self.hidden_1(inputs)
        x = self.hidden_2(x)
        return self.logits(x)


class Critic(tf.keras.Sequential):
    def __init__(self,
                 name: str = None):
        hidden_1 = tf.keras.layers.Dense(128, activation='relu')
        hidden_2 = tf.keras.layers.Dense(128, activation='relu')
        state_out = tf.keras.layers.Dense(1)
        super(Critic, self).__init__([hidden_1, hidden_2, state_out], name)


class ActorCriticModel(tf.keras.Model):
    def __init__(self,
                 actor_network: tf.keras.Model,
                 critic_network: tf.keras.Model,
                 name: str = None):
        super(ActorCriticModel, self).__init__(name=name)

        self.actor_network = actor_network
        self.critic_network = critic_network
        self.sampler = CategoricalSampler()

    def call(self, inputs):
        logits = self.actor_network(inputs)
        value = self.critic_network(inputs)
        return logits, value

    def get_action(self, inputs):
        logits, value = self.predict_on_batch(inputs)
        action = self.sampler.predict_on_batch(logits)

        return np.squeeze(action, axis=-1), np.squeeze(value, axis=-1)
