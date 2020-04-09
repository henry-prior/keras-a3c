import tensorflow as tf
import numpy as np

class CategoricalSampler(tf.keras.Model):
    def call(self, logits):
      return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)


class ActorLoss(tf.keras.losses.Loss):
    def __init__(self,
                 entropy_weight: float,
                 name: str = None):
        super(ActorLoss, self).__init__(name=name)
        self.entropy_weight = entropy_weight

    def call(self, y_true, y_pred):
        actions, advantages = tf.split(y_true, 2, axis=-1)
        actions = tf.cast(actions, tf.int32)
        advantages = tf.keras.backend.stop_gradient(advantages)

        policy_loss_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = policy_loss_ce(actions, y_pred, sample_weight=advantages)

        pmf = tf.keras.activations.softmax(y_pred)
        entropy_loss = tf.keras.losses.categorical_crossentropy(pmf, pmf)

        return policy_loss - self.entropy_weight * entropy_loss


class Actor(tf.keras.Model):
    def __init__(self,
                 action_space_size: int,
                 name: str = None):
        super(Actor, self).__init__(name=name)
        self.hidden_1 = tf.keras.layers.Dense(64, activation='relu')
        self.hidden_2 = tf.keras.layers.Dense(64, activation='relu')

        self.logits = tf.keras.layers.Dense(action_space_size)


    def call(self, inputs):
        x = self.hidden_1(inputs)
        x = self.hidden_2(x)
        return self.logits(x)


class Critic(tf.keras.Sequential):
    def __init__(self,
                 name: str = None):
        hidden_1 = tf.keras.layers.Dense(64, activation='relu')
        hidden_2 = tf.keras.layers.Dense(64, activation='relu')
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
        inputs = tf.convert_to_tensor(inputs)

        logits = self.actor_network(inputs)
        value = self.critic_network(inputs)

        return logits, value

    def get_action(self, inputs):
        logits, value = self.predict(inputs)
        action = self.sampler.predict(logits)

        return action.item(), np.squeeze(value, axis=-1)
