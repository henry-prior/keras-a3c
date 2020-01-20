import tensorflow as tf

class CategoricalSample(tf.keras.Model):
  def call(self, logits):
    return tf.squeeze(tf.random.categorical(logits, 1), axis=-1)

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
                 critic_network: tf.keras.Model):
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.sampler = CategoricalSampler()

    def call(self, inputs):
        logits = self.actor_network(inputs)
        value = self.critic_network(inputs)
        return logits, value

    def get_action(self, inputs):
        # Action selection needs to be outside of the call() function because we don't do backprop through it
        # predict_on_batch is faster than call in TF 2.0
        logits, value = self.predict_on_batch(inputs)
        action = self.sampler.predict_on_batch(inputs)

        return action, value

