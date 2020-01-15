import tensorflow as tf

class Actor(tf.keras.Model):
    def __init__(self,
                 state: tf.keras.layers.Layer,
                 action_space_size: int,
                 name: str = None):
        super(Actor, self).__init__(name=name)
        self.state_in = state
        self.hidden_1 = tf.keras.layers.Dense(24, activation='relu')
        self.hidden_2 = tf.keras.layers.Dense(24, activation='relu')

        self.mu_in = tf.keras.layers.Dense(action_space_size, activation='tanh')
        self.mu = tf.keras.layers.Lambda(lambda x: x * 2)

        self.sigma_in = tf.keras.layers.Dense(action_space_size, activation='softplus')
        self.sigma = tf.keras.layers.Lambda(lambda x: x + 0.0001)

    def call(self, inputs):
        x = self.state_in(inputs)
        x = self.hidden_1(x)
        x = self.hidden_2(x)

        mu_in = self.mu_in(x)
        sigma_in = self.sigma_in(x)

        return self.mu(mu_in), self.sigma(sigma_in)

class Critic(tf.keras.Sequential):
    def __init__(self,
                 state: tf.keras.layers.Layer,
                 action_space_size: int,
                 name: str = None):
        hidden_1 = tf.keras.layers.Dense(24, activation='relu')
        hidden_2 = tf.keras.layers.Dense(24, activation='relu', kernel_initializer='he_uniform')
        state_out = tf.keras.layers.Dense(1, kernel_initializer='he_uniform')
        super(Critic, self).__init__([state, hidden_1, hidden_2, state_out], name)

