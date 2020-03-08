import gym
import ray
import threading
import os
import numpy as np

from networks import Actor, Critic, ActorCriticModel, ActorLoss


class Storage:
    def __init__(self, env: gym.Env):
        self.states = np.empty((0,) + env.observation_space.shape)
        self.actions = np.array([])
        self.rewards = np.array([])
        self.values = np.array([])

    def append(self, state, action, reward, value):
        self.states = np.append(self.states, state[None, :], axis=0)
        self.actions = np.append(self.actions, action)
        self.rewards = np.append(self.rewards, reward)
        self.values = np.append(self.values, value)


@ray.remote
class SingleAgent(object):
    def __init__(self,
                 env_name: str,
                 save_dir: str,
                 entropy_weight: float,
                 discount_factor: float,
                 id: int):
        import tensorflow as tf
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.id = id

        self.save_dir = save_dir

        self.discount_factor = discount_factor

        actor = Actor(action_space_size=self.env.action_space.n)
        critic = Critic()
        self.local_model = ActorCriticModel(actor, critic)

        self.actor_loss = ActorLoss(entropy_weight)

        self.local_model(tf.convert_to_tensor(np.random.random((1, self.env.observation_space.shape[0]))))

    def run(self, global_weights, batch_size=64):
        import tensorflow as tf
        self.local_model.set_weights(global_weights)
        actions = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size,) + self.env.observation_space.shape)
        next_state = self.env.reset()

        for step in range(batch_size):
            observations[step] = next_state.copy()
            actions[step], values[step] = self.local_model.get_action(next_state[None, :])
            next_state, rewards[step], dones[step], _ = self.env.step(actions[step])

            if dones[step]:
                next_state = self.env.reset()

        _, next_value = self.local_model.get_action(next_state[None, :])

        returns = np.append(np.ones_like(rewards), next_value, axis=-1)
        returns = self.discount_returns(returns, rewards, dones)

        advantages = returns - values

        actions_and_advantages = np.concatenate([actions[:, None], advantages[:, None]], axis=-1)
        observations = tf.convert_to_tensor(observations)

        with tf.GradientTape() as tape:
            tape.watch(observations)
            logits, pred_values = self.local_model(observations)
            loss = self.actor_loss(actions_and_advantages, logits)
            loss += 0.5 * tf.keras.losses.MeanSquaredError()(returns, pred_values)

        grads = tape.gradient(loss, self.local_model.trainable_weights)
        return grads, self.id

    def discount_returns(self, returns, rewards, dones):
        for i in reversed(range(len(rewards))):
            returns[i] *= self.discount_factor * returns[i+1] * (1 - dones[i])
            returns[i] += rewards[i]
        returns = returns[:-1]

        return returns
