import tensorflow as tf
import logging
import numpy as np
import gym
import threading
from typing import Union
from scipy.signal import lfilter
from networks import ActorCriticModel

EPISODES = 5000

class ActorLoss(tf.keras.losses.Loss):
    def __init__(self,
                 entropy_weight: float,
                 name: str = None):
        super(ActorLoss, self).__init__(name=name)
        self.entropy_weight = entropy_weight

    def call(self, y_true, y_pred):
        actions, advantages = tf.split(y_true, 2, axis=-1)
        actions = tf.cast(actions, tf.int32)

        policy_loss_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        policy_loss = policy_loss_ce(actions, y_pred, sample_weight=advantages)

        pmf = tf.keras.activations.softmax(y_pred)
        entropy_loss = tf.keras.losses.categorical_crossentropy(pmf, pmf)

        return policy_loss - self.entropy_weight * entropy_loss


class SingleAgent: #(threading.Thread):
    def __init__(self,
                 model: ActorCriticModel,
                 env: str,
                 learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule],
                 entropy_weight: float,
                 discount_factor: float):
        #super(SingleAgent, self).__init__()

        self.states, self.rewards, self.actions = [], [], []

        self.env = env

        self.discount_factor = discount_factor

        self.actor_loss = ActorLoss(entropy_weight)
        self.model = model
        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                                   loss=[self.actor_loss, tf.keras.losses.MeanSquaredError()])

    def test(self, render=True):
        obs, done, episode_reward = self.env.reset(), False, 0
        while not done:
            action, _ = self.model.get_action(obs[None, :])
            obs, reward, done, _ = self.env.step(action)
            episode_reward += reward
            if render:
                self.env.render()
        return episode_reward

    def train(self, batch_size=64, episodes=100):
        actions = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size,) + self.env.observation_space.shape)

        episode_rewards = [0.0]
        next_state = self.env.reset()
        for episode in range(episodes):
            for step in range(batch_size):
                observations[step] = next_state.copy()
                actions[step], values[step] = self.model.get_action(next_state[None, :])
                next_obs, rewards[step], dones[step], _ = self.env.step(actions[step])

                episode_rewards[-1] += rewards[step]
                if dones[step]:
                    episode_rewards.append(0.0)
                    next_state = self.env.reset()
                    print(episode_rewards[-2])

            _, next_value = self.model.get_action(next_state[None, :])

            returns = np.append(1 - dones, next_value)
            for i in reversed(range(len(rewards))):
                returns[i] *= self.discount_factor * returns[i+1]
                returns[i] += rewards[i]

            returns = returns[:-1]
            advantages = returns - values

            actions_and_advantages = np.concatenate([actions[:, None], advantages[:, None]], axis=-1)

            losses = self.model.train_on_batch(observations, [actions_and_advantages, returns])

        return episode_rewards
