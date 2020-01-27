import tensorflow as tf
import logging
import numpy as np
import gym
import threading
from typing import Union
from scipy.signal import lfilter
from networks import Actor, Critic, ActorCriticModel

episode = 0
EPISODES = 8

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


class A3CRunner:
    def __init__(self,
                 env_name: str,
                 threads: int,
                 entropy_weight: float,
                 learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule],
                 discount_factor: float):

        self.env = gym.make(env_name)

        self.threads = threads
        self.discount_factor = discount_factor

        actor = Actor(action_space_size=self.env.action_space.n)
        critic = Critic()
        self.model = ActorCriticModel(actor, critic)

        self.actor_loss = ActorLoss(entropy_weight)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                           loss=[self.actor_loss, tf.keras.losses.MeanSquaredError()],
                           loss_weights = [1., 0.5])

    def test(self, render=True):
        obs, done, episode_reward = self.env.reset(), False, 0
        while not done:
            action, _ = self.model.get_action(obs[None, :])
            obs, reward, done, _ = self.env.step(action)
            episode_reward += reward
            if render:
                self.env.render()
        return episode_reward

    def train(self):
        agents = [SingleAgent(self.model, self.env, self.discount_factor)
                  for _ in range(self.threads)]

        for agent in agents:
            agent.start()

        self.test()

class SingleAgent(threading.Thread):
    def __init__(self,
                 model: ActorCriticModel,
                 env: gym.Env,
                 discount_factor: float):
        super(SingleAgent, self).__init__()

        self.env = env

        self.discount_factor = discount_factor
        self.model = model

    def run(self, batch_size=64):
        global episode
        actions = np.empty((batch_size,), dtype=np.int32)
        rewards, dones, values = np.empty((3, batch_size))
        observations = np.empty((batch_size,) + self.env.observation_space.shape)

        episode_rewards = [0.]
        next_state = self.env.reset()
        while episode < EPISODES:
            for step in range(batch_size):
                observations[step] = next_state.copy()
                actions[step], values[step] = self.model.get_action(next_state[None, :])
                next_state, rewards[step], dones[step], _ = self.env.step(actions[step])

                episode_rewards[-1] += rewards[step]
                if dones[step]:
                    episode_rewards.append(0.)
                    next_state = self.env.reset()

            episode += 1

            _, next_value = self.model.get_action(next_state[None, :])

            returns = np.append(1 - dones, next_value)
            returns = self.discount_returns(returns, rewards)
            advantages = returns - values

            actions_and_advantages = np.concatenate([actions[:, None], advantages[:, None]], axis=-1)

            self.model.train_on_batch(observations, [actions_and_advantages, returns])


    def discount_returns(self, returns, rewards):
        for i in reversed(range(len(rewards))):
            returns[i] *= self.discount_factor * returns[i+1]
            returns[i] += rewards[i]
        returns = returns[:-1]

        return returns
