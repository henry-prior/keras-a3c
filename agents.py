import tensorflow as tf
import logging
import numpy as np
import gym
import threading
from queue import Queue
from typing import Union
from scipy.signal import lfilter
from networks import Actor, Critic, ActorCriticModel, ActorLoss
from single_agent import SingleAgent

class A3CRunner:
    def __init__(self,
                 env_name: str,
                 threads: int,
                 episodes: int,
                 entropy_weight: float,
                 learning_rate: Union[float, tf.keras.optimizers.schedules.LearningRateSchedule],
                 discount_factor: float):

        self.env_name = env_name
        env = gym.make(env_name)

        self.threads = threads
        self.EPISODES = episodes
        self.entropy_weight = entropy_weight
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.queue = Queue()

        actor = Actor(action_space_size=env.action_space.n)
        critic = Critic()
        self.global_model = ActorCriticModel(actor, critic)

        self.actor_loss = ActorLoss(entropy_weight)

        self.global_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                                  loss=[self.actor_loss, tf.keras.losses.MeanSquaredError()],
                                  loss_weights = [1., 0.5])

        self.global_model(tf.convert_to_tensor(np.random.random((1, env.observation_space.shape[0]))))

    def test(self, render=True):
        env = gym.make(self.env_name)
        obs, done, episode_reward = env.reset(), False, 0
        while not done:
            action, _ = self.global_model.get_action(obs[None, :])
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
        return episode_reward

    def train(self):
        queue = Queue()

        agents = [SingleAgent(self.global_model, self.env_name, queue, self.EPISODES,
        self.entropy_weight, self.learning_rate, self.discount_factor) for _ in range(self.threads)]

        for agent in agents:
            agent.start()

        moving_average_rewards = []  # record episode reward to plot
        while True:
            reward = queue.get()
            if reward is not None:
                moving_average_rewards.append(reward)
            else:
                break

        [a.join() for a in agents]
