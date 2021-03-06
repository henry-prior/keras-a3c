import tensorflow as tf
import logging
import ray
import numpy as np
import gym
import os
from typing import Union

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

        self.save_dir = os.path.expanduser('~/keras-a3c/models/')

        self.threads = threads
        self.EPISODES = episodes
        self.entropy_weight = entropy_weight
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        actor = Actor(action_space_size=env.action_space.n)
        critic = Critic()
        self.global_model = ActorCriticModel(actor, critic)

        self.actor_loss = ActorLoss(entropy_weight)

        self.optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)

        self.global_model(tf.convert_to_tensor(np.random.random((1, env.observation_space.shape[0]))))

    def test(self, render=True):
        env = gym.make(self.env_name)
        model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.env_name))
        self.global_model.load_weights(model_path)
        obs, done, episode_reward = env.reset(), False, 0
        while not done:
            action, _ = self.global_model.get_action(obs[None, :])
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
        return episode_reward

    def train(self):
        agents = [SingleAgent.remote(self.env_name, self.save_dir,
        self.entropy_weight, self.discount_factor, i) for i in range(self.threads)]

        parameters = self.global_model.get_weights()
        gradient_list = [agent.run.remote(parameters) for agent in agents]
        episode = 0

        while episode < self.EPISODES:
            done_id, gradient_list = ray.wait(gradient_list)
            episode += 1

            gradients, id = ray.get(done_id)[0]

            self.optimizer.apply_gradients(zip(gradients, self.global_model.trainable_weights))
            parameters = self.global_model.get_weights()
            gradient_list.extend([agents[id].run.remote(parameters)])

        model_path = os.path.join(self.save_dir, 'model_{}.h5'.format(self.env_name))
        self.global_model.save_weights(model_path)
