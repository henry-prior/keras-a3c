from networks import Actor, Critic, ActorCriticModel, ActorLoss
import gym
import threading
import os
import numpy as np
import tensorflow as tf
from queue import Queue

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


class SingleAgent(threading.Thread):
    # Easier to set up global variables here as opposed to in the run function
    episode = 0
    best_score = 0
    lock = threading.Lock()
    def __init__(self,
                 global_model: ActorCriticModel,
                 env_name: str,
                 queue: Queue,
                 episodes: int,
                 entropy_weight: float,
                 learning_rate: float,
                 discount_factor: float):
        super(SingleAgent, self).__init__()

        self.env_name = env_name
        self.env = gym.make(env_name)

        self.save_dir = os.path.expanduser('~/keras-a3c/models/')

        self.queue = queue

        self.EPISODES = episodes
        self.discount_factor = discount_factor
        self.global_model = global_model

        actor = Actor(action_space_size=self.env.action_space.n)
        critic = Critic()
        self.local_model = ActorCriticModel(actor, critic)

        self.actor_loss = ActorLoss(entropy_weight)

        self.local_model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                                  loss=[self.actor_loss, tf.keras.losses.MeanSquaredError()],
                                  loss_weights = [1., 0.5])

    def run(self, batch_size=64):
        storage = Storage(self.env)
        while SingleAgent.episode < self.EPISODES:
            print(SingleAgent.episode)
            next_state = self.env.reset()
            batch = 0
            score = 0
            done = False

            while not done:
                state = next_state.copy()
                action, value = self.local_model.get_action(next_state[None, :])
                next_state, reward, done, _ = self.env.step(action)

                score += reward

                storage.append(state, action, reward, value)
                if batch == batch_size or done:
                    _, next_value = self.local_model.get_action(next_state[None, :])

                    returns = np.append(np.zeros_like(storage.values), next_value)
                    returns = self.discount_returns(returns, storage.rewards)
                    advantages = returns - storage.values

                    actions_and_advantages = np.concatenate([storage.actions[:, None], advantages[:, None]], axis=-1)

                    with tf.GradientTape() as tape:
                        logits, value = self.local_model(storage.states)
                        loss = self.actor_loss(actions_and_advantages, logits) + 0.5 * tf.keras.losses.MeanSquaredError()(returns, value)
                        #loss = self.local_model.loss(storage.states, [actions_and_advantages, returns])

                    grads = tape.gradient(loss, self.local_model.trainable_weights)
                    self.global_model.optimizer.apply_gradients(zip(grads, self.global_model.trainable_weights))
                    self.local_model.set_weights(self.global_model.get_weights())

                    storage.__init__(self.env)
                    batch = 0

                    if done and score > SingleAgent.best_score:
                        with SingleAgent.lock:
                            self.global_model.save_weights(os.path.join(self.save_dir,
                                 'model_{}_{}.h5'.format(self.env_name, int(score))))
                            SingleAgent.best_score = score
                        print(SingleAgent.episode, score)
                    SingleAgent.episode += 1

                batch += 1
            self.queue.put(None)

    def discount_returns(self, returns, rewards):
        for i in reversed(range(len(rewards))):
            returns[i] *= self.discount_factor * returns[i+1]
            returns[i] += rewards[i]
        returns = returns[:-1]

        return returns
