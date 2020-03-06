from agents import *
from networks import *
import ray

if __name__ == '__main__':
    env_name = 'CartPole-v0'

    runner = A3CRunner(env_name, threads=1, episodes=500, entropy_weight=1e-4,
                       learning_rate=1e-3, discount_factor=0.99)

    ray.init(num_cpus=1)
    runner.train()
    print(runner.test())
