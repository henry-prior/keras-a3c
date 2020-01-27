from agents import *
from networks import *

if __name__ == '__main__':
    #env = gym.make('CartPole-v0')

    #actor = Actor(action_space_size=env.action_space.n)
    #critic = Critic()
    #model = ActorCriticModel(actor, critic)

    #agent = SingleAgent(model, learning_rate=1e-3, entropy_weight=1e-4, discount_factor=0.99)

    #rewards_history = agent.train(env)

    #print("%d out of 200" % agent.test(env))

    env_name = 'CartPole-v0'

    runner = A3CRunner(env_name, threads=8, entropy_weight=1e-4,
                       learning_rate=1e-3, discount_factor=0.99)

    runner.train()
