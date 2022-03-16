#!/usr/bin/env python

from multiprocessing import cpu_count

import gym
import neat


def train_agent(genome, config, render=False):
    env = gym.make('Acrobot-v1')

    observation = env.reset()

    agent = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0

    for _ in range(1000):
        if render:
            env.render()

        action = agent.activate(observation)
        observation, reward, done, info = env.step(action.index(max(action)))

        fitness += reward

        if done:
            break

    return fitness


if __name__ == '__main__':
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config')

    # Create the population
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Train population for GENERATIONS generations
    GENERATIONS = 30
    training_environment = neat.ParallelEvaluator(cpu_count(), train_agent)
    winner = population.run(training_environment.evaluate, GENERATIONS)

    # nodeNames = {-1: 'Cart Position', -2: 'Cart Velocity', -3: 'Pole Angle', -4: 'Pole Angular Velocity', 0: 'right', 1: 'left'}
    # visualize.draw_net(config, winner, True, node_names=nodeNames)
    # visualize.draw_net(config, winner, True)
    # visualize.plot_stats(stats, ylog=False, view=True)
    # visualize.plot_species(stats, view=True)

    # Visualize the winner
    train_agent(winner, config, True)
