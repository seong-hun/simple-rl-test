import numpy as np
import sys
from ray.rllib.agents import ppo

from main import MyEnv


def test(path):
    agent = ppo.PPOTrainer(config=config, env=MyEnv)
    agent.restore(path)


def main():
    path = sys.argv[1]
    test(path)


if __name__ == "__main__":
    main()
