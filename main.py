import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from pathlib import Path

import gym
import ray
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print

import fym

import test_exercises


class Plant(fym.BaseEnv):
    def __init__(self):
        super().__init__()
        self.pos = fym.BaseSystem(shape=(2, 1))
        self.vel = fym.BaseSystem(shape=(2, 1))


class MyEnv(fym.BaseEnv, gym.Env):
    def __init__(self, env_config):
        super().__init__(**env_config)
        self.plant = Plant()

        # self.action_space = <gym.Space>
        self.action_space = gym.spaces.Box(
            low=-10, high=10, shape=(2,))
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.plant.state.shape)

    def reset(self, initial="random"):
        if initial == "random":
            self.plant.initial_state = 5 * (
                2 * np.random.rand(*self.plant.state.shape) - 1)
        else:
            self.plant.initial_state = initial
        super().reset()
        return self.plant.state

    def step(self, action):
        x = np.float32(self.plant.state)
        u = np.vstack(action)
        *_, done = self.update(u=u)
        next_obs = np.float32(self.plant.state)
        reward = np.float32(
            np.exp(
                1e-6 * (
                    - x.T @ np.diag([100, 100, 1, 1]) @ x
                    - u.T @ np.diag([10, 10]) @ u
                ).item()
            )
        )
        return next_obs, reward, done, {}

    def set_dot(self, t, u):
        pos, vel = self.plant.observe_list()
        self.plant.pos.dot = vel
        self.plant.vel.dot = u
        return dict(t=t, pos=pos, vel=vel, u=u)


def train(path=None):
    cfg = fym.config.load(as_dict=True)

    ray.init(ignore_reinit_error=True, log_to_driver=False)

    analysis = ray.tune.run(ppo.PPOTrainer, **cfg)

    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial(
            "episode_reward_mean",
            mode="max",
        ),
        metric="episode_reward_mean",
    )
    checkpoint_path = checkpoints[0][0]

    print(f"checkpoint path: {checkpoint_path}")
    return checkpoint_path


def test(path):
    fym.config.update({
        "config.env_config.max_t": 50,
    })
    env_config = fym.config.load("config.env_config", as_dict=True)

    agent = ppo.PPOTrainer(env=MyEnv, config={"explore": False})
    agent.restore(path)

    env = MyEnv(env_config)
    env.logger = fym.Logger("data.h5")

    # run until episode ends
    done = False
    obs = env.reset()
    while not done:
        env.render()
        action = agent.compute_single_action(obs)
        obs, reward, done, info = env.step(action)

    env.close()


def plot(path):
    data = fym.load(path)

    ax = plt.subplot(311)
    plt.plot(data["t"], data["pos"].squeeze())
    plt.ylabel("pos")

    plt.subplot(312, sharex=ax)
    plt.plot(data["t"], data["vel"].squeeze())
    plt.ylabel("vel")

    plt.subplot(313, sharex=ax)
    plt.plot(data["t"], data["u"].squeeze())
    plt.ylabel("action")
    plt.xlabel("time")

    plt.show()


@ray.remote(num_cpus=12)
def single_run(initial, train_path, exp_path):
    fym.config.update({
        "config.env_config.max_t": 20,
    })
    env_config = fym.config.load("config.env_config", as_dict=True)

    agent = ppo.PPOTrainer(env=MyEnv, config={"explore": False})
    agent.restore(str(train_path))

    env = MyEnv(env_config)

    run_name = "_".join(["run"] + [f"{x:05.2f}" for x in initial]) + ".h5"
    run_path = exp_path / train_path.name / run_name
    env.logger = fym.Logger(run_path)

    # run until episode ends
    done = False
    obs = env.reset(initial[:, None])
    total_reward = 0.

    while not done:
        action = agent.compute_single_action(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    env.close()
    data = fym.load(run_path)

    ckpt_num = int(train_path.parent.name.split("_")[1])
    index = data["t"] % 0.1 == 0
    return np.hstack((
        data["t"][index, None],
        data["pos"][index].squeeze(),
        data["vel"][index].squeeze(),
        data["u"][index].squeeze(),
        np.tile([ckpt_num, total_reward, run_path], reps=(np.sum(index), 1)),
    ))


def mc_test(path):
    ray.init(ignore_reinit_error=True, log_to_driver=False)
    assert ray.is_initialized() is True

    exp_path = Path("data")
    exp_path.mkdir(exist_ok=True)

    ckpts = Path(path).glob("checkpoint*")

    # Random initials
    initials = 3 * (2 * np.random.rand(100, 4) - 1)
    initials = initials[
        np.all([
            np.sqrt(np.sum(initials[:, :2]**2, axis=1)) < 3,
            np.sqrt(np.sum(initials[:, 2:]**2, axis=1)) < 3,
        ], axis=0)
    ]
    assert len(initials) > 0

    print("Initials were generated")

    df = pd.DataFrame(columns=[
        "t", "x", "y", "vx", "vy", "ux", "uy", "ckpt", "return", "file_index"])

    for ckpt in ckpts:
        train_path = list(filter(lambda x: "." not in x.name, ckpt.iterdir()))[0]

        print(f"Checkpoint {ckpt} ...")

        dfs = ray.get([
            single_run.remote(x, train_path, exp_path) for x in initials])

        for new_df in dfs:
            df = df.append(pd.DataFrame(new_df, columns=df.columns))

    df.to_csv(exp_path / "out.csv", index=False)

    print("DataFrame was saved into csv file.")

    ray.shutdown()
    assert ray.is_initialized() is not True


def main():
    fym.config.update({
        "config": {
            "env": MyEnv,
            "env_config": {
                "dt": 0.05,
                "max_t": 20,
                "solver": "odeint"
            },
            "num_gpus": 0,
            "num_workers": 12,
            "lr": 0.001,
            "gamma": 0.999,
            # "lr": ray.tune.grid_search([0.001, 0.0001]),
            # "gamma": ray.tune.grid_search([0.99, 0.999, 0.9999]),
        },
        "stop": {
            "training_iteration": 500,
        },
        "local_dir": "./ray-results",
        "checkpoint_freq": 30,
        "checkpoint_at_end": True,
    })

    if len(sys.argv) == 1:
        # path = train()
        pass
    else:
        path = sys.argv[1]

    # test(path)
    # plot("data.h5")

    mc_test(path)


if __name__ == "__main__":
    main()
