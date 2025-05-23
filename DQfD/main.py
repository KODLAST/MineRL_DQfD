from functools import reduce

import minerl
import gym
import os
import torch
import yaml

from policy.agent import create_flat_agent
from hierarchy.subtask_agent import ItemAgent
from hierarchy.subtasks_extraction import TrajectoryInformation
from utils.fake_env import FakeEnv

import argparse

from utils.config_validation import Pipeline, Task
from utils.wrappers import wrap_env
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# max_trj = 300
def load_trajectories( task, max_trj = 300):
    data = minerl.data.make(task.environment, task.data_dir)

    trajectories = []
    for trj_name in data.get_trajectory_names()[:max_trj]:
        trajectories.append(TrajectoryInformation(env_name=data.environment, trajectory_name=trj_name))

    return trajectories

def save_json( list, filePath ):
        try:
            with open(filePath, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.extend(list)

        with open(filePath, "w") as f:
            json.dump(data, f, indent=4)

def run_task(task: Task):
    if task.agent_type == 'flat':
        env = wrap_env(gym.make(task.environment), task.cfg.wrappers)
        agent = create_flat_agent(task, env)

        if task.source == 'expert':
            for trajectory in load_trajectories(task):
                # todo replace 'log' with parameter name
                agent.add_demo(wrap_env(FakeEnv(data=trajectory.trajectory_by_subtask['log']), task.cfg.wrappers))

            agent.pre_train(task)
            agent.save(task.cfg.agent.save_dir)

        elif task.source == 'agent':
            print( '-----------------agent learn-------------' )
            scores = agent.train(env, task)
            
            env.close()
            agent.save(task.cfg.agent.save_dir)

    elif task.agent_type == 'hierarchical':

        if task.source == 'expert':
            env = wrap_env(gym.make(task.environment), task.cfg.wrappers)
            trajectories = load_trajectories(task)
            unique_subtasks = reduce(lambda x, y: x.union(y),
                                     [set(q) for q in [t.trajectory_by_subtask.keys() for t in trajectories]])
            for subtask in unique_subtasks:
                if subtask not in ["cobblestone", "iron_ore"]:
                    continue
                agent = create_flat_agent(task, env)
                for trj in trajectories:
                    if not trj.trajectory_by_subtask.get(subtask, None):
                        continue

                    agent.add_demo(wrap_env(FakeEnv(data=trj.trajectory_by_subtask[subtask]), task.cfg.wrappers))
                agent.pre_train(task)
                agent.save(task.cfg.agent.save_dir + subtask + '/')

        elif task.source == 'agent':
            env = gym.make(task.environment)
            item_agent = ItemAgent(task)
            item_agent.train(env, task)


def run_pipeline(pipeline: Pipeline):
    for task in pipeline.pipeline:
        run_task(task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, action="store", help='yaml file with settings', required=False, default='configs/eval-diamond.yaml')
    params = parser.parse_args()

    with open(params.config, "r") as f:
        config = yaml.safe_load(f)
    run_pipeline(Pipeline(**config))



if __name__ == '__main__':
    main()
