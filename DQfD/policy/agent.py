import random
import timeit
from collections import deque
import pathlib
import numpy as np
import torch
import torch.nn.functional as F
from policy.models import get_network_builder
from policy.replay_buffer import AggregatedBuff
from utils.config_validation import AgentCfg, Task
from utils.discretization import get_dtype_dict
import json

TRAIN_POLICY_MODEL_NAME = 'policy_pretrain_treechop_wtraining.pth'
TRAIN_TARGET_MODEL_NAME = 'network_pretrain_treechop_wtraining.pth'

PRE_TRAIN_ALL_LOSS_PATH = 'loss/TREECHOP_PRE_TRAIN_AVG_TOTAL_LOSE_100000STEP_2.json'
PRE_TRAIN_ALL_TD_PATH = 'loss/TREECHOP_PRE_TRAIN_TD_LOSE_100000STEP_2.json'
PRE_TRAIN_ALL_NTD_PATH = 'loss/TREECHOP_PRE_TRAIN_NTD_LOSE_100000STEP_2.json'
PRE_TRAIN_ALL_EXPERT_PATH = 'loss/TREECHOP_PRE_TRAIN_EXPERT_LOSE_100000STEP_2.json'

TRAIN_ALL_LOSS_PATH = 'loss/TREECHOP_TRAIN_AVG_TOTAL_LOSE_100000STEP_2.json'
TRAIN_ALL_TD_PATH = 'loss/TREECHOP_TRAIN_TD_LOSE_100000STEP_2.json'
TRAIN_ALL_NTD_PATH = 'loss/TREECHOP_TRAIN_NTD_LOSE_100000STEP_2.json'
TRAIN_ALL_EXPERT_PATH = 'loss/TREECHOP_TRAIN_EXPERT_LOSE_100000STEP_2.json'

SCORE_PATH = 'score/TREECHOP_TRAIN_SCORE_100000STEP_2.json'
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def create_flat_agent(task: Task, env):
    make_model = get_network_builder(task.model_name)
    env_dict, dtype_dict = get_dtype_dict(env)
    # print( 'env dict ', env_dict )
    replay_buffer = AggregatedBuff(env_dict, task.cfg.buffer)
    agent = Agent(task.cfg.agent, replay_buffer, make_model, env.observation_space, env.action_space, device)
    if not task.from_scratch:
        agent.load(task.cfg.agent.save_dir)
    return agent


class Agent:
    def __init__(self, cfg, replay_buffer, build_model, obs_space, act_space, device,
                 log_freq=100):

        self.cfg = cfg
        self.n_deque = deque([], maxlen=cfg.n_step)

        self.replay_buff = replay_buffer
        self.priorities_store = []
        self.sampler = self.sample_generator  # PyTorch doesn't use tf.data.Dataset
        self.device = device
        # Build models
        self.online_model = build_model('Online_Model', obs_space, act_space, self.cfg.l2)
        self.target_model = build_model('Target_Model', obs_space, act_space, self.cfg.l2)
        self.target_model.load_state_dict(self.online_model.state_dict())
        self.target_model.eval()

        self.online_model.to( self.device )
        self.target_model.to( self.device )

        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.cfg.learning_rate)

        # Logging & scheduling (matching structure, not active yet)
        self._run_time_deque = deque(maxlen=log_freq)
        self._schedule_dict = {
            self.target_update: self.cfg.update_target_net_mod,
            self.update_log: log_freq
        }
        self.avg_metrics = {}

        self.action_dim = act_space.n
        self.global_step = 0


    def train(self, env, task):
        print('starting from step:', self.global_step)
        scores = []
        epsilon = self.cfg.initial_epsilon
        current_episode = 0
        print( ' task.max_train_steps :',  task.max_train_steps, 'task.max_train_episodes', task.max_train_episodes )

        while self.global_step < task.max_train_steps and current_episode < task.max_train_episodes:
            score = self.train_episode(env, task, epsilon)
            print(f'Steps: {self.global_step}, Episode: {current_episode}, Reward: {score}, Eps Greedy: {round(epsilon, 3)}')
            current_episode += 1

            if self.global_step >= self.cfg.epsilon_time_steps:
                epsilon = self.cfg.final_epsilon
            else:
                epsilon = (self.cfg.initial_epsilon - self.cfg.final_epsilon) * \
                          (self.cfg.epsilon_time_steps - self.global_step) / self.cfg.epsilon_time_steps

            self.save_json( [ score ],     SCORE_PATH )
            scores.append(score)

        return scores
    
    def train_episode(self, env, task: Task, epsilon=0.0):
        if self.global_step == 0:
            self.target_update()
        done, score, state = False, 0, env.reset()
        while not done:
            action = self.choose_act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            if task.cfg.wrappers.render:
                env.render()
            score += reward

            self.global_step += 1
            if not task.evaluation:
                self.perceive(to_demo=0, state=state, action=action, reward=reward, next_state=next_state,
                              done=done, demo=False)
                
                if self.replay_buff.get_stored_size() > self.cfg.replay_start_size:
                    if self.global_step % self.cfg.frames_to_update == 0:
                        self.update(task.cfg.agent.update_quantity)
                        print( 'self.global_step :', self.global_step )
                        self.save(task.cfg.agent.save_dir)
                        print(f'saving to {task.cfg.agent.save_dir}')

            state = next_state
        return score
    
    def pre_train(self, task):
        """
        pre_train phase in policy alg.
        :return:
        """
        print('Pre-training ...')
        self.target_update()
        self.update(task.pretrain_num_updates, preTrain = True)
        # self.save(os.path.join(self.cfg.save_dir, "pre_trained_model.ckpt"))
        print('All pre-train finish.')

        # torch.save(self.online_model.state_dict(), TRAIN_POLICY_MODEL_NAME)
        # torch.save(self.target_model.state_dict(), TRAIN_TARGET_MODEL_NAME)

    def update(self, num_updates, preTrain = False):
        all_losses_list = list()
        mean_td_list = list()
        mean_ntd_list = list()
        l2_list = list()
        margin_list = list()
        
        step_all_losses_list = list()
        step_mean_td_list = list()
        step_mean_ntd_list = list()
        step_margin_list = list()

        for i in range(num_updates):
            batch = self.replay_buff.sample(self.cfg.batch_size)
            indexes = batch.pop('indexes')
            priorities, all_losses, mean_td, mean_ntd, l2, margin = self.q_network_update(gamma=self.cfg.gamma, batch=batch)
            all_losses_list.append( all_losses.item() )
            mean_td_list.append( mean_td.item() )
            mean_ntd_list.append(  mean_ntd.item() )
            margin_list.append( margin.item() )

            if i % 100 == 0:
                mean_loss = sum(all_losses_list) / len(all_losses_list)
                mean_tds = sum(mean_td_list)/ len(mean_td_list)
                mean_ntds = sum(mean_ntd_list) / len(mean_ntd_list)
                mean_margin = sum(margin_list)/ len(margin_list)
                
                print( '---------------------------------------------------' )
                print( "iteration : ", i )
                print( " mean total loss : {:<10.3f}, mean td lose : {:<10.3f}, mean ntd lose : {:<10.3f}, mean expert lose : {:<10.3f}".format( mean_loss, mean_tds,mean_ntds, mean_margin ) )
                # tqdm.write("Iteration {}. Total Loss {:<10.3f} TDLoss  {:<10.3f} Expert Loss {:<10.3f} nStep Loss {:<10.3f}".format(iter_count, mean_loss, mean_td, mean_expert, mean_nStep))
                

                step_all_losses_list.append( mean_loss )
                step_mean_td_list.append( mean_tds )
                step_mean_ntd_list.append( mean_ntds )
                step_margin_list.append( mean_margin )
                
                all_losses_list.clear()
                mean_td_list.clear()
                mean_ntd_list.clear()
                mean_ntd_list.clear()

            self.priorities_store.append({'indexes': indexes, 'priorities': priorities})
        
        if preTrain:
            torch.save(self.online_model.state_dict(), TRAIN_POLICY_MODEL_NAME)
            torch.save(self.target_model.state_dict(), TRAIN_TARGET_MODEL_NAME)
            
            self.save_json( step_all_losses_list, PRE_TRAIN_ALL_LOSS_PATH )
            self.save_json( step_mean_td_list,    PRE_TRAIN_ALL_TD_PATH )
            self.save_json( step_mean_ntd_list,   PRE_TRAIN_ALL_NTD_PATH )
            self.save_json( step_margin_list,     PRE_TRAIN_ALL_EXPERT_PATH )
        else:
            self.save_json( step_all_losses_list, TRAIN_ALL_LOSS_PATH )
            self.save_json( step_mean_td_list,    TRAIN_ALL_TD_PATH )
            self.save_json( step_mean_ntd_list,   TRAIN_ALL_NTD_PATH )
            self.save_json( step_margin_list,     TRAIN_ALL_EXPERT_PATH )

        while len( self.priorities_store ) > 0:
            priorities = self.priorities_store.pop(0)
            self.replay_buff.update_priorities(**priorities)

    def save_json( self, list, filePath ):
        try:
            with open(filePath, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.extend(list)

        with open(filePath, "w") as f:
            json.dump(data, f, indent=4)

    def sample_generator(self, steps=None):
        steps_done = 0
        finite_loop = bool(steps)
        steps = steps if finite_loop else 1
        while steps_done < steps:
            yield self.replay_buff.sample(self.cfg.batch_size)
            if len(self.priorities_store) > 0:
                priorities = self.priorities_store.pop(0)
                self.replay_buff.update_priorities(**priorities)
            steps += int(finite_loop)
    
    def huber_loss(self, x, delta=1.0):
        """
        Compute the Huber loss in PyTorch.
        
        Args:
            x (Tensor): The input tensor (TD errors).
            delta (float, optional): The delta for the Huber loss. Default is 1.0.
            
        Returns:
            Tensor: The computed Huber loss.
        """
        return torch.where(torch.abs(x) < delta,
                        0.5 * x ** 2,
                        delta * (torch.abs(x) - 0.5 * delta))

    def q_network_update(self, gamma, batch):
        """
        Update the Q-network using the given batch of data.

        Args:
            batch (dict): A dictionary containing the following keys:
                - 'state': The state of the environment at the current time step.
                - 'action': The actions taken by the agent.
                - 'reward': The rewards received from the environment.
                - 'next_state': The state of the environment at the next time step.
                - 'done': A flag indicating whether the episode is done.
                - 'demo': Whether the data comes from an expert demo.
                - 'n_state': The next state, used for n-step learning.
                - 'n_reward': The reward for the n-step learning.
                - 'n_done': Flag indicating whether the next state is terminal for n-step learning.
                - 'actual_n': The actual number of steps in the n-step reward.
                - 'weights': Weights used for weighted updates.
            gamma (float): The discount factor for future rewards.

        Returns:
            priorities (Tensor): A tensor of priorities for the sampled batch, used for updating the replay buffer.
        """

        # Extract data from the batch dictionary
        state_data = torch.tensor(batch['state'], dtype=torch.float32).to(self.device)
        action_data = torch.tensor(batch['action'], dtype=torch.long).to(self.device)
        reward_data = torch.tensor(batch['reward'], dtype=torch.float32).to(self.device)
        next_state_data = torch.tensor(batch['next_state'], dtype=torch.float32).to(self.device)
        done_data = torch.tensor(batch['done'], dtype=torch.float32).to(self.device)
        demo_data = torch.tensor(batch['demo'], dtype=torch.float32).to(self.device)
        n_state_data = torch.tensor(batch['n_state'], dtype=torch.float32).to(self.device)
        n_reward_data = torch.tensor(batch['n_reward'], dtype=torch.float32).to(self.device)
        n_done_data = torch.tensor(batch['n_done'], dtype=torch.float32).to(self.device)
        actual_n_data = torch.tensor(batch['actual_n'], dtype=torch.float32).to(self.device)
        weights_data = torch.tensor(batch['weights'], dtype=torch.float32).to(self.device)

        # Now proceed with your logic using the extracted tensors
        # Example: 
        q_value = self.online_model(state_data)
        margin = self.margin_loss(q_value, action_data, demo_data, weights_data)
        self.update_metrics('margin', margin)

        q_value = q_value.gather(1, action_data.unsqueeze(1))

        td_loss = self.td_loss(q_value, next_state_data, done_data, reward_data, 1, gamma)
        huber_td = self.huber_loss(td_loss, delta=0.4)
        mean_td = torch.mean(huber_td * weights_data)
        self.update_metrics('TD', mean_td)

        ntd_loss = self.td_loss(q_value, n_state_data, n_done_data, n_reward_data, actual_n_data, gamma)
        huber_ntd = self.huber_loss(ntd_loss, delta=0.4)
        mean_ntd = torch.mean(huber_ntd * weights_data)
        self.update_metrics('nTD', mean_ntd)
        
        # l2_lambda = 1e-4
        # l2_reg = torch.tensor(0.0, device= self.device)
        # for param in self.parameters():
        #     l2_reg += torch.norm(param, p=2)  # L2 norm
        l2 = 0
        # l2 = sum(self.online_model.losses)
        self.update_metrics('l2', l2)

        all_losses = mean_td + mean_ntd + l2 + margin
        self.update_metrics('all_losses', all_losses)

        # Apply gradients
        self.optimizer.zero_grad()
        all_losses.backward()
        self.optimizer.step()

        priorities = torch.abs(td_loss)
        return priorities, all_losses, mean_td, mean_ntd, l2, margin


    def td_loss(self, q_value, next_state, done, reward, actual_n, gamma):
        with torch.no_grad():
            q_next = self.online_model( next_state)
            next_actions = q_next.argmax(dim=1, keepdim=True)
            q_target = self.target_model(next_state)
            target = q_target.gather(1, next_actions).squeeze(1)
            target = torch.where(done.bool(), torch.zeros_like(target), target)
            if actual_n is None:
                actual_n = torch.ones_like(target)
            return q_value - (reward + (gamma ** actual_n) * target)


    def compute_target(self, next_state, done, reward, actual_n, gamma):
        with torch.no_grad():
            q_next = self.online_model({'features': next_state})
            next_actions = q_next.argmax(dim=1, keepdim=True)
            q_target = self.target_model({'features': next_state})
            target = q_target.gather(1, next_actions).squeeze(1)
            target = torch.where(done, torch.zeros_like(target), target)
            return reward + (gamma ** actual_n) * target

    def margin_loss(self, q_value, action, demo, weights):
        one_hot = F.one_hot(action, self.action_dim).float().to(self.device)
        ae = (1 - one_hot) * self.cfg.margin
        max_margin_q = torch.max(q_value + ae, dim=1).values
        selected_q = torch.sum(q_value * one_hot, dim=1)
        j_e = torch.abs(selected_q - max_margin_q)
        return torch.mean(j_e * weights * demo)


    def add_demo(self, expert_env, expert_data=1):
        while not expert_env.are_all_frames_used():
            done = False
            obs = expert_env.reset()

            while not done:
                next_obs, reward, done, info = expert_env.step(0)
                action = info['expert_action']
                self.perceive(to_demo=1, state=obs, action=action, reward=reward, next_state=next_obs, done=done,
                              demo=expert_data)
                obs = next_obs

    def perceive(self, **kwargs):
        self.n_deque.append(kwargs)

        if len(self.n_deque) == self.n_deque.maxlen or kwargs['done']:
            while len(self.n_deque) != 0:
                n_state = self.n_deque[-1]['next_state']
                n_done = self.n_deque[-1]['done']
                n_reward = sum([t['reward'] * self.cfg.gamma ** i for i, t in enumerate(self.n_deque)])
                self.n_deque[0]['n_state'] = n_state
                self.n_deque[0]['n_reward'] = n_reward
                self.n_deque[0]['n_done'] = n_done
                self.n_deque[0]['actual_n'] = len(self.n_deque)
                self.replay_buff.add(**self.n_deque.popleft())
                if not n_done:
                    break

    def choose_act(self, state, epsilon=0.01):
        device = next(self.online_model.parameters()).device
        nn_input = np.array(state)[None]
        nn_input = torch.from_numpy(nn_input)  # or use torch.from_numpy(nn_input)
        nn_input = nn_input.to(device)
        # q_value = self.online_model(nn_input, training=False)
        self.online_model.train()
        q_value = self.online_model(nn_input)
        if random.random() <= epsilon:
            return random.randint(0, self.action_dim - 1)
        # return np.argmax(q_value)
        return np.argmax(q_value.cpu().detach().numpy())

    def schedule(self):
        for key, value in self._schedule_dict.items():
            if self.global_step % value == 0:
                key()

    def target_update(self):
        self.target_model.load_state_dict(self.online_model.state_dict())

    # def save(self, out_dir=None):
    #     self.online_model.save_weights(pathlib.Path(out_dir) / 'model.ckpt')
    def save(self, out_dir=None):
        # Ensure the output directory exists
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save the model's state_dict
        torch.save(self.online_model.state_dict(), out_dir / 'model.pth')

    # def load(self, out_dir=None):

    #     if pathlib.Path(out_dir).exists():
    #         self.online_model.load_weights(pathlib.Path(out_dir) / 'model.pth')
    #     else:
    #         raise KeyError(f"Can not import weights from {pathlib.Path(out_dir)}")

    def load(self, out_dir=None):
        model_path = pathlib.Path(out_dir) / 'model.pth'
                # Check if the model file exists
        if model_path.exists():
            # Load the model's state_dict from the saved file
            self.online_model.load_state_dict(torch.load(model_path))
        else:
            raise KeyError(f"Cannot import weights from {model_path}")

    def update_log(self):
        pass

    def update_metrics(self, key, value):
        pass
