import numpy as np
np.random.seed(1000)
from copy import deepcopy
import csv
import Agent
import Plotter
from os import path, makedirs


### User specified parameters ###
import FrameEnv as env
# N_FEATURE = 100
# RECORD_INTERVAL = 10
MAX_STEPS = 10000
#################################


class Environment():
    def __init__(self, use_gpu: bool, mode: str, model_name: str, section_type: str, code_type: str, reward_type: str, loss_type: str, 
                 n_feature: int, capacity: int, batch_size: int, memorize_frequency: int, gamma: float, target_update_freq: int, record_interval: int, 
                 optimizer: str, epsilon_schedule: callable):
        
        # for Environment
        self.mode = mode
        self.model_name = model_name
        self.RECORD_INTERVAL = record_interval
        self.memorize_frequency = memorize_frequency
        self.epsilon_schedule = epsilon_schedule
        # for Agent
        self.loss_type = loss_type
        self.N_FEATURE = n_feature
        self.CAPACITY = capacity
        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.TARGET_UPDATE_FREQ = target_update_freq
        self.optimizer = optimizer

        self.env = env.Frame(self.mode, section_type, code_type, reward_type)
        v, w, _, infeasible_action = self.env.reset(test=True)
        self.agent = Agent.Agent(v.shape[1], w.shape[1], self.N_FEATURE, infeasible_action.shape[1], use_gpu, self.loss_type, self.CAPACITY, self.BATCH_SIZE, self.GAMMA, self.TARGET_UPDATE_FREQ, self.optimizer)
        #self.agent.brain.model.Load(filename="trained_model_{0}_{1}".format(env.__name__, self.mode), directory=f"src/{self.model_name}")
        if use_gpu:
            self.agent.brain.model = self.agent.brain.model.to("cuda")
        pass

    def Train(self, n_episode):
        history = np.zeros(n_episode//self.RECORD_INTERVAL, dtype=float)
        train_fail_reasons = []
        train_losses = np.zeros(n_episode, dtype=float)

        top_score = -np.inf
        top_scored_iteration = -1
        top_scored_model = deepcopy(self.agent.brain.model)
        test_fail_reasons = []

        for episode in range(n_episode):
            v, w, C, infeasible_action = self.env.reset()
            total_reward = 0.0
            aveQ = 0.0
            aveloss = 0.0
            epsilon = self.epsilon_schedule(episode)
            for t in range(MAX_STEPS):
                action, q = self.agent.get_action(v, w, C, epsilon, infeasible_action)
                aveQ += q
                v_next, w_next, reward, ep_end, fail_reason, infeasible_action = self.env.step(action)
                if t % self.memorize_frequency == 0:
                    self.agent.memorize(C, v, w, action, reward, v_next, w_next, ep_end, infeasible_action)
                v = np.copy(v_next)
                w = np.copy(w_next)
                aveloss += self.agent.update_q_function()
                total_reward += reward
                if ep_end:
                    break
            print("episode {0:<4}: step={1:<3} reward={2:<+5.1f} aveQ={3:<+7.2f} loss={4:<7.2f} fail_reason={5}".format(episode, t+1, total_reward, aveQ/(t+1), aveloss/(t+1), fail_reason))
            train_fail_reasons.append(fail_reason)
            train_losses[episode] = aveloss/(t+1)
            
            if episode % self.RECORD_INTERVAL == self.RECORD_INTERVAL-1:
                v, w, C, infeasible_action = self.env.reset(test=True)
                total_reward = 0.0
                for t in range(MAX_STEPS):
                    action, _ = self.agent.get_action(v, w, C, 0.0, infeasible_action)
                    v, w, reward, ep_end, fail_reason, infeasible_action = self.env.step(action)
                    total_reward += reward
                    if ep_end:
                        break
                if (total_reward >= top_score):
                    top_score = total_reward
                    top_scored_iteration = episode
                    top_scored_model = deepcopy(self.agent.brain.model)
                    
                history[episode//self.RECORD_INTERVAL] = total_reward
                test_fail_reasons.append(fail_reason)

        
        result_dir = f"result/{self.model_name}"
        if not path.exists(result_dir): makedirs(result_dir)

        with open(f"{result_dir}/info.txt", 'a') as f:
            f.write(str.format("top-scored iteration: {0} \n", top_scored_iteration+1))
            f.write(f"top-score: {top_score:.3f}\n")

        top_scored_model.Save(filename="trained_model_{0}_{1}".format(env.__name__, self.mode), directory=result_dir)

        Plotter.plot_reward(history, result_dir)
        Plotter.plot_loss(train_losses, result_dir)
        Plotter.plot_fail_reasons(train_fail_reasons, test_fail_reasons, result_dir)

        with open(f"{result_dir}/reward.csv", 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(history)
        with open(f"{result_dir}/loss.csv", 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(train_losses)       


    def Test(self, test_model):
        v, w, C, infeasible_action = self.env.reset(test=test_model)
        self.agent = Agent.Agent(v.shape[1], w.shape[1], self.N_FEATURE, infeasible_action.shape[1], False, self.loss_type, self.CAPACITY, self.BATCH_SIZE, self.GAMMA, self.TARGET_UPDATE_FREQ, self.optimizer)
        self.agent.brain.model.Load(filename="trained_model_{0}_{1}".format(env.__name__, self.mode), directory=f"result/{self.model_name}")  # "src" or f"result/{self.model_name}"
        self.env.render(show=False, title="Initial cross-sections", result_dir=f"result/{self.model_name}")
        total_reward = 0.0

        for i in range(MAX_STEPS):
            if i % 10 == 9:
                print('step:' + str(i+1))
            action, _ = self.agent.get_action(v, w, C, 0.0, infeasible_action)
            v, w, reward, ep_end, fail_reason, infeasible_action = self.env.step(action)
            # print(action)
            total_reward += reward
            print(f"{total_reward = :.3f}")
            
            if ep_end:
                print(f"{fail_reason = }")
                if self.mode == "inc":
                    self.env.render(show=False, title="After optimization ({0} steps)".format(i+1), result_dir=f"result/{self.model_name}")
                elif self.mode == "dec":
                    self.env.sec_num[action[0]] += 50
                    self.env.render(show=False, title="After optimization ({0} steps)".format(i), result_dir=f"result/{self.model_name}")
                break

                    
