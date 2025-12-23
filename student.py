import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.vector import AsyncVectorEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import cma
import os
from shapely import affinity
from shapely.geometry import Point, Polygon

from models.C import C
from models.M import MDN_LSTM
from models.V import VAE


class GrasspenaltyWrapper(gym.Wrapper):
    """Wrapper that applies extra penalty when car is on grass instead of track."""
    def __init__(self, env, grass_penalty: float = -0.5, border_width: float = 0.5):
        super().__init__(env)
        self.grass_penalty = grass_penalty
        self.border_width = border_width
        print(f"GrasspenaltyWrapper initialized with penalty {self.grass_penalty} for off-track driving.")
    
    def car_on_track(self):
        car_on_track = False
        x, y = self.unwrapped.car.hull.position
        point = Point(x, y)
        for poly in self.unwrapped.road_poly:
            polygon = Polygon(poly[0])
            # Create a larger polygon representing the track + safe border
            if self.border_width > 0:
                border_scale = 1 + self.border_width
                polygon = affinity.scale(polygon, xfact=border_scale, yfact=border_scale)

            if polygon.contains(point):
                car_on_track = True
                break
        return car_on_track

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if not self.car_on_track():
            reward += self.grass_penalty
            terminated = True # End the episode immediately
        
        return obs, reward, terminated, truncated, info


class Policy(nn.Module):
    continuous = True 

    def __init__(self, device=torch.device('cpu')):
        super(Policy, self).__init__()
        self.device = device
        
        self.z_size = 32
        self.action_size = 3
        self.hidden_size = 512
        self.grass_penalty = -150.0
        
        # --- Load VAE ---
        vae_path = "residual_vae_final.ckpt"
        if os.path.exists(vae_path):
            print(f"Loading VAE from {vae_path}...")
            self.vae = VAE.load_from_checkpoint(vae_path, latent_size=self.z_size)
        else:
            print("Warning: VAE checkpoint not found. Random init.")
            self.vae = VAE(latent_size=self.z_size)
        
        # --- Load MDN-RNN ---
        mdn_path = "MDN_LSTM_checkpoint.ckpt"
        input_size = self.z_size + self.action_size
        if os.path.exists(mdn_path):
            print(f"Loading MDN-RNN from {mdn_path}...")
            self.mdn = MDN_LSTM.load_from_checkpoint(
                mdn_path, 
                input_size=input_size, 
                output_size=self.z_size, 
                hidden_size=self.hidden_size
            )
        else:
            print("Warning: MDN checkpoint not found. Random init.")
            self.mdn = MDN_LSTM(input_size=input_size, 
                               output_size=self.z_size, 
                               hidden_size=self.hidden_size)
        
        self.vae.freeze()
        self.mdn.freeze()

        # --- Controller ---
        self.controller = C(input_size=self.z_size + self.hidden_size, 
                            output_size=self.action_size)

    
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        self.hidden_state = None
        self.cell_state = None
        self.prev_action = None
        
        self.to(self.device)

    def to(self, device):
        ret = super().to(device)
        ret.device = device
        # Fix for PL modules defaulting to CPU
        if hasattr(self.vae, '_device'): self.vae._device = device
        if hasattr(self.mdn, '_device'): self.mdn._device = device
        return ret

    def reset(self):
        self.hidden_state = torch.zeros(1, 1, self.hidden_size).to(self.device)
        self.cell_state = torch.zeros(1, 1, self.hidden_size).to(self.device)
        self.prev_action = torch.zeros(1, self.action_size).to(self.device)

    def act(self, state):
        if self.hidden_state is None:
            self.reset()

        # 1. Preprocess
        obs = self.transform(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mu, logvar = self.vae.encode(obs)
            z = self.vae.latent(mu, logvar)
            # 2. Memory Update
            rnn_input = torch.cat([z, self.prev_action], dim=1)
            _, (h, c) = self.mdn.forward_lstm(rnn_input, (self.hidden_state, self.cell_state))
            
            self.hidden_state = h
            self.cell_state = c
            
            # 3. Controller
            context = torch.cat([z, h.squeeze(0)], dim=1)
            raw_action = self.controller(context).cpu().numpy()[0]
            
            action = raw_action.copy()
            
            if action[1] < 0: action[1] = 0  # Clip negative gas
            if action[2] < 0: action[2] = 0  # Clip negative brake
            action[2] = action[2] * (1 - action[1])  # Reduce brake when gas is pressed

            # Feed the RNN the executed action to keep its state aligned with environment dynamics
            self.prev_action = torch.from_numpy(action).unsqueeze(0).to(self.device)
            
            return action

    def train(self):
        print("Loading controller weights...")
        self.load()
        print(f"Starting Training on {self.device}...")
        
        POP_SIZE = 32 
        N_GENERATIONS = 600 
        SIGMA_INIT = 0.1
        
        self.vae.eval()
        self.mdn.eval()
        
        def get_flat_params(model):
            w = model.weights.detach().cpu().numpy().flatten()
            b = model.bias.detach().cpu().numpy().flatten()
            return np.concatenate([w, b])

        def set_flat_params(model, flat_params):
            input_sz = model.input_size
            output_sz = model.output_size
            w_end = input_sz * output_sz
            w_flat = flat_params[:w_end]
            b_flat = flat_params[w_end:]
            
            w_new = torch.tensor(w_flat).view(output_sz, input_sz).float().to(self.device)
            b_new = torch.tensor(b_flat).view(output_sz).float().to(self.device)
            
            model.weights = w_new
            model.bias = b_new

        initial_params = get_flat_params(self.controller)
        es = cma.CMAEvolutionStrategy(initial_params, SIGMA_INIT, {'popsize': POP_SIZE})
        
        # Create vectorized environment for parallel evaluation
        def make_env():
            env = gym.make('CarRacing-v2', continuous=True, render_mode=None)
            env = GrasspenaltyWrapper(env, grass_penalty=self.grass_penalty)
            return env
        
        vec_env = AsyncVectorEnv([make_env for _ in range(POP_SIZE)])
        best_global_reward = -float('inf')

        for generation in range(N_GENERATIONS):
            if es.stop(): break
            
            solutions = es.ask()
            rewards = np.zeros(POP_SIZE)
            
            # Reset all environments
            obs_list, _ = vec_env.reset()
            
            # Initialize hidden states for all candidates
            hidden_states = [torch.zeros(1, 1, self.hidden_size).to(self.device) for _ in range(POP_SIZE)]
            cell_states = [torch.zeros(1, 1, self.hidden_size).to(self.device) for _ in range(POP_SIZE)]
            prev_actions = [torch.zeros(1, self.action_size).to(self.device) for _ in range(POP_SIZE)]
            
            dones = np.zeros(POP_SIZE, dtype=bool)
            steps = np.zeros(POP_SIZE, dtype=int)
            max_steps = 1000
            
            while not dones.all() and steps.min() < max_steps:
                actions = np.zeros((POP_SIZE, self.action_size))
                
                # Compute actions for all candidates in parallel
                for i, candidate in enumerate(solutions):
                    if dones[i]:
                        continue
                        
                    # Set controller weights for this candidate
                    set_flat_params(self.controller, candidate)
                    
                    # Preprocess observation
                    obs = self.transform(obs_list[i]).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        mu, logvar = self.vae.encode(obs)
                        z = mu
                        
                        # Memory Update
                        rnn_input = torch.cat([z, prev_actions[i]], dim=1)
                        _, (h, c) = self.mdn.forward_lstm(rnn_input, (hidden_states[i], cell_states[i]))
                        
                        hidden_states[i] = h
                        cell_states[i] = c
                        
                        # Controller
                        context = torch.cat([z, h.squeeze(0)], dim=1)
                        raw_action = self.controller(context).cpu().numpy()[0]
                        
                        # Action mapping
                        action = raw_action.copy()
                        if action[1] < 0: action[1] = 0
                        if action[2] < 0: action[2] = 0
                        action[2] = action[2] * (1 - action[1])
                        
                        prev_actions[i] = torch.from_numpy(action).unsqueeze(0).to(self.device)
                        actions[i] = action
                
                # Step all environments
                obs_list, reward_list, terminated_list, truncated_list, _ = vec_env.step(actions)
                
                # Update rewards and done flags
                for i in range(POP_SIZE):
                    if not dones[i]:
                        rewards[i] += reward_list[i]
                        steps[i] += 1
                        dones[i] = terminated_list[i] or truncated_list[i] or (steps[i] >= max_steps)
                        
                        # Penalize early termination
                        if dones[i] and steps[i] < max_steps and rewards[i] < 0:
                            rewards[i] -= 100
            
            es.tell(solutions, [-r for r in rewards])
            
            curr_best = max(rewards)
            mean_reward = np.median(rewards)

            # Identify best candidate of this generation
            best_idx = int(np.argmax(rewards))
            # Set controller to best candidate for reproducible logging
            set_flat_params(self.controller, solutions[best_idx])

            # Record video of the best candidate every generation
            try:
                self.record_best_run(solutions[best_idx], generation+1)
                print(f"Saved video for generation {generation+1} in videos/ directory")
            except Exception as e:
                print(f"Warning: failed to record video for gen {generation+1}: {e}")

            # Save model when the best improves
            if mean_reward > best_global_reward:
                best_global_reward = mean_reward
                self.save()
            
            print(f"Gen {generation+1}: Best: {curr_best:.2f} | Median : {mean_reward:.2f}")
            es.disp()
            
        vec_env.close()
        set_flat_params(self.controller, es.result.xbest)
        self.save()

    def save(self):
        print("Saving Policy to model.pt...")
        torch.save(self.controller.state_dict(), 'model.pt')

    def load(self):
        if os.path.exists('controller.pt'):
            print("Loading Policy from controller.pt...")
            self.controller.load_state_dict(torch.load('controller.pt', map_location=self.device))
        else:
            print("No controller.pt found.")

    def record_best_run(self, candidate_params, generation: int, max_steps: int = 1000):
        """Replay a single episode with given controller params and record a video."""
        # Set controller weights from flat params
        input_sz = self.controller.input_size
        output_sz = self.controller.output_size
        w_end = input_sz * output_sz
        w_flat = candidate_params[:w_end]
        b_flat = candidate_params[w_end:]

        w_new = torch.tensor(w_flat).view(output_sz, input_sz).float().to(self.device)
        b_new = torch.tensor(b_flat).view(output_sz).float().to(self.device)
        self.controller.weights = w_new
        self.controller.bias = b_new

        # Prepare output folder
        video_dir = os.path.join('videos')
        os.makedirs(video_dir, exist_ok=True)

        # Create env with video recorder and grass penalty
        env = gym.make('CarRacing-v2', continuous=True, render_mode='rgb_array')
        env = GrasspenaltyWrapper(env, grass_penalty=self.grass_penalty)  # Extra penalty for grass
        env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda ep: True,
                          name_prefix=f'gen{generation:03d}_best')

        obs, _ = env.reset()
        self.reset()

        steps = 0
        done = False
        truncated = False
        while not (done or truncated) and steps < max_steps:
            action = self.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        env.close()
