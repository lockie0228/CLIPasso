import os
import torch
import clip
import gym
import numpy as np
from gym import spaces
from torchvision.datasets import ImageFolder
from torchvision import transforms
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# =============================================================================
# 1. Environment Definition: ClipassoEnv
#    - Sequential stroke-by-stroke drawing
#    - State = [canvas_image, target_image, stroke_fraction]
#    - Action = (stop_flag — discrete, stroke_params — continuous)
#    - Reward = delta_CLIP - step_cost
# =============================================================================
class ClipassoEnv(gym.Env):
    metadata = {'render.modes': []}

    def __init__(self, images, max_strokes=64, stroke_dim=8, cost=0.005, device='cuda'):
        super().__init__()
        self.device = device
        # List of target images (torch.Tensor [3,H,W])
        self.images = images
        self.max_strokes = max_strokes
        self.stroke_dim = stroke_dim
        self.step_cost = cost

        # Load CLIP (frozen) for similarity
        self.clip_model, _ = clip.load('ViT-B/32', device=self.device)
        self.clip_model.eval()

        # Placeholder for action/observation spaces
        # Observation: concatenated flattened canvas + target + normalized stroke count
        C, H, W = self.images[0].shape
        obs_dim = 2 * C * H * W + 1
        self.observation_space = spaces.Box(-1e6, 1e6, (obs_dim,), dtype=np.float32)
        # Action: stop in [0,1], stroke_params in [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1 + self.stroke_dim,), dtype=np.float32)

    def reset(self):
        # Choose a random target image each episode
        self.target = self.images[np.random.randint(len(self.images))].to(self.device)
        # Initialize blank white canvas
        C, H, W = self.target.shape
        self.canvas = torch.ones((1, C, H, W), device=self.device)
        # Compute initial CLIP embedding
        with torch.no_grad():
            self.target_embed = self.clip_model.encode_image(self.target.unsqueeze(0))
            self.canvas_embed = self.clip_model.encode_image(self.canvas)
        self.n_strokes = 0
        return self._get_state()

    def step(self, action):
        # Unpack action vector: first element = stop logit, rest = stroke parameters
        stop_logit = action[0]
        stroke_params = torch.tensor(action[1:], device=self.device).float()

        # Discrete stop decision: sigmoid(stop_logit) > 0.5
        stop_prob = torch.sigmoid(torch.tensor(stop_logit, device=self.device))
        stop_flag = (stop_prob > 0.5).item()

        if stop_flag:
            # Final reward = CLIP similarity of final canvas
            with torch.no_grad():
                final_sim = torch.cosine_similarity(self.canvas_embed, self.target_embed).item()
            return None, final_sim, True, {}

        # Otherwise, render one stroke onto canvas
        # --- Technique: use diffvg differentiable rasterizer ---
        # Replace this placeholder with your diffvg rendering call
        # e.g., self.canvas = diffvg_rasterize(self.canvas, stroke_params)
        self.canvas = self._render_stroke(self.canvas, stroke_params)
        self.n_strokes += 1

        # Compute new CLIP embedding and marginal gain
        with torch.no_grad():
            new_embed = self.clip_model.encode_image(self.canvas)
            sim_old = torch.cosine_similarity(self.canvas_embed, self.target_embed)
            sim_new = torch.cosine_similarity(new_embed, self.target_embed)
        delta_sim = (sim_new - sim_old).item()
        reward = delta_sim - self.step_cost  # Step cost shapes parsimony

        # Update state
        self.canvas_embed = new_embed
        done = (self.n_strokes >= self.max_strokes)

        return self._get_state(), reward, done, {}

    def _get_state(self):
        # Flatten canvas + target + stroke fraction
        c = self.canvas.view(-1).cpu().numpy()
        t = self.target.view(-1).cpu().numpy()
        frac = np.array([self.n_strokes / self.max_strokes], dtype=np.float32)
        return np.concatenate([c, t, frac], axis=0)

    def _render_stroke(self, canvas, stroke_params):
        # Placeholder: replace with actual diffvg render call
        # Here we simply return canvas unchanged
        return canvas

# =============================================================================
# 2. Policy Network (PyTorch) for PPO
#    - Outputs: [stop_logit], [stroke_params_mean], [stroke_params_logstd]
#    - Shared backbone processes the flattened state
# =============================================================================
import torch.nn as nn
import torch.nn.functional as F

class SketchPolicy(nn.Module):
    def __init__(self, obs_dim, stroke_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        # Heads
        self.stop_head = nn.Linear(256, 1)
        self.mean_head = nn.Linear(256, stroke_dim)
        self.logstd_head = nn.Linear(256, stroke_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        stop_logit = self.stop_head(x)
        mean = self.mean_head(x)
        logstd = self.logstd_head(x).clamp(-5, 2)
        return stop_logit, mean, logstd

# =============================================================================
# 3. Training Script Using Stable-Baselines3 PPO
#    - Wrap ClipassoEnv in DummyVecEnv for parallelism
#    - Use MlpPolicy but override the policy network with SketchPolicy
# =============================================================================

def make_env(images):
    return ClipassoEnv(images)

if __name__ == '__main__':
    # 1) Load target images from 'target_image' directory
    transform = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(*clip_normalize)
    ])
    ds = ImageFolder('target_images', transform=transform)
    images = [ds[i][0] for i in range(len(ds))]

    # 2) Create vectorized env
    vec_env = DummyVecEnv([lambda: make_env(images)])

    # 3) Instantiate PPO with custom network
    from stable_baselines3.common.torch_layers import BasePolicy
    from stable_baselines3.common.policies import ActorCriticPolicy

    class CustomPolicy(ActorCriticPolicy):
        def _build_mlp_extractor(self):
            self.mlp_extractor = nn.Module()
            self.mlp_extractor.policy_net = SketchPolicy(self.observation_space.shape[0], self.action_space.shape[1]-1)
            self.mlp_extractor.value_net = nn.Sequential(
                nn.Linear(self.observation_space.shape[0], 64), nn.Tanh(), nn.Linear(64,1)
            )

    model = PPO(CustomPolicy, vec_env, verbose=1, batch_size=16, n_steps=128, tensorboard_log='./logs')

    # 4) Train
    model.learn(total_timesteps=200_000)

    # 5) Save
    os.makedirs('models', exist_ok=True)
    model.save('models/clipasso_rl')
    print("Training complete. Model saved to models/clipasso_rl.zip")
