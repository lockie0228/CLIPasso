# file: train_rl.py
import gym, torch
from gym.spaces import Box, Dict
from CLIPasso_env import ClipassoEnv
from rl_policy import SketchPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ClipassoGymEnv(gym.Env):
    def __init__(self, image_dataset):
        super().__init__()
        # assume image_dataset is a list of preloaded torch tensors [3,H,W]
        self.images = image_dataset
        # state_dim = 2*3*H*W + 1  (canvas + target + stroke_frac)
        # but Gym wants shapes; we’ll return a flat Box:
        H, W = self.images[0].shape[1:]
        dim = 2*3*H*W + 1
        self.observation_space = Box(-1e6, 1e6, (dim,), dtype=float)
        # action: stop_flag in {0,1}, stroke_params continuous in [-1,1]
        # we model it as continuous; stop_flag will be sampled via Bernoulli(logit)
        self.action_space = Dict({
            'stop': Box(0.0, 1.0, (1,)),
            'stroke': Box(0.0, 1.0, (8,))  # 8‐d Bézier
        })

    def reset(self):
        self.env = ClipassoEnv(image_tensor=random.choice(self.images))
        return self.env.reset().cpu().numpy()

    def step(self, action):
        stop = int(action['stop'][0] > 0.5)
        stroke = torch.tensor(action['stroke'], device='cuda').float()
        ns, r, done, info = self.env.step((stop, stroke))
        return (ns.cpu().numpy() if ns is not None else None), r, done, info
    


from torchvision.datasets import ImageFolder
from torchvision import transforms

# load your images into a list of torch tensors
ds = ImageFolder('/content/target_image', transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(*clip_normalize)  # use CLIP’s normalization
]))
images = [ds[i][0] for i in range(len(ds))]

gym_env = ClipassoGymEnv(images)
model = PPO('MlpPolicy', gym_env, verbose=1, batch_size=16, n_steps=128)
model.learn(total_timesteps=200_000)

# save
model.save('clipasso_rl')
