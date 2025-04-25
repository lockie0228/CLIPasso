# clipasso_rl.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pydiffvg
import CLIP_.clip as clip
from torchvision import transforms
from torchvision.datasets import ImageFolder
from gym import spaces, Env
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sklearn.neighbors import KernelDensity
from PIL import Image

# ------------------------------------------------------------------------------
# 1) CONFIG & ARGS
# ------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--target-dir', type=str, default='target_images',
                   help='Folder with subfolder(s) of target images')
    p.add_argument('--image-size', type=int, default=224,
                   help='Canvas & CLIP resolution')
    p.add_argument('--max-strokes', type=int, default=64,
                   help='Hard limit on strokes per episode')
    p.add_argument('--kde-bandwidth', type=float, default=0.2,
                   help='Bandwidth for human-stroke KDE prior')
    p.add_argument('--timesteps', type=int, default=200_000,
                   help='Total PPO timesteps')
    p.add_argument('--device', type=str, default='cuda',
                   help='torch device')
    return p.parse_args()

# ------------------------------------------------------------------------------
# 2) HUMAN STROKE PRIOR (KDE)
# ------------------------------------------------------------------------------
def extract_stroke_features(bezier_pts):
    """
    bezier_pts: Tensor[4,2] in pixel coords
    returns: np.array([dir, length, curvature], dtype=float)
    """
    p0, p1, p2, p3 = bezier_pts.cpu().numpy()
    # direction
    vec = p3 - p0
    direction = np.arctan2(vec[1], vec[0])
    # length
    length = np.linalg.norm(vec)
    # curvature approx
    a1 = p1 - p0
    a2 = p2 - p1
    a3 = p3 - p2
    def ang(u,v):
        return np.arccos(np.clip(u.dot(v) / (np.linalg.norm(u)*np.linalg.norm(v)+1e-8), -1, 1))
    curvature = ang(a1,a2) + ang(a2,a3)
    return np.array([direction, length, curvature], dtype=np.float32)

def fit_kde_prior(samples_np, bw):
    kde = KernelDensity(kernel='gaussian', bandwidth=bw)
    kde.fit(samples_np)
    return kde

# ------------------------------------------------------------------------------
# 3) ENVIRONMENT
# ------------------------------------------------------------------------------
class ClipassoEnv(Env):
    def __init__(self, image_list, args):
        super().__init__()
        self.images = image_list
        self.args   = args
        self.device = args.device

        # action: (halt_flag) + 4 control points (8 dims)
        self.action_dim = 1 + 8
        self.action_space = spaces.Box(
            low = np.array([0.] + [-1.]*8, dtype=np.float32),
            high= np.array([1.] + [ 1.]*8, dtype=np.float32),
            dtype=np.float32
        )

        # observation: CLIP embedding of canvas + target = 512*2 dims
        self.clip_model, _ = clip.load('ViT-B/32', device=self.device, jit=False)
        self.clip_model.eval()
        self.preproc = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(*clip._transform.normalize)
        ])
        # 512-d CLIP image feat
        self.obs_dim = 512*2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        # pick a random target
        idx = np.random.randint(len(self.images))
        pil = self.images[idx]
        self.target_pil = pil.convert('RGB')
        self.target_img = self.preproc(self.target_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.target_feat = self.clip_model.encode_image(self.target_img).cpu().numpy().ravel()

        # start with blank white canvas
        c = torch.ones(1,3,self.args.image_size,self.args.image_size, device=self.device)
        self.canvas = c
        with torch.no_grad():
            self.canvas_feat = self.clip_model.encode_image(c).cpu().numpy().ravel()

        self.strokes = []
        self.step_count = 0
        return self._get_obs()

    def step(self, action):
        halt = bool(action[0] > 0.5)
        pts  = torch.from_numpy(action[1:]).float().to(self.device)

        reward = 0.0
        done   = False

        if halt or self.step_count >= self.args.max_strokes:
            done   = True
            # final CLIP similarity
            reward = float(np.dot(self.canvas_feat, self.target_feat) /
                           (np.linalg.norm(self.canvas_feat)*np.linalg.norm(self.target_feat) + 1e-8))
            return self._get_obs(), reward, True, {}

        # --- render new stroke ---
        new_canvas = self._render_stroke(self.canvas, pts)
        # compute CLIP features
        with torch.no_grad():
            new_feat = self.clip_model.encode_image(new_canvas).cpu().numpy().ravel()
        # delta match
        clip_gain = np.dot(new_feat, self.target_feat) / (
                    np.linalg.norm(new_feat)*np.linalg.norm(self.target_feat)+1e-8) \
                    - (np.dot(self.canvas_feat, self.target_feat) / (
                       np.linalg.norm(self.canvas_feat)*np.linalg.norm(self.target_feat)+1e-8))
        # naturalness penalty
        # map normalized pts→pixel coords
        H = self.args.image_size; W = self.args.image_size
        bezier = ((pts.view(4,2)+1) * torch.tensor([W-1,H-1],device=self.device)/2.0)
        feat = extract_stroke_features(bezier)

        # step cost
        step_cost = -0.005

        reward = float(clip_gain + step_cost)

        # accept new canvas
        self.canvas = new_canvas
        self.canvas_feat = new_feat
        self.strokes.append(bezier.cpu().numpy())
        self.step_count += 1

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        # concat canvas and target CLIP feats
        return np.concatenate([self.canvas_feat, self.target_feat]).astype(np.float32)

    def _render_stroke(self, canvas, stroke_params):
        # decode 4 control points normalized [-1,1]→pixel coords
        _,_,H,W = canvas.shape
        cps = (stroke_params.view(4,2)+1) * torch.tensor([W-1,H-1],device=self.device)/2
        cps = cps.unsqueeze(0)  # [1,4,2]

        # build diffvg Path
        num_ctrl = torch.tensor([4], dtype=torch.int32, device=self.device)
        path = pydiffvg.Path(num_control_points=num_ctrl,
                           points=cps,
                           stroke_width=torch.tensor([1.],device=self.device),
                           is_closed=torch.tensor([False],device=self.device))
        group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0],dtype=torch.int32,device=self.device),
                                  fill_color=None,
                                  stroke_color=torch.tensor([0,0,0,1],device=self.device))
        shapes       = [path]
        shape_groups = [group]

        # prepare RGBA background
        bg_np = (canvas[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        bg_np = np.concatenate([bg_np, 255*np.ones((H,W,1),dtype=np.uint8)], axis=2)
        bg     = torch.from_numpy(bg_np).to(self.device)

        # rasterize
        img_rgba = pydiffvg.RenderFunction.apply(W, H, 2, 2,
                                               shapes, shape_groups,
                                               bg)
        img_rgb  = img_rgba[...,:3].permute(2,0,1).unsqueeze(0) / 255.0
        return img_rgb.float()

# ------------------------------------------------------------------------------
# 4) MAIN TRAINING
# ------------------------------------------------------------------------------
def main():
    args = parse_args()
    # load images
    ds = ImageFolder(args.target_dir, transform=transforms.Compose([transforms.ToTensor()]))
    images = [Image.open(p) for p,_ in ds.samples]

    # build human-stroke prior from QuickDraw/Sketchy offline pickles
    # here we fake with random normals for demonstration; replace with real data!
    dummy = np.random.randn(50000,3).astype(np.float32)

    # make env
    env = ClipassoEnv(images, args)
    model = PPO('MlpPolicy', env,
                verbose=1, n_steps=256, batch_size=32,
                tensorboard_log='./logs/')
    model.learn(total_timesteps=args.timesteps)
    model.save('clipasso_rl_model')

if __name__ == '__main__':
    main()
