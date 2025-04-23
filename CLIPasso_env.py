# file: clipasso_env.py
import torch, clip
from diffvg import render  # pseudocode import

class ClipassoEnv:
    def __init__(self, image_tensor, max_strokes=64, stroke_dim=8, device='cuda'):
        # image_tensor: (3,H,W) normalized for CLIP
        self.device       = device
        self.target_img   = image_tensor.to(device)
        self.max_strokes  = max_strokes
        self.stroke_dim   = stroke_dim   # e.g. 8: (x0,y0,x1,y1,x2,y2,x3,y3)
        # load CLIP once
        self.clip_model, _ = clip.load('ViT-B/32', device=device)
        self.clip_model.eval()
        # precompute target embedding
        with torch.no_grad():
            self.target_embed = self.clip_model.encode_image(self.target_img.unsqueeze(0))
        self.reset()

    def reset(self):
        # blank white canvas
        _, H, W = self.target_img.shape
        self.canvas      = torch.ones(1, 3, H, W, device=self.device)
        with torch.no_grad():
            self.canvas_embed = self.clip_model.encode_image(self.canvas)
        self.n_strokes   = 0
        return self._get_state()

    def step(self, action):
        """
        action = (stop_flag:0/1, stroke_params: tensor[stroke_dim])
        """
        stop, stroke = action
        if stop:
            # final reward = CLIP similarity of finished sketch
            with torch.no_grad():
                sim = torch.cosine_similarity(self.canvas_embed, self.target_embed).item()
            return None, sim, True, {}

        # otherwise render one more stroke:
        # *your* renderer will look roughly like this:
        new_canvas = render(self.canvas, stroke.view(1, -1))  
        # compute CLIP embedding & marginal gain
        with torch.no_grad():
            new_embed = self.clip_model.encode_image(new_canvas)
            sim_old   = torch.cosine_similarity(self.canvas_embed, self.target_embed)
            sim_new   = torch.cosine_similarity(new_embed,      self.target_embed)
        delta = (sim_new - sim_old).item()
        # tiny step cost to discourage unlimited strokes
        cost  = 0.005
        reward = delta - cost

        # update internal state
        self.canvas       = new_canvas
        self.canvas_embed = new_embed
        self.n_strokes   += 1

        done = (self.n_strokes >= self.max_strokes)
        next_state = None if done else self._get_state()
        return next_state, reward, done, {}

    def _get_state(self):
        """
        Return whatever the agent needs as input.
        Easiest: downsample canvas+target to a vector with a small CNN
        or just concatenate the flattened images and stroke count.
        """
        # here’s a toy example: flatten both and append stroke count
        c = torch.cat([
            self.canvas.view(-1),
            self.target_img.view(-1),
            torch.tensor([self.n_strokes / self.max_strokes], device=self.device)
        ])
        return c
