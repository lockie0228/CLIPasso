# file: rl_policy.py
import torch, torch.nn as nn, torch.nn.functional as F

class SketchPolicy(nn.Module):
    def __init__(self, state_dim, stroke_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        # continuous head: Bézier control‐points
        self.mean = nn.Linear(128, stroke_dim)
        self.logstd = nn.Linear(128, stroke_dim)
        # discrete head: stop vs continue
        self.stop = nn.Linear(128, 1)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        mean   = self.mean(x)
        logstd = self.logstd(x).clamp(-5, 2)  # keeps std > 0
        stop_logit = self.stop(x)
        return mean, logstd, stop_logit
