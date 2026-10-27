import torch
import torch.nn as nn

class FC(nn.Module):
    #TODO 3072 -> 1024 ->  512 -> 256 -> 128 -> 64 ->32 -> 16 ->10
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Flatten(),                       # (N,3,32,32) -> (N,3072)
            nn.Linear(3072, 1024), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 10),
        )
    def forward(self, x):
        return self.net(x)
    
