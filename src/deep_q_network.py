import torch.nn as nn


class DeepQNetwork(nn.Module):
    """Small MLP that evaluates one possible Tetris placement.

    Input: 4 handcrafted features from Tetris.get_state_properties():
        [lines_cleared, holes, bumpiness, total_height]

    Output: 1 Q-value for that placement.
    """

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Sequential(nn.Linear(4, 64), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(inplace=True))
        self.out = nn.Linear(64, 1)

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x
