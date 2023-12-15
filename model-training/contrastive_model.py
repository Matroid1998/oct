import torch
import torch.nn as nn
from torchvision.models import inception_v3

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class ContrastiveModel(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.backbone = inception_v3(pretrained=False, aux_logits=False)
        in_features = self.backbone.fc.in_features  # Store the in_features of the last layer
        self.backbone.fc = nn.Identity()  # Replace the fully connected layer with an identity layer

        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.apply(xavier_init)  # Apply Xavier weight initialization

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection_head(x)
        return x
if __name__ == '__main__':
    model = ContrastiveModel(512)
    x = torch.randn(2, 3, 299, 299)  # Random tensor with the shape [2, 3, 299, 299]
    y = model(x)

    print("Output shape: ", y.shape)