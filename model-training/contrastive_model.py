import torch
import torch.nn as nn
from torchvision.models import inception_v3

class ContrastiveModel(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.backbone = inception_v3(pretrained=False, aux_logits=False)
        in_features = self.backbone.fc.in_features  # Store the in_features of the last layer
        self.backbone.fc = nn.Identity()  # Replace the fully connected layer with an identity layer
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 512),  # Use the stored in_features
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection_head(x)
        return x

from torchviz import make_dot
model = ContrastiveModel(512)  # Initialize your model
x = torch.randn(1, 3, 299, 299)  # Create a random tensor that matches the input size of your model
y = model(x)  # Forward pass

# Visualize the computation graph
dot = make_dot(y, params=dict(list(model.named_parameters())))
dot.view()