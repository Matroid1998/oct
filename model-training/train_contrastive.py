from contrastive_model import ContrastiveModel
from loss_function import CustomNTXentLoss
from custom_data_loader import ContrastiveOCTDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
train_data_paths = 'D:\octalldata'
data_paths = [os.path.join(train_data_paths,file) for file in os.listdir(train_data_paths) if
              os.path.isfile(os.path.join(train_data_paths,file))]

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness = 0.2,contrast = 0.2,saturation = 0.2,hue = 0.1),
    transforms.RandomResizedCrop((299,299),scale = (0.8,1.0),ratio = (0.75,1.33),antialias = True),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size = 3)],p = 0.5),
    transforms.ToTensor(),
])
model = ContrastiveModel(output_dim=512)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = CustomNTXentLoss(temperature=0.5)
train_loader = DataLoader(ContrastiveOCTDataset(data_paths,num_negatives = 64, transform=train_transforms), batch_size=2, shuffle=True)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        positive1_images, positive2_images, *negative_images = batch
        print(positive1_images)
        # positive1_images, positive2_images = positive1_images.to(device), positive2_images.to(device)
        # negative_images = [img.to(device) for img in negative_images]
        z_positive1 = model(positive1_images)
        z_positive2 = model(positive2_images)
        z_negatives = [model(neg) for neg in negative_images]

        # Loss calculation
        loss = loss_fn(z_positive1, z_positive2, z_negatives)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
