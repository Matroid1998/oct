from contrastive_model import ContrastiveModel
from loss_function import CustomNTXentLoss
from custom_data_loader import ContrastiveOCTDataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
tb_writer = SummaryWriter('OCT_trainer')
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
train_loader = DataLoader(ContrastiveOCTDataset(data_paths,num_negatives = 10, transform=train_transforms), batch_size=2, shuffle=True)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, batch in progress_bar:
        optimizer.zero_grad()
        positive1_images,positive2_images,*negative_images = batch
        # positive1_images,positive2_images = positive1_images.to(device),positive2_images.to(device)
        all_negatives = torch.cat(negative_images,dim = 0)
        # .to(device)
        z_positive1 = model(positive1_images)
        z_positive2 = model(positive2_images)
        z_negatives = model(all_negatives)
        z_negatives = z_negatives.split(positive1_images.size(0)) ## final shape is (number of negatives, batch size)
        loss = loss_fn(z_positive1,z_positive2,z_negatives)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 9:
            last_loss = total_loss / 10 # loss per batch
            print('  batch {} loss: {}'.format(batch_idx + 1, last_loss))
            tb_x = epoch * len(train_loader) + batch_idx + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            total_loss = 0
    avg_loss = total_loss / len(train_loader)
    tb_writer.add_scalar('Epoch/Average Loss', avg_loss, epoch)
    print(f"Epoch Completed: {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
