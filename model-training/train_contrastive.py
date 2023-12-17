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
from torch.optim.lr_scheduler import StepLR
import json
from PIL import Image
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')
with open('config\config.json', 'r') as config_file:
    config = json.load(config_file)
train_data_paths = config["train_data_path"]
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]
learning_rate = config["learning_rate"]
weight_decay = config["weight_decay"]
scheduler_step_size = config["scheduler_step_size"]
scheduler_gamma = config["scheduler_gamma"]
save_interval = config["save_interval"]
num_negatives = config["num_negatives"]
output_dim = config["output_dim"]
temperature = config["temperature"]
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
tb_writer = SummaryWriter('OCT_trainer')
data_paths = [os.path.join(train_data_paths,file) for file in os.listdir(train_data_paths) if
              os.path.isfile(os.path.join(train_data_paths,file))]
def calculate_dataset_stats(data_paths):
    transform = transforms.Compose([transforms.ToTensor()])
    total_sum = np.zeros(3)
    total_square_sum = np.zeros(3)
    num_pixels = 0

    for path in data_paths:
        image = Image.open(path).convert('RGB')
        tensor = transform(image)
        total_sum += tensor.mean([1, 2]).numpy()
        total_square_sum += (tensor ** 2).mean([1, 2]).numpy()
        num_pixels += tensor.size(1) * tensor.size(2)

    mean = total_sum / len(data_paths)
    std = np.sqrt(total_square_sum / len(data_paths) - mean ** 2)
    return mean, std

# Calculate mean and std
mean, std = calculate_dataset_stats(data_paths)
print("Calculated Mean:", mean)
print("Calculated Std:", std)

train_transforms = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop((299, 299), scale=(0.8, 1.0), ratio=(0.75, 1.33), antialias=True),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)  # Normalization step
])
model = ContrastiveModel(output_dim=output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
loss_fn = CustomNTXentLoss(temperature=temperature)
train_loader = DataLoader(ContrastiveOCTDataset(data_paths,num_negatives = num_negatives,
                                                transform=train_transforms), batch_size=batch_size, shuffle=True
                                                ,num_workers = 12)
log_filename = f"logs\\training_log_{timestamp}.txt"
best_loss = float('inf')
best_model_paths = [None, None]
with open(log_filename, 'a') as log_file:
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        running_loss = 0
        best_ave_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in progress_bar:
            optimizer.zero_grad()
            positive1_images,positive2_images,*negative_images = batch
            positive1_images,positive2_images = positive1_images.to(device),positive2_images.to(device)
            all_negatives = torch.cat(negative_images,dim = 0).to(device)
            z_positive1 = model(positive1_images)
            z_positive2 = model(positive2_images)
            z_negatives = model(all_negatives)
            z_negatives = z_negatives.split(positive1_images.size(0)) ## final shape is (number of negatives, batch size)
            loss = loss_fn(z_positive1,z_positive2,z_negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            running_loss += loss.item()
            tb_writer.add_scalar('Loss/train',loss,batch_idx)
            best_ave_loss += loss.item()
            scheduler.step()
            if batch_idx % save_interval == save_interval-1:
                current_loss = best_ave_loss/save_interval
                best_ave_loss = 0
                if current_loss < best_loss:
                    best_loss = current_loss
                    if best_model_paths[0] is not None:
                        os.remove(best_model_paths[0])
                    best_model_paths[0] = best_model_paths[1]
                    best_model_filename = f"best_model_batch_{epoch+1}_{batch_idx+1}_{timestamp}.pth"
                    best_model_paths[1] = os.path.join('saved_models',best_model_filename)
                    torch.save(model.state_dict(),best_model_paths[1])
                    print(f"New best model saved: {best_model_paths[1]}")
                    print('________________________')
            if batch_idx % 10 == 9:
                current_lr = scheduler.get_last_lr()[0]
                tb_writer.add_scalar('Learning Rate',current_lr,epoch*len(train_loader)+batch_idx)
                last_loss = total_loss / 10
                log_message = f'\nEpoch {epoch+1}, Batch {batch_idx + 1}, Batch Loss: {last_loss}\n'
                log_file.write(log_message)
                log_file.write(f'\nLearning rate : {current_lr}')
                log_file.flush()
                print('\nbatch {} loss: {}'.format(batch_idx + 1, last_loss))
                tb_x = epoch * len(train_loader) + batch_idx + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0
                print('____________')
        avg_loss = total_loss / len(train_loader)
        tb_writer.add_scalar('Epoch/Average Loss', avg_loss, epoch)
        log_message = f'\nEpoch Completed: {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}\n'
        log_file.write(log_message)
        log_file.flush()
        print(f"\nEpoch Completed: {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
