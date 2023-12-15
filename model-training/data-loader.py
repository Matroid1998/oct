import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import os
from torch.utils.data import DataLoader

class ContrastiveOCTDataset(Dataset):
    def __init__(self,data,num_negatives = 10,transform = None,image_size=(299, 299)):
        self.data = data  # List of paths to your images
        self.num_negatives = num_negatives
        self.transform = transform  # Your augmentation function
        self.image_size = image_size
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        anchor_image_path = self.data[index]
        anchor_image = self.load_image(anchor_image_path)
        anchor_disease = self.extract_disease(anchor_image_path)
        anchor_patient_id, anchor_image_number = self.extract_patient_id_and_image_number(anchor_image_path)
        strategy = np.random.choice(['typical', 'patient', 'disease', 'spatial'])
        positive1_image = anchor_image

        if strategy == 'typical':
            positive2_image = self.load_image(anchor_image_path)
            negative_indices = torch.randint(0, len(self.data), (self.num_negatives,))
            negative_images = [self.load_image(self.data[i]) for i in negative_indices]
        else:
            if strategy == 'patient':
                other_images_same_patient = [path for path in self.data if
                                             self.extract_patient_id(path) == anchor_patient_id]
                negative_indices = [i for i in range(len(self.data)) if
                                    self.extract_patient_id(self.data[i]) != anchor_patient_id]
            elif strategy == 'disease':
                other_images_same_patient = [path for path in self.data if self.extract_disease(path) == anchor_disease]
                negative_indices = [i for i in range(len(self.data)) if
                                    self.extract_disease(self.data[i]) != anchor_disease]
            elif strategy == 'spatial':
                other_images_same_patient = [path for path in self.data if
                                             self.extract_patient_id_and_image_number(path)[0] == anchor_patient_id and abs(
                                                 int(self.extract_patient_id_and_image_number(path)[
                                                         1])-anchor_image_number) == 1]
                negative_indices = [i for i in range(len(self.data)) if
                                    self.extract_patient_id(self.data[i]) != anchor_patient_id]
            if len(other_images_same_patient) >= 2:
                positive2_image_path = np.random.choice(other_images_same_patient)
                positive2_image = self.load_image(positive2_image_path)
            else:
                positive2_image = positive1_image
            negative_indices = torch.randint(0,len(negative_indices),(self.num_negatives,))
            negative_images = [self.load_image(self.data[i]) for i in negative_indices]

        return (positive1_image,positive2_image, *negative_images)

    def load_image(self,path):
        image = Image.open(path).convert('RGB')
        image = image.resize(self.image_size)  # Resize the image to the desired size
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).float()
            image = image.permute(2, 0, 1)
        return image
    def extract_patient_id(self, path):
        _, _, patient_id, _ = path.split('_')
        return patient_id

    def extract_disease(self, path):
        _, disease, _, _ = path.split('_')
        return disease

    def extract_patient_id_and_image_number(self, path):
        _, _, patient_id, image_number = path.split('_')
        return int(patient_id),int(image_number.split('.')[0])

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop((299, 299), scale=(0.8, 1.0), ratio=(0.75, 1.33),antialias=True),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
        transforms.ToTensor(),
    ])
    image_directory = 'D:\octalldata'

    data_paths = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, file))]
    dataset = ContrastiveOCTDataset(data_paths, num_negatives=10, transform=transform)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    for epoch in range(1):
        for batch in dataloader:
            positive1_images, positive2_images, *negative_images = batch
            print("Shape of positive1_images: ",positive1_images.shape)
            print("Shape of positive2_images: ",positive2_images.shape)
            for i,negative_image in enumerate(negative_images):
                print(f"Shape of negative_image {i}: ",negative_image.shape)