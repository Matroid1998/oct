import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import os
from torch.utils.data import DataLoader

class ContrastiveOCTDataset(Dataset):
    def __init__(self, data, num_negatives=10, transform=None, image_size=(299, 299)):
        self.data = data
        self.num_negatives = num_negatives
        self.transform = transform
        self.image_size = image_size

        # Pre-compute groupings
        self.patient_groupings = self._create_groupings(lambda path: self.extract_patient_id(path))
        self.disease_groupings = self._create_groupings(lambda path: self.extract_disease(path))
        self.spatial_groupings = self._create_groupings(lambda path: self.extract_patient_id_and_image_number(path)[0])

    def _create_groupings(self, key_function):
        groupings = {}
        for idx, path in enumerate(self.data):
            key = key_function(path)
            if key not in groupings:
                groupings[key] = []
            groupings[key].append(idx)
        return groupings
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        anchor_image_path = self.data[index]
        anchor_image = self.load_image(anchor_image_path)
        anchor_disease = self.extract_disease(anchor_image_path)
        anchor_patient_id,anchor_image_number = self.extract_patient_id_and_image_number(anchor_image_path)
        strategy = np.random.choice(['typical','patient','disease','spatial'])
        positive1_image = anchor_image

        # Selecting positive and negative pairs based on the strategy
        if strategy == 'typical':
            positive2_image = self.load_image(anchor_image_path)  # Same image but different augmentation
            negative_indices = torch.randint(0,len(self.data),(self.num_negatives,))
        else:
            if strategy == 'patient':
                positive_images = self.patient_groupings[anchor_patient_id]
            elif strategy == 'disease':
                positive_images = self.disease_groupings[anchor_disease]
            elif strategy == 'spatial':
                positive_images = self.spatial_groupings[anchor_patient_id]

            # Choose a different positive image
            if len(positive_images) > 1:
                positive_images.remove(index)
                positive2_index = np.random.choice(positive_images)
                positive2_image = self.load_image(self.data[positive2_index])
            else:
                positive2_image = positive1_image

            negative_indices = self._select_negative_samples(index,self.patient_groupings if strategy in ['patient',
                                                                                                          'spatial'] else self.disease_groupings,
                                                             anchor_patient_id if strategy in ['patient',
                                                                                               'spatial'] else anchor_disease)

        negative_images = [self.load_image(self.data[i]) for i in negative_indices]

        return (positive1_image,positive2_image,*negative_images)

    def _select_negative_samples(self,index,groupings,key):
        positive_indices = groupings[key]
        negative_indices = [i for i in range(len(self.data)) if i not in positive_indices]
        return torch.randint(0,len(negative_indices),(self.num_negatives,))

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