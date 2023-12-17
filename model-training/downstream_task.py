import os.path
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from contrastive_model import ContrastiveModel
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm
with open('config/downstream_config.json') as config_file:
    config = json.load(config_file)
learning_rate = config["learning_rate"]
weight_decay = config["weight_decay"]
dataset_path = config["dataset_path"]
csv_path = config["csv_path"]
num_epochs = config["num_epochs"]
batch_size = config["batch_size"]
scheduler_step_size = config["scheduler_step_size"]
scheduler_gamma = config["scheduler_gamma"]
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
saved_model_path = config['saved_model_path']

def read_worstcase_images(file_path,csv_path,imageSize = 299):
    df = pd.read_csv(csv_path)
    X_patient = []
    y_patient = []

    transform = transforms.Compose([
        transforms.Resize((imageSize,imageSize)),
        transforms.ToTensor()
    ])

    for patient_class in np.unique(df['Class']):
        df_classwise = df[df['Class'] == patient_class]

        for patient_index in np.unique(df_classwise['Patient ID']):
            X = []
            y = []

            df_patientwise = df_classwise[df_classwise['Patient ID'] == patient_index]

            for i in range(len(df_patientwise)):
                if df_patientwise.iloc[i]['Class'] == df_patientwise.iloc[i]['Label']:
                    img_path = os.path.join(file_path,df_patientwise.iloc[i]['Directory'])
                    img = Image.open(img_path).convert('RGB')
                    img = transform(img)
                    X.append(img)
                    label_map = {'normal':0,'drusen':1,'cnv':2}
                    y.append(label_map[df_patientwise.iloc[i]['Label'].lower()])
            X_patient.append(X)
            y_patient.append(y)
    return X_patient,y_patient

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class FineTunedContrastiveModel(nn.Module):
    def __init__(self, original_model, num_classes):
        super().__init__()
        self.model = original_model
        self.classifier = nn.Linear(original_model.projection_head[-1].out_features, num_classes)

    def forward(self, x):
        features = self.model(x)
        output = self.classifier(features)
        return output

original_model = ContrastiveModel(output_dim=512)
original_model.load_state_dict(torch.load(saved_model_path))
model = FineTunedContrastiveModel(original_model, num_classes=3).to(device)

X_patient, y_patient = read_worstcase_images(dataset_path, csv_path)
num_epochs = num_epochs
learning_rate = learning_rate
X = [img for patient_imgs in X_patient for img in patient_imgs]
y = [label for patient_labels in y_patient for label in patient_labels]
X = torch.stack(X)
y = torch.tensor(y, dtype=torch.long)
print(X.shape)
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, shuffle=True)
train_dataset = TensorDataset(X_train, y_train)
eval_dataset = TensorDataset(X_eval, y_eval)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=2, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
loss_fn = nn.CrossEntropyLoss()
num_classes = 3
for epoch in range(num_epochs):
    model.train()
    train_progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch+1}/{num_epochs}")
    total_train_loss = 0
    correct_train = 0
    total_train = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    for batch_idx, (inputs, labels) in train_progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
        train_progress_bar.set_postfix({'Train Loss':loss.item()})
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

        if (batch_idx + 1) % 10 == 0:
            train_accuracy = 100 * correct_train / total_train
            train_loss = total_train_loss / 10
            print(f"Batch {batch_idx + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

            for i in range(num_classes):
                if class_total[i] > 0:
                    print(f'Class {i} - Sensitivity: {100 * class_correct[i] / class_total[i]:.2f}%')
                else:
                    print(f'Class {i} - Sensitivity: N/A')
            total_train_loss = 0
            correct_train = 0
            total_train = 0
            class_correct = list(0. for i in range(num_classes))
            class_total = list(0. for i in range(num_classes))
    model.eval()
    total_eval_loss = 0
    class_correct_eval = list(0. for i in range(num_classes))
    class_total_eval = list(0. for i in range(num_classes))

    with torch.no_grad():
        for inputs,labels in eval_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs,labels)
            total_eval_loss += loss.item()

            _,predicted = torch.max(outputs,1)
            correct = (predicted == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct_eval[label] += correct[i].item()
                class_total_eval[label] += 1

    eval_loss = total_eval_loss/len(eval_loader)
    eval_accuracy = sum(class_correct_eval)/sum(class_total_eval)
    print(f'Epoch {epoch+1}: Eval Loss: {eval_loss:.4f}, Eval Accuracy: {eval_accuracy:.4f}')
    for i in range(num_classes):
        sensitivity = 100*class_correct_eval[i]/class_total_eval[i] if class_total_eval[i] > 0 else 0
        print(f'Class {i} - Sensitivity: {sensitivity:.2f}%')

