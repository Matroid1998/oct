import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NTXentLoss(torch.nn.Module):

    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def forward(self, anchor, positive, negatives):
        batch_size = anchor.shape[0]
        embeddings = torch.cat([anchor, positive, *negatives])
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        mask = torch.eye(batch_size * 2, device=similarity_matrix.device).bool()
        similarity_matrix = similarity_matrix[~mask].view(batch_size * 2, -1)

        logits = torch.cat([torch.sum(anchor * positive, dim=-1), torch.sum(anchor.unsqueeze(1) * torch.stack(negatives), dim=-1)], dim=1)
        logits = logits / self.temperature

        labels = torch.zeros(batch_size * 2).to(logits.device).long()
        loss = self.criterion(logits, labels)

        return loss
loss_fn = NTXentLoss()

for epoch in range(1):
    for batch in dataloader:
        positive1_images, positive2_images, \*negative_images = batch
        # forward pass through your model to get embeddings
        anchor = model(positive1_images)
        positive = model(positive2_images)
        negatives = [model(negative_image) for negative_image in negative_images]
        loss = loss_fn(anchor, positive, negatives)
        # backward pass and optimization...