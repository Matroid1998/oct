import torch

class CustomNTXentLoss(torch.nn.Module):
    def __init__(self, temperature):
        super(CustomNTXentLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    def forward(self, anchor, positive, negatives):
        """
        Calculate the modified NT-Xent loss.

        Parameters:
        - anchor: Representations for the anchor images
        - positive: Representations for the positive images
        - negatives: List of representations for the negative images

        All inputs should have the shape (batch_size, feature_dim)
        """
        anchor_positive_similarity = self.cosine_similarity(anchor, positive) / self.temperature
        negatives_similarity = torch.stack([self.cosine_similarity(anchor, neg) for neg in negatives]) / self.temperature
        negatives_similarity = torch.logsumexp(negatives_similarity, dim=0)
        loss = - anchor_positive_similarity + negatives_similarity
        return loss.mean()


if __name__ == '__main__':
    batch_size = 32
    feature_dim = 512
    temperature = 0.5
    anchor = torch.randn(batch_size, feature_dim)
    positive = torch.randn(batch_size, feature_dim)
    negatives = [torch.randn(batch_size, feature_dim) for _ in range(10)]  # 10 negative samples
    loss_fn = CustomNTXentLoss(temperature)
    losss = loss_fn(anchor, positive, negatives)
    print("Loss: ", losss)
