import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy = 0.15):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()

class SymmetricCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int = 6, alpha: float = 1.0, beta: float = 1.0, eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        loss_ce = self.ce(logits, targets)

        prob = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, self.num_classes).float().clamp(min=self.eps)
        loss_rce = (- prob * torch.log(one_hot)).sum(dim=1).mean()

        return self.alpha * loss_ce + self.beta * loss_rce