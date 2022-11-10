import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia import losses

class FocalLossMod(nn.Module):
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=-100):
        super(FocalLossMod, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs_in, targets_in):

        # clone tensors to not disturb tensors that were provided
        inputs = torch.clone(inputs_in)
        targets = torch.clone(targets_in)

        # do everything using a 1d array
        if inputs.dim() > 2:
            # N,C,H,W => N,C,H*W
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            # N,C,H*W => N,H*W,C
            inputs = inputs.transpose(1, 2)
            # N,H*W,C => N*H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))    

        targets = targets.view(-1, 1)

        # drop ignored_index
        mask = targets==self.ignore_index
        targets = targets[~mask.ravel(),:].ravel()
        inputs = inputs[~mask.ravel(),:]

        return losses.focal_loss(inputs, targets, self.alpha, self.gamma, self.reduction)


if __name__ == '__main__':
    import random
    import kornia

    n_classes = 5
    ignore_index = 3
    # Example of target with class indices
    loss = nn.CrossEntropyLoss()
    inputs = torch.randn(3, n_classes, requires_grad=True)
    targets = torch.empty(3, dtype=torch.long).random_(n_classes)
    output = loss(inputs, targets)
    
    focal_loss = FocalLossMod(alpha=1., gamma=0)
    output_focal = focal_loss(inputs, targets)

    print(f'test 1 difference {output_focal-output}')

    x = torch.rand(12800,n_classes)*random.randint(1,10)
    l = torch.empty(12800, dtype=torch.long).random_(n_classes)

    output0 = FocalLossMod(alpha=0.999999, gamma=0)(x,l)
    output1 = nn.CrossEntropyLoss()(x,l)
    
    print(f'test 2 difference: {output1.item() - output0.item()}')

    output0 = FocalLossMod(alpha=1., gamma=0, ignore_index=ignore_index)(x,l)
    output1 = nn.CrossEntropyLoss(ignore_index=ignore_index)(x,l)
    
    print(f'test 3 difference: {output1.item() - output0.item()}')

    alpha = 0.25
    gamma = 2
    output0 = FocalLossMod(alpha=alpha, gamma=gamma, reduction='mean')(x,l)
    output1 = kornia.losses.focal_loss(x, l, alpha, gamma, reduction='mean')
    
    print(f'test 4 difference: {output1.item() - output0.item()}')

    # multi dimensional
    x = torch.rand(32, n_classes, 224 ,224)*random.randint(1,10)
    l = torch.empty(32, 224, 224, dtype=torch.long).random_(n_classes)

    output0 = FocalLossMod(alpha=1., gamma=0, ignore_index=ignore_index)(x,l)
    output1 = nn.CrossEntropyLoss(ignore_index=3)(x,l)
    
    print(f'test 5 difference: {output1.item() - output0.item()}')
    
