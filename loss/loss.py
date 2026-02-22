import torch
import torch.nn.functional as F


class DiceLoss:
    "Dice loss for segmentation"

    def __init__(self,
                 axis: int = 1,  # Class axis
                 smooth: float = 1e-6,  # Helps with numerical stabilities in the IoU division
                 reduction: str = "sum",  # PyTorch reduction to apply to the output
                 square_in_union: bool = False  # Squares predictions to increase slope of gradients
                 ):
        self.axis = axis
        self.smooth = smooth
        self.reduction = reduction
        self.square_in_union = square_in_union

    def __call__(self, pred, targ):
        "One-hot encodes targ, then runs IoU calculation then takes 1-dice value"
        targ = self._one_hot(targ, pred.shape[self.axis])
        assert pred.shape == targ.shape, 'input and target dimensions differ, DiceLoss expects non one-hot targs'
        pred = self.activation(pred)
        sum_dims = list(range(2, len(pred.shape)))
        inter = torch.sum(pred * targ, dim=sum_dims)
        union = (torch.sum(pred ** 2 + targ, dim=sum_dims) if self.square_in_union
                 else torch.sum(pred + targ, dim=sum_dims))
        dice_score = (2. * inter + self.smooth) / (union + self.smooth)
        loss = 1 - dice_score
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

    @staticmethod
    def _one_hot(
            x,  # Non one-hot encoded targs
            classes: int,  # The number of classes
            axis: int = 1  # The axis to stack for encoding (class dimension)
    ):
        "Creates one binary mask per class"
        return torch.stack([torch.where(x == c, 1, 0) for c in range(classes)], axis=axis)

    def activation(self, x):
        "Activation function applied to model output"
        return F.softmax(x, dim=self.axis)

    def decodes(self, x):
        "Converts model output to target format"
        return x.argmax(dim=self.axis)


def compute_boundary_loss_with_text(pred_logit, target_mask, text_feat, threshold=0.5):
    pred_logit = pred_logit[:,1:,:,:]
    pred_mask = torch.sigmoid(pred_logit) > threshold
    pred_mask = pred_mask.float()
    target_mask = target_mask.float().unsqueeze(1)

    pred_grad = F.pad(torch.abs(pred_mask[:, :, 1:, :] - pred_mask[:, :, :-1, :]),(0,0,1,0),mode='constant',value=0) + F.pad(torch.abs(pred_mask[:, :, :, 1:] - pred_mask[:, :, :, :-1]),(1, 0, 0, 0), mode='constant', value=0)
    target_grad = F.pad(torch.abs(target_mask[:, :, 1:, :] - target_mask[:, :, :-1, :]),(0,0,1,0),mode='constant',value=0) + F.pad(torch.abs(target_mask[:, :, :, 1:] - target_mask[:, :, :, :-1]),(1, 0, 0, 0), mode='constant', value=0)
    #text_feat b,1,h,w
    # text_weight = text_feat.unsqueeze(-1).unsqueeze(-1)
    # text_weight = text_weight / torch.max(text_weight)
    text_weight = text_feat

    weighted_pred_grad = pred_grad * text_weight
    weighted_target_grad = target_grad * text_weight
    boundary_loss = F.mse_loss(weighted_pred_grad.float(), weighted_target_grad.float())

    return boundary_loss

class Loss():
    def __init__(self, weight=0.1):
        self.dice_loss = DiceLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.9, 1.1]).cuda())
        self.weight = weight
    def __call__(self, pred, targ,text):
        dice_loss = self.dice_loss(pred, targ)
        ce_loss = self.ce_loss(pred, targ)
        con_loss = compute_boundary_loss_with_text(pred,targ,text)
        return (1 - self.weight) * ce_loss + self.weight * dice_loss+0.2*con_loss