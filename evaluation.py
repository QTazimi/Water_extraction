import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)
        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


# loss = DiceLoss()
# predict = torch.tensor([[1, 0, 1], [1, 1, 0]])
# target = torch.tensor([[1, 0, 0], [0, 1, 1]])
# score = loss(predict, target)
# print(score)

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    Inter = SR * GT
    TP = torch.sum(Inter)
    TP_FN = torch.sum(GT)

    SE = float(TP)/(float(TP_FN) + 1e-6)
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    Inter = SR * GT
    TP = torch.sum(Inter)
    FP = torch.sum(SR) - TP

    FN = torch.sum(GT) - TP
    TN = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3) - FN - TP - FP


    SP = float(TN)/(float(TN + FP) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    Inter = SR * GT
    TP = torch.sum(Inter)
    TP_FP = torch.sum(SR)

    PC = float(TP)/(float(TP_FP) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    Inter = SR * GT
    Inter = torch.sum(Inter)
    Union = torch.sum(SR)+torch.sum(GT)-Inter
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)
    Inter = SR * GT
    Inter = torch.sum(Inter)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC


def get_iou(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    Inter = SR * GT
    TP = torch.sum(Inter)
    FP = torch.sum(SR) - TP

    FN = torch.sum(GT) - TP

    iou = float(TP) / (float(TP + FN + FP) + 1e-6)

    return iou


def get_FWiou(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TN : True Negative
    # FP : False Positive
    Inter = SR * GT
    TP = torch.sum(Inter)
    FP = torch.sum(SR) - TP

    FN = torch.sum(GT) - TP
    TN = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3) - FN - TP - FP

    a = float(TP + FN) / float(TP + FP + FN + TN + 1e-6)
    b = float(TP) / float(TP + FP + FN + 1e-6)
    fwiou = a * b

    return fwiou

