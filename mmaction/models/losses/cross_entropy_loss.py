# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from mmaction.registry import MODELS
from .base import BaseWeightedLoss


@MODELS.register_module()
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probability distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Defaults to None.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        if cls_score.size() == label.size():
            # calculate loss for soft label
            # print('soft label')
            # print('cls_score:', cls_score)
            # print('label:', label)
            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                loss_cls = loss_cls.mean()
        else:
            # calculate loss for hard label
            # print('hard label')
            # print('cls_score:', cls_score)  # cls_score: tensor([[ 0.6804, -0.5817,  0.5114,  2.4573, -1.0788, -0.3206,  0.1546, -1.0031,
                                           #   -0.2245,  0.4871, -1.0211, -0.1635, -0.2379],
                                           #  [ 0.4942, -0.4930,  1.0297,  0.0996, -1.0854,  0.0908,  0.5750, -0.8472,
                                           #   -0.0659,  0.8533, -1.1168,  0.0570, -0.3476],
                                           #  [ 0.2663, -0.7142,  0.0531,  0.0825, -0.9067,  1.0241,  0.4129, -0.5612,
                                           #    1.3992,  0.5599, -1.0390,  1.1355, -0.8737],
                                           #  [ 0.5956, -0.5400,  0.8914,  0.3879, -0.8952, -0.0573,  0.2179, -0.6920,
                                           #    0.2763,  0.6272, -0.8259, -0.2360,  0.1491],
                                           #  [-0.7531, -0.9480, -0.9167, -0.7423,  1.7028, -0.7063, -0.4561,  2.4574,
                                           #    0.1509, -0.2781,  1.6213, -0.0817, -1.1917],
                                           #  [-0.7809, -0.8428, -0.6399, -0.7726,  2.1535, -1.1793, -0.3430,  3.2318,
                                           #   -0.6830, -0.1172,  2.2221, -0.4461, -1.2806],
                                           #  [ 0.7622, -0.0091,  0.4011,  0.5146, -1.4788,  0.6520,  0.2168, -0.7811,
                                           #    0.1489,  0.7584, -0.9804,  0.4133, -0.7688],
                                           #  [-0.3940, -0.6191, -0.8035, -0.4442,  1.3174, -0.4972, -0.4582,  2.0263,
                                           #    0.1547,  0.0223,  1.5662, -0.2369, -0.9165]], device='cuda:0',
                                           # grad_fn=<AddmmBackward>)
            # print('label:', label)  # label: tensor([7, 6, 7, 5, 9, 8, 3, 3], device='cuda:0')
            if self.class_weight is not None:
                assert 'weight' not in kwargs, \
                    "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)
            loss_cls = F.cross_entropy(cls_score, label, **kwargs)

        return loss_cls
        # return "aaa"


@MODELS.register_module()
class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Defaults to None.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        if self.class_weight is not None:
            assert 'weight' not in kwargs, "The key 'weight' already exists."
            kwargs['weight'] = self.class_weight.to(cls_score.device)
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, label,
                                                      **kwargs)
        return loss_cls


@MODELS.register_module()
class CBFocalLoss(BaseWeightedLoss):
    """Class Balanced Focal Loss. Adapted from https://github.com/abhinanda-
    punnakkal/BABEL/. This loss is used in the skeleton-based action
    recognition baseline for BABEL.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        samples_per_cls (list[int]): The number of samples per class.
            Defaults to [].
        beta (float): Hyperparameter that controls the per class loss weight.
            Defaults to 0.9999.
        gamma (float): Hyperparameter of the focal loss. Defaults to 2.0.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 samples_per_cls: List[int] = [],
                 beta: float = 0.9999,
                 gamma: float = 2.) -> None:
        super().__init__(loss_weight=loss_weight)
        self.samples_per_cls = samples_per_cls
        self.beta = beta
        self.gamma = gamma
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)
        self.weights = weights
        self.num_classes = len(weights)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        weights = torch.tensor(self.weights).float().to(cls_score.device)
        label_one_hot = F.one_hot(label, self.num_classes).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(label_one_hot.shape[0], 1) * label_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.num_classes)

        BCELoss = F.binary_cross_entropy_with_logits(
            input=cls_score, target=label_one_hot, reduction='none')

        modulator = 1.0
        if self.gamma:
            modulator = torch.exp(-self.gamma * label_one_hot * cls_score -
                                  self.gamma *
                                  torch.log(1 + torch.exp(-1.0 * cls_score)))

        loss = modulator * BCELoss
        weighted_loss = weights * loss

        focal_loss = torch.sum(weighted_loss)
        focal_loss /= torch.sum(label_one_hot)

        return focal_loss
