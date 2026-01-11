# Copyright (c) OpenMMLab. All rights reserved.
import inspect
import warnings
from abc import ABCMeta, abstractmethod
import psutil

import torch
import torch.nn as nn
from torch.nn import functional as F
from mmengine.model import BaseModel, merge_dict

from mmaction.registry import MODELS
from mmaction.utils import (ConfigType, ForwardResults, OptConfigType,
                            OptSampleList, SampleList)

import numpy as np
from sklearn.manifold import TSNE

from mmaction.models.losses import DistillationKernel
from mmaction.models.losses.KnowledgeDistillationLoss import KnowledgeDistillationLoss

# class guide():
#     def __init__(self, r, r1, b, c, t):
#         super(guide, self).__init__()
#         self.r = r.cuda()
#         self.r1 = r1.cuda()
#         self.b = b
#         self.c = c
#         self.t = t
#         # nn.Sequential()创建一个容器
#         self.glo_fc = nn.Sequential(nn.Linear(self.c, self.c),
#                                     nn.BatchNorm1d(self.c),
#                                     nn.ReLU()).cuda()

#         self.corr_atte = nn.Sequential(
#             # nn.Conv2d(2048 + 1024, 1024, 1, 1, bias=False),
#             nn.Conv3d(self.c + self.c, self.c, 1, 1, bias=False),
#             nn.BatchNorm3d(self.c),
#             nn.Conv3d(self.c, 256, 1, 1, bias=False),
#             nn.BatchNorm3d(256),
#             nn.ReLU(),
#             nn.Conv3d(256, 1, 1, 1, bias=False),
#             nn.BatchNorm3d(1),
#         ).cuda()

#     def __iter__(self):
#         return self

#     def __next__(self):
#         r_glo = self.r.mean(dim=-1).mean(dim=-1).mean(dim=-1)  # r_glo = tensor<(32, 2048), float32, cuda:0, grad>
#         # r1_glo = self.r1.mean(dim=-1).mean(dim=-1).mean(dim=1)  # r1_glo = tensor<(32, 2048), float32, cuda:0, grad>
#         r_glo = r_glo.cuda()
#         # r1_glo = r1_glo.cuda()
#         # view():改变形状 contiguous():tensor开辟新内存空间
#         '''
#         glo = self.glo_fc(r_glo).view(self.b, 1, 1024, 1, 1).contiguous().expand(self.b, self.t, 1024, 16, 8).contiguous().view(self.b * self.t,
#                                                                                                                  1024,
#                                                                                                                  16, 8)
#         # glo = tensor<(32, 1024, 16, 8), float32, cuda:0, grad>
#         glo1 = self.glo_fc(r1_glo).view(self.b, 1, 1024, 1, 1).contiguous().expand(self.b, self.t, 1024, 16, 8).contiguous().view(
#             self.b * self.t,
#             1024,
#             16, 8)
#         # glo1 = tensor<(32, 1024, 16, 8), float32, cuda:0, grad>
#         '''

#         glo = self.glo_fc(r_glo).view(self.b, self.c, 1, 1, 1).contiguous().expand(self.b, self.c, self.t, 7, 7)
#         # print('glo.shape()', glo.shape) # torch.Size([4, 2048, 16, 7, 7])

#         # print('self.r1.shape()1:', self.r1.shape)  # self.r1.shape()1: torch.Size([4, 432, 16, 7, 7])
#         r1 = self.r1.mean(dim=-1).mean(dim=-1).mean(dim=-1)
#         r1 = self.glo_fc(r1).view(self.b, self.c, 1, 1, 1).contiguous().expand(self.b, self.c, self.t, 7, 7)
#         # print('r1.shape()', r1.shape) # r1.shape() torch.Size([4, 2048, 16, 7, 7])
#         r_corr = torch.cat((r1, glo), dim=1)  # r_corr = tensor<(32, 1, 2048, 4, 4), float32, cuda:0, grad>
#         # print('r_corr.shape()', r_corr.shape) # r_corr.shape() torch.Size([4, 4096, 16, 7, 7])

#         # r_corr = r_corr.permute(0, 2, 1, 3, 4)  # r_corr = tensor<(32, 2048, 1, 4, 4), float32, cuda:0, grad>
#         corr_map = self.corr_atte(r_corr)  # corr_map = tensor<(32, 1, 1, 4, 4), float32, cuda:0, grad>
#         # corr_map = F.sigmoid(corr_map).view(self.b * self.t, 1, 4, 4).contiguous()
#         corr_map = F.sigmoid(corr_map).view(self.b, 1, self.t, 7, 7).contiguous()  # torch.Size([32, 1, 1, 4, 4])
#         # print('corr_map.size()', corr_map.size())
#         # r_uncorr = tensor<(32, 2048, 1, 4, 4)
#         r_corr = r1 * corr_map
#         r_uncorr = r1 * (1 - corr_map)

#         return r_corr, r_uncorr

class guide():
    def __init__(self, r, r1, b, c, t):
        super(guide, self).__init__()
        self.r = r.cuda()
        self.r1 = r1.cuda()
        self.b = b
        self.c = c
        self.t = t
        
        # 定义全局特征提取器
        self.global_fc = nn.Sequential(
            nn.Linear(self.c, self.c),
            nn.BatchNorm1d(self.c),
            nn.ReLU()
        ).cuda()

        self.corr_atte = nn.Sequential(
            nn.Conv3d(2 * self.c, self.c, 1, 1, bias=False),
            nn.BatchNorm3d(self.c),
            nn.Conv3d(self.c, 256, 1, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 1, 1, 1, bias=False),
            nn.BatchNorm3d(1),
        ).cuda()

    def __iter__(self):
        return self

    def __next__(self):
        r_glo = self.global_fc(self.r.mean(dim=-1).mean(dim=-1).mean(dim=-1))
        r1_glo = self.global_fc(self.r1.mean(dim=-1).mean(dim=-1).mean(dim=-1))

        # 将全局特征与视角特征合并
        combined_features = torch.cat((r1_glo.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), r_glo.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)), dim=1)

        # 将合并后的特征传入相关性注意力模块
        corr_map = self.corr_atte(combined_features)

        # 计算权重
        weight = F.sigmoid(corr_map)

        # 根据权重解耦视角特征
        r1_decoupled = self.r1 * weight
        r_decoupled = self.r1 * (1 - weight)

        return r_decoupled, r1_decoupled


class BaseRecognizer(BaseModel, metaclass=ABCMeta):
    """Base class for recognizers.

    Args:
        backbone (Union[ConfigDict, dict]): Backbone modules to
            extract feature.
        cls_head (Union[ConfigDict, dict], optional): Classification head to
            process feature. Defaults to None.
        neck (Union[ConfigDict, dict], optional): Neck for feature fusion.
            Defaults to None.
        train_cfg (Union[ConfigDict, dict], optional): Config for training.
            Defaults to None.
        test_cfg (Union[ConfigDict, dict], optional): Config for testing.
            Defaults to None.
        data_preprocessor (Union[ConfigDict, dict], optional): The pre-process
           config of :class:`ActionDataPreprocessor`.  it usually includes,
            ``mean``, ``std`` and ``format_shape``. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 cls_head: OptConfigType = None,
                 neck: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None) -> None:
        if data_preprocessor is None:
            # This preprocessor will only stack batch data samples.
            data_preprocessor = dict(type='ActionDataPreprocessor')

        super(BaseRecognizer,
              self).__init__(data_preprocessor=data_preprocessor)

        def is_from(module, pkg_name):
            # check whether the backbone is from pkg
            model_type = module['type']
            if isinstance(model_type, str):
                return model_type.startswith(pkg_name)
            elif inspect.isclass(model_type) or inspect.isfunction(model_type):
                module_name = model_type.__module__
                return pkg_name in module_name
            else:
                raise TypeError(
                    f'Unsupported type of module {type(module["type"])}')

        # Record the source of the backbone.
        self.backbone_from = 'mmaction2'
        if is_from(backbone, 'mmcls.'):
            try:
                # Register all mmcls models.
                import mmcls.models  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install mmcls to use this backbone.')
            self.backbone = MODELS.build(backbone)
            self.backbone_from = 'mmcls'
        elif is_from(backbone, 'mmpretrain.'):
            try:
                # Register all mmpretrain models.
                import mmpretrain.models  # noqa: F401
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    'Please install mmpretrain to use this backbone.')
            self.backbone = MODELS.build(backbone)
            self.backbone_from = 'mmpretrain'
        elif is_from(backbone, 'torchvision.'):
            try:
                import torchvision.models
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install torchvision to use this '
                                  'backbone.')
            self.backbone_from = 'torchvision'
            self.feature_shape = backbone.pop('feature_shape', None)
            backbone_type = backbone.pop('type')
            if isinstance(backbone_type, str):
                backbone_type = backbone_type[12:]
                self.backbone = torchvision.models.__dict__[backbone_type](
                    **backbone)
            else:
                self.backbone = backbone_type(**backbone)
            # disable the classifier
            self.backbone.classifier = nn.Identity()
            self.backbone.fc = nn.Identity()
        elif is_from(backbone, 'timm.'):
            # currently, only support use `str` as backbone type
            try:
                import timm
            except (ImportError, ModuleNotFoundError):
                raise ImportError('Please install timm>=0.9.0 to use this '
                                  'backbone.')
            self.backbone_from = 'timm'
            self.feature_shape = backbone.pop('feature_shape', None)
            # disable the classifier
            backbone['num_classes'] = 0
            backbone_type = backbone.pop('type')
            if isinstance(backbone_type, str):
                backbone_type = backbone_type[5:]
                self.backbone = timm.create_model(backbone_type, **backbone)
            else:
                raise TypeError(
                    f'Unsupported timm backbone type: {type(backbone_type)}')
        else:
            self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if cls_head is not None:
            self.cls_head = MODELS.build(cls_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @abstractmethod
    def extract_feat(self, inputs: torch.Tensor, **kwargs) -> ForwardResults:
        """enter recognizer3d.py"""
        """Extract features from raw inputs."""

    @property
    def with_neck(self) -> bool:
        """bool: whether the recognizer has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_cls_head(self) -> bool:
        """bool: whether the recognizer has a cls_head"""
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def init_weights(self) -> None:
        """Initialize the model network weights."""
        if self.backbone_from in ['torchvision', 'timm']:
            warnings.warn('We do not initialize weights for backbones in '
                          f'{self.backbone_from}, since the weights for '
                          f'backbones in {self.backbone_from} are initialized '
                          'in their __init__ functions.')

            def fake_init():
                pass

            # avoid repeated initialization
            self.backbone.init_weights = fake_init
        super().init_weights()

    # def loss(self, inputs: torch.Tensor, data_samples: SampleList,
    #          **kwargs) -> dict:
        
    def softmax(self, w, t=1.0, axis=None):
        w = np.array(w) / t
        e = np.exp(w - np.amax(w, axis=axis, keepdims=True))
        dist = e / np.sum(e, axis=axis, keepdims=True)
        return dist
    
    def loss(self, data, data1, data2, data3, data_samples: SampleList,
        **kwargs) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): Raw Inputs of the recognizer.
                These should usually be mean centered and std scaled.
            data_samples (List[``ActionDataSample``]): The batch
                data samples. It usually includes information such
                as ``gt_labels``.

        Returns:
            dict: A dictionary of loss components.
        """

        # 获取内存使用情况
        # memory_info1 = psutil.virtual_memory()
        # used_memory1 = memory_info1.used
        # print("Used Memory1:", used_memory1)
        # print('data2:', data2)
        # vinputs1 = 'inputsinputs'
        inputs = data['inputs']
        data_samples = data['data_samples']
        inputs1 = data1['inputs']
        data_samples1 = data1['data_samples']
        inputs2 = data2['inputs']
        data_samples2 = data2['data_samples']
        inputs3 = data3['inputs']
        data_samples3 = data3['data_samples']

        # # 获取已使用的内存量（单位：字节）
        # memory_info2 = psutil.virtual_memory()
        # used_memory2 = memory_info2.used
        # print("Used Memory2:", used_memory2)

        # print('inputs:', inputs)
        # view1_feats : common feature
        # view1_feat: special feature
        feats, view1_feats, view2_feats, view3_feats, view1_feat, view2_feat, view3_feat, loss_kwargs = \
            self.extract_feat(inputs, inputs1, inputs2, inputs3,
                              data_samples=data_samples)
        # print('feats.shape:', feats.shape)  # feats.shape: torch.Size([16, 432, 16, 7, 7])

        # memory_info3 = psutil.virtual_memory()
        # used_memory3 = memory_info3.used
        # print("Used Memory3:", used_memory3)

        # guide
        b, c, t, h, w = view1_feat.size()
        # print('view1_feats.shape:', view1_feats.shape)
        # print('view1_feat.shape:', view1_feat.shape)
        a1, v1 = guide(view1_feats, view1_feat, b, c, t).__next__() # a,v
        # print('wr1.shape:', wr1.shape)  # wr1.shape: torch.Size([4, 2048, 16, 7, 7])
        # print('sr1.shape:', sr1.shape)  # wr1.shape: torch.Size([4, 2048, 16, 7, 7])
        a2, v2 = guide(view2_feats, view2_feat, b, c, t).__next__()
        # print('wr2.shape:', wr2.shape)  # wr1.shape: torch.Size([4, 2048, 16, 7, 7])
        # print('sr2.shape:', sr2.shape)  # wr1.shape: torch.Size([4, 2048, 16, 7, 7])
        a3, v3 = guide(view3_feats, view3_feat, b, c, t).__next__()
        # print('data_samples:', data_samples)
        # print('view1_data_samples:', view1_data_samples)
        # print('view2_data_samples:', view2_data_samples)
        # print('feats.shape:', feats.shape)  # torch.Size([2, 432, 16, 7, 7])
        # print('a1.shape:', a1.shape)  # torch.Size([2, 432, 16, 7, 7])
        # print('a2.shape:', a2.shape)  # torch.Size([2, 432, 16, 7, 7])
        # print('a3.shape:', a3.shape)  # torch.Size([2, 432, 16, 7, 7])
        
        # memory_info3 = psutil.virtual_memory()
        # used_memory3 = memory_info3.used
        # print("Used Memory3:", used_memory3)

        # loss_aux will be a empty dict if `self.with_neck` is False.
        loss_aux = loss_kwargs.get('loss_aux', dict())

        loss_cls, _ = self.cls_head.loss(feats, data_samples, **loss_kwargs)
        # print('loss_cls:', loss_cls)
        view1_loss_cls, view1_cls_scores = self.cls_head.loss(a1, data_samples1, **loss_kwargs)
        # print('view1_loss_cls:', view1_loss_cls)
        view2_loss_cls, view2_cls_scores = self.cls_head.loss(a2, data_samples2, **loss_kwargs)
        # print('view2_loss_cls:', view2_loss_cls)
        view3_loss_cls, view3_cls_scores = self.cls_head.loss(a3, data_samples3, **loss_kwargs)
        # print('view3_loss_cls:', view3_loss_cls)
        
        lce_all = loss_cls + view1_loss_cls + view2_loss_cls + view3_loss_cls

        # memory_info4 = psutil.virtual_memory()
        # used_memory4 = memory_info4.used
        # print("Used Memory4:", used_memory4)

        # lkd_a1_a2 = KnowledgeDistillationLoss().forward(view1_cls_scores, view2_cls_scores)
        lkd_a1_a2 = KnowledgeDistillationLoss().forward(view2_cls_scores, view1_cls_scores)
        # print('lkd_a1_a2:', lkd_a1_a2)
        lkd_a1_a3 = KnowledgeDistillationLoss().forward(view1_cls_scores, view3_cls_scores)
        # # print('lkd_a1_a3:', lkd_a1_a3)
        lkd_a2_a3 = KnowledgeDistillationLoss().forward(view2_cls_scores, view3_cls_scores)
        # print('lkd_a2_a3:', lkd_a2_a3)

        lkd_all = lkd_a1_a2 + lkd_a1_a3 + lkd_a2_a3
        # lkd_all = lkd_a1_a2

        # memory_info5 = psutil.virtual_memory()
        # used_memory5 = memory_info5.used
        # print("Used Memory5:", used_memory5)

        kl_loss = nn.KLDivLoss(reduction='batchmean')

        B, C, T, H, W = v1.size()
        a1_flat = a1.view(B * T, C * H * W)
        a2_flat = a2.view(B * T, C * H * W)
        a3_flat = a3.view(B * T, C * H * W)

        # memory_info6 = psutil.virtual_memory()
        # used_memory6 = memory_info6.used
        # print("Used Memory6:", used_memory6)

        a1_probs = torch.softmax(a1_flat, dim=1)
        a2_probs = torch.softmax(a2_flat, dim=1)
        a3_probs = torch.softmax(a3_flat, dim=1)

        # memory_info7 = psutil.virtual_memory()
        # used_memory7 = memory_info7.used
        # print("Used Memory7:", used_memory7)

        view1_feats_flat = view1_feats.view(B * T, C * H * W)
        view2_feats_flat = view2_feats.view(B * T, C * H * W)
        view3_feats_flat = view3_feats.view(B * T, C * H * W)

        # memory_info8 = psutil.virtual_memory()
        # used_memory8 = memory_info8.used
        # print("Used Memory8:", used_memory8)

        view1_feats_probs = torch.softmax(view1_feats_flat, dim=1)
        view2_feats_probs = torch.softmax(view2_feats_flat, dim=1)
        view3_feats_probs = torch.softmax(view3_feats_flat, dim=1)

        # memory_info9 = psutil.virtual_memory()
        # used_memory9 = memory_info9.used
        # print("Used Memory9:", used_memory9)

        kl_a1_a1 = kl_loss(torch.log(a1_probs), view1_feats_probs)
        kl_a2_a2 = kl_loss(torch.log(a2_probs), view2_feats_probs)
        kl_a3_a3 = kl_loss(torch.log(a3_probs), view3_feats_probs)

        kl_a_all = kl_a1_a1 + kl_a2_a2 + kl_a3_a3
        # kl_a_all = kl_a1_a1 + kl_a2_a2

        # memory_info10 = psutil.virtual_memory()
        # used_memory10 = memory_info10.used
        # print("Used Memory10:", used_memory10)

        B, C, T, H, W = v1.size()
        v1_flat = v1.view(B * T, C * H * W)
        v2_flat = v2.view(B * T, C * H * W)
        v3_flat = v3.view(B * T, C * H * W)

        # memory_info11 = psutil.virtual_memory()
        # used_memory11 = memory_info11.used
        # print("Used Memory11:", used_memory11)

        probs1 = torch.softmax(v1_flat, dim=1)
        probs2 = torch.softmax(v2_flat, dim=1)
        probs3 = torch.softmax(v3_flat, dim=1)

        # memory_info12 = psutil.virtual_memory()
        # used_memory12 = memory_info12.used
        # print("Used Memory12:", used_memory12)

        # kl_v1_v2 = kl_loss(torch.log(probs1), probs2)
        kl_v1_v2 = kl_loss(torch.log(probs2), probs1)

        kl_v1_v3 = kl_loss(torch.log(probs1), probs3)
        # kl_v2_v3 = kl_loss(torch.log(probs2), probs3)

        # kl_v_all = -(kl_v1_v2 + kl_v1_v3 + kl_v2_v3)
        kl_v_all = kl_v1_v2 + kl_v1_v3


        v1_gen = a1_flat + v1_flat
        probs_v1_gen = torch.softmax(v1_gen, dim=1)
        v2_gen = a2_flat + v2_flat
        probs_v2_gen = torch.softmax(v2_gen, dim=1)
        v3_gen = a3_flat + v3_flat
        probs_v3_gen = torch.softmax(v3_gen, dim=1)

        B, C, T, H, W = view1_feat.size()
        view1_flat = view1_feat.view(B * T, C * H * W)
        view2_flat = view2_feat.view(B * T, C * H * W)
        view3_flat = view3_feat.view(B * T, C * H * W)

        probs_v1 = torch.softmax(view1_flat, dim=1)
        probs_v2 = torch.softmax(view2_flat, dim=1)
        probs_v3 = torch.softmax(view3_flat, dim=1)

        l_gen1 = kl_loss(torch.log(probs_v1_gen), probs_v1)
        l_gen2 = kl_loss(torch.log(probs_v2_gen), probs_v2)
        l_gen3 = kl_loss(torch.log(probs_v3_gen), probs_v3)

        l_gen = l_gen1 + l_gen2 + l_gen3 
        # memory_info13 = psutil.virtual_memory()
        # used_memory13 = memory_info13.used
        # print("Used Memory13:", used_memory13)

        # loss_all = loss_cls['loss_cls'] + lkd_all +  0.01 * (kl_a_all + kl_v_all)
        loss_all = lce_all + l_gen +  0.01 * (kl_a_all + kl_v_all)

        loss_cls['loss_cls'] = loss_all
        # cls_list = torch.stack((view1_cls_scores, view2_cls_scores, view3_cls_scores))
        # feature_list= torch.stack((v1, v2, v3))


        # to_idx = [0, 1, 2]  # all modalities can be distilled from each other simultaneously
        # from_idx = [0, 1, 2]  # all modalities can be distilled from each other simultaneously
        # w_losses_low = [1, 10]
        # metric_low = 'l1'

        # model_distill_hetero = DistillationKernel(n_classes=13,
        #                                         hidden_size=50,
        #                                         gd_size=32,
        #                                         to_idx=to_idx, from_idx=from_idx,
        #                                         gd_prior=self.softmax([0, 0, 1, 0, 1, 1], 0.25),
        #                                         gd_reg=10,
        #                                         w_losses=w_losses_low,
        #                                         metric=metric_low,
        #                                         alpha=1 / 8)

        # # edges for hetero distill
        # edges_hetero, edges_origin_hetero = model_distill_hetero(cls_list, feature_list)

        # loss_reg_hetero, loss_logit_hetero, loss_repr_hetero = \
        #                 model_distill_hetero.distillation_loss(cls_list, feature_list, edges_hetero)
        # graph_distill_loss_hetero = 0.05 * (loss_logit_hetero + loss_repr_hetero + loss_reg_hetero)

        # print('loss_cls:', loss_cls)  # loss_cls: {'top1_acc': tensor(0.0625, device='cuda:0', dtype=torch.float64), 'top5_acc': tensor(0.6875, device='cuda:0', dtype=torch.float64), 'loss_cls': tensor(2.2830, device='cuda:0', grad_fn=<MulBackward0>)}
        # print('view1_loss_cls:', view1_loss_cls)  # loss_cls: {'top1_acc': tensor(0.0625, device='cuda:0', dtype=torch.float64), 'top5_acc': tensor(0.6875, device='cuda:0', dtype=torch.float64), 'loss_cls': tensor(2.2830, device='cuda:0', grad_fn=<MulBackward0>)}
        # print('view2_loss_cls:', view2_loss_cls)  # loss_cls: {'top1_acc': tensor(0.0625, device='cuda:0', dtype=torch.float64), 'top5_acc': tensor(0.6875, device='cuda:0', dtype=torch.float64), 'loss_cls': tensor(2.2830, device='cuda:0', grad_fn=<MulBackward0>)}
        # losses = merge_dict(loss_cls, loss_aux) 
        # losses = merge_dict(loss_cls, view1_loss_cls, view2_loss_cls, loss_aux)
        # losses = {'loss_cls': loss_cls, 'view1_loss_cls': view1_loss_cls, 'view2_loss_cls': view2_loss_cls, 'view3_loss_cls': view3_loss_cls, 'loss_aux': loss_aux}
        losses = merge_dict(loss_cls, loss_aux) 
        merged_losses = merge_dict(losses)

        # memory_info14 = psutil.virtual_memory()
        # used_memory14 = memory_info14.used
        # print("Used Memory14:", used_memory14)

        # del inputs, inputs1, inputs2, inputs3
        # del feats, view1_feats, view2_feats, view3_feats, view1_feat, view2_feat, view3_feat
        # del a1,v1,a2,v2,a3,v3
        # del view1_loss_cls, view1_cls_scores, view2_loss_cls, view2_cls_scores, view3_loss_cls, view3_cls_scores,
        # del a1_flat, a2_flat, a3_flat
        # del a1_probs, a2_probs, a3_probs
        # del view1_feats_flat, view2_feats_flat, view3_feats_flat
        # del view1_feats_probs, view2_feats_probs, view3_feats_probs
        # del v1_flat, v2_flat, v3_flat
        # del probs1, probs2, probs3
        # del lkd_a1_a2, lkd_a1_a3, lkd_a2_a3, lkd_all
        # del kl_a1_a1, kl_a2_a2, kl_a3_a3, kl_a_all
        # del kl_v1_v2, kl_v1_v3, kl_v2_v3, kl_v_all

        # memory_info15 = psutil.virtual_memory()
        # used_memory15 = memory_info15.used
        # print("Used Memory15:", used_memory15)
        
        print(sadwa)
        # print('merged_losses:', merged_losses)
        # merged_losses = "{'top1_acc': sdsd, 'top5_acc': tensor(1., device='cuda:1', dtype=torch.float64), 'loss_cls': tensor(3.9507, device='cuda:1', grad_fn=<AddBackward0>)}"
        return merged_losses

    def predict(self, data: torch.Tensor, data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (torch.Tensor): Raw Inputs of the recognizer.
                These should usually be mean centered and std scaled.
            data_samples (List[``ActionDataSample``]): The batch
                data samples. It usually includes information such
                as ``gt_labels``.

        Returns:
            List[``ActionDataSample``]: Return the recognition results.
            The returns value is ``ActionDataSample``, which usually contains
            ``pred_scores``. And the ``pred_scores`` usually contains
            following keys.

                - item (torch.Tensor): Classification scores, has a shape
                    (num_classes, )
        """
        inputs = data['inputs']
        data_samples = data['data_samples']
        inputs1 = {}
        inputs2 = {}
        inputs3 = {}
        # print('inputs', inputs)
        # print('SampleList', SampleList)
        feats, predict_kwargs = self.extract_feat(inputs, inputs1, inputs2, inputs3, test_mode=True)
        # print('feats:', feats)

        # print('feats.shape:', feats.shape)  # feats.shape: torch.Size([8, 432, 16, 7, 7])

        # data_samples = 'aaa'
        predictions = self.cls_head.predict(feats, data_samples,
                                            **predict_kwargs)
        # predictions = 'aaa'
        return predictions

    def _forward(self,
                 inputs: torch.Tensor,
                 stage: str = 'backbone',
                 **kwargs) -> ForwardResults:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            inputs (torch.Tensor): Raw Inputs of the recognizer.
            stage (str): Which stage to output the features.

        Returns:
            Union[tuple, torch.Tensor]: Features from ``backbone`` or ``neck``
            or ``head`` forward.
        """
        feats, _ = self.extract_feat(inputs, stage=stage)
        return feats

    def forward(self,
                inputs: torch.Tensor,
                inputs1: torch.Tensor,
                inputs2: torch.Tensor,
                inputs3: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes:

        - ``tensor``: Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - ``predict``: Forward and return the predictions, which are fully
        processed to a list of :obj:`ActionDataSample`.
        - ``loss``: Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[``ActionDataSample], optional): The
                annotation data of every samples. Defaults to None.
            mode (str): Return what kind of value. Defaults to ``tensor``.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of ``ActionDataSample``.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'tensor':
            # print('mode==tensor')
            return self._forward(inputs, **kwargs)
        if mode == 'predict': 
            # print('mode==predict')  # val使用
            return self.predict(inputs, data_samples, **kwargs)
        elif mode == 'loss':
            # print('mode==loss')  # train使用
            return self.loss(inputs, inputs1, inputs2, inputs3, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')
