# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms.functional as TF

from mmaction.registry import MODELS
from mmaction.utils import OptSampleList
from .base import BaseRecognizer

import copy
import psutil

@MODELS.register_module()
class Recognizer3D(BaseRecognizer):
    """3D recognizer model framework."""

    def extract_feat(self,
                     inputs: Tensor,
                     inputs1: Tensor,
                     inputs2: Tensor,
                     inputs3: Tensor,
                     stage: str = 'neck',
                     data_samples: OptSampleList = None,
                     test_mode: bool = False) -> tuple:
        """Extract features of different stages.

        Args:
            inputs (torch.Tensor): The input data.
            stage (str): Which stage to output the feature.
                Defaults to ``'neck'``.
            data_samples (list[:obj:`ActionDataSample`], optional): Action data
                samples, which are only needed in training. Defaults to None.
            test_mode (bool): Whether in test mode. Defaults to False.

        Returns:
                torch.Tensor: The extracted features.
                dict: A dict recording the kwargs for downstream
                    pipeline. These keys are usually included:
                    ``loss_aux``.
        """

        # Record the kwargs required by `loss` and `predict`
        # print('111111111')
        loss_predict_kwargs = dict()

        num_segs = inputs.shape[1]
        # [N, num_crops, C, T, H, W] ->
        # [N * num_crops, C, T, H, W]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        

        # Check settings of test
        if test_mode:
            inputs = inputs.view((-1, ) + inputs.shape[2:])
            if self.test_cfg is not None:
                loss_predict_kwargs['fcn_test'] = self.test_cfg.get(
                    'fcn_test', False)
            if self.test_cfg is not None and self.test_cfg.get(
                    'max_testing_views', False):
                max_testing_views = self.test_cfg.get('max_testing_views')
                assert isinstance(max_testing_views, int)

                total_views = inputs.shape[0]
                assert num_segs == total_views, (
                    'max_testing_views is only compatible '
                    'with batch_size == 1')
                view_ptr = 0
                feats = []
                while view_ptr < total_views:
                    batch_imgs = inputs[view_ptr:view_ptr + max_testing_views]
                    feat = self.backbone(batch_imgs)
                    if self.with_neck:
                        feat, _ = self.neck(feat)
                    feats.append(feat)
                    view_ptr += max_testing_views

                def recursively_cat(feats):
                    # recursively traverse feats until it's a tensor,
                    # then concat
                    out_feats = []
                    for e_idx, elem in enumerate(feats[0]):
                        batch_elem = [feat[e_idx] for feat in feats]
                        if not isinstance(elem, torch.Tensor):
                            batch_elem = recursively_cat(batch_elem)
                        else:
                            batch_elem = torch.cat(batch_elem)
                        out_feats.append(batch_elem)

                    return tuple(out_feats)

                if isinstance(feats[0], tuple):
                    x = recursively_cat(feats)
                else:
                    x = torch.cat(feats)
            else:
                x = self.backbone(inputs)
                if self.with_neck:
                    x, _ = self.neck(x)

            return x, loss_predict_kwargs
        else:
            # Return features extracted through backbone
            # print('input.shape:', inputs.shape) # inputs.shape: torch.Size([16, 3, 16, 224, 224])
            # 遍历批量中的每个样本
            # sample = inputs[0]
            # sample = sample.permute(1, 2, 3, 0)
            # for i in range(sample.size(0)):
            #     sample1 = sample[i].permute(2, 0, 1)
            #     img = TF.to_pil_image(sample1)
            #     # 保存图像
            #     img.save(f'/home/zhouzhuo/project/mmaction2/image/frame_{i}.jpg')  # 保存当前帧图像为文件

            # print(sdwasd)
            # memory_info1 = psutil.virtual_memory()
            # used_memory1 = memory_info1.used
            # print("Used Memory1:", used_memory1)
            
            inputs = inputs.view((-1, ) + inputs.shape[2:])
            inputs1 = inputs1.view((-1, ) + inputs1.shape[2:])
            inputs2 = inputs2.view((-1, ) + inputs2.shape[2:])
            inputs3 = inputs3.view((-1, ) + inputs3.shape[2:])
            # print('inputs1:', inputs1)

            # memory_info2 = psutil.virtual_memory()
            # used_memory2 = memory_info2.used
            # print("Used Memory2:", used_memory2)

            network = self.backbone
            network1 = self.backbone
            network2 = self.backbone
            network3 = self.backbone

            # memory_info3 = psutil.virtual_memory()
            # used_memory3 = memory_info3.used
            # print("Used Memory3:", used_memory3)

            # inputs = 'sqawr'
            x = network(inputs)
            view1_x = network(inputs1)
            view2_x = network(inputs2)
            view3_x = network(inputs3)
            view1_y = network1(inputs1)
            view2_y = network2(inputs2)
            view3_y = network3(inputs3)

            # memory_info4 = psutil.virtual_memory()
            # used_memory4 = memory_info4.used
            # print("Used Memory1:", used_memory4)

            # view1_x = 0
            # view2_x = 0
            # print('view1_x:', view1_x)
            del inputs, inputs1, inputs2, inputs3
            # 获取内存使用情况
            # memory_info = psutil.virtual_memory()

            # # 获取物理内存总量（单位：字节）
            # total_memory = memory_info.total

            # # 获取可用内存量（单位：字节）
            # available_memory = memory_info.available

            # # 获取已使用的内存量（单位：字节）
            # used_memory = memory_info.used

            # # 获取空闲内存量（单位：字节）
            # free_memory = memory_info.free

            # # 打印内存使用情况
            # print("Total Memory:", total_memory)
            # print("Available Memory:", available_memory)
            # print("Used Memory:", used_memory)
            # print("Free Memory:", free_memory)

            # print('x:', x)
            # print('stage:', stage)  # stage: neck
            if stage == 'backbone':
                # print('stage == backbone')  # 未进入
                return x, loss_predict_kwargs

            loss_aux = dict()
            # print('self.with_neck:', self.with_neck)  # self.with_neck: False
            if self.with_neck:
                # print('stage == self.with_neck:')  # 未进入
                x, loss_aux = self.neck(x, data_samples=data_samples)

            # Return features extracted through neck
            loss_predict_kwargs['loss_aux'] = loss_aux
            if stage == 'neck':
                # print('stage == neck')  # 进入
                # x = 'asedw'
                # return x, loss_predict_kwargs
                return x, view1_x, view2_x, view3_x, view1_y, view2_y, view3_y, loss_predict_kwargs

            # Return raw logits through head.
            if self.with_cls_head and stage == 'head':
                x = self.cls_head(x, **loss_predict_kwargs)
                # print('loss_predict_kwargs------:', loss_predict_kwargs)  # 未输出
                return x, loss_predict_kwargs
