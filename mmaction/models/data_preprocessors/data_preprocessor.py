# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Tuple, Union

import torch
from mmengine.model import BaseDataPreprocessor, stack_batch

from mmaction.registry import MODELS
from mmaction.utils import SampleList


@MODELS.register_module()
class ActionDataPreprocessor(BaseDataPreprocessor):
    """Data pre-processor for action recognition tasks.

    Args:
        mean (Sequence[float or int], optional): The pixel mean of channels
            of images or stacked optical flow. Defaults to None.
        std (Sequence[float or int], optional): The pixel standard deviation
            of channels of images or stacked optical flow. Defaults to None.
        to_rgb (bool): Whether to convert image from BGR to RGB.
            Defaults to False.
        to_float32 (bool): Whether to convert data to float32.
            Defaults to True.
        blending (dict, optional): Config for batch blending.
            Defaults to None.
        format_shape (str): Format shape of input data.
            Defaults to ``'NCHW'``.
    """

    def __init__(self,
                 mean: Optional[Sequence[Union[float, int]]] = None,
                 std: Optional[Sequence[Union[float, int]]] = None,
                 to_rgb: bool = False,
                 to_float32: bool = True,
                 blending: Optional[dict] = None,
                 format_shape: str = 'NCHW') -> None:
        super().__init__()
        self.to_rgb = to_rgb
        self.to_float32 = to_float32
        self.format_shape = format_shape

        if mean is not None:
            assert std is not None, 'To enable the normalization in ' \
                                    'preprocessing, please specify both ' \
                                    '`mean` and `std`.'
            # Enable the normalization in preprocessing.
            self._enable_normalize = True
            if self.format_shape == 'NCHW':
                normalizer_shape = (-1, 1, 1)
            elif self.format_shape in ['NCTHW', 'MIX2d3d']:
                normalizer_shape = (-1, 1, 1, 1)
            else:
                raise ValueError(f'Invalid format shape: {format_shape}')

            self.register_buffer(
                'mean',
                torch.tensor(mean, dtype=torch.float32).view(normalizer_shape),
                False)
            self.register_buffer(
                'std',
                torch.tensor(std, dtype=torch.float32).view(normalizer_shape),
                False)
        else:
            self._enable_normalize = False

        if blending is not None:
            self.blending = MODELS.build(blending)
        else:
            self.blending = None

    def forward(self,
                data_all: Union[dict, Tuple[dict]],
                training: bool = False) -> Union[dict, Tuple[dict]]:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation based on ``BaseDataPreprocessor``.

        Args:
            data (dict or Tuple[dict]): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict or Tuple[dict]: Data in the same format as the model input.
        """
        # print('data111:', data) # 输出tesnor
        # print('data:', data)
        data = data_all[0]
        data = self.cast_data(data)
        data1 = data_all[1]
        data1 = self.cast_data(data1)
        data2 = data_all[2]
        data2 = self.cast_data(data2)
        data3 = data_all[3]
        data3 = self.cast_data(data3)
        if isinstance(data, dict):
            # print('1111111111')  # 输出
            # data = 'sdwar'
            return self.forward_onesample(data, data1, data2, data3, training=training)
        elif isinstance(data, tuple):
            # print('222222222222')  # 无输出
            outputs = []
            for data_sample in data:
                output = self.forward_onesample(data_sample, training=training)
                outputs.append(output)
            return tuple(outputs)
        else:
            raise TypeError(f'Unsupported data type: {type(data)}!')

    def forward_onesample(self, data, data1, data2, data3, training: bool = False) -> dict:
        """Perform normalization, padding, bgr2rgb conversion and batch
        augmentation on one data sample.

        Args:
            data (dict): data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        
        inputs, data_samples = data['inputs'], data['data_samples']
        # print('inputs:', inputs.shape)  # tensor list
        # print('data_samples:', data_samples.shape)  # label list
        # print('inputs:', inputs)  
        # print('data_samples:', data_samples)
        inputs, data_samples = self.preprocess(inputs, data_samples, training)
        data['inputs'] = inputs
        data['data_samples'] = data_samples
        # print('data:', data)
        # data = {'inputs': (), 'data_samples': []}
        
        inputs1, data_samples1 = data1['inputs'], data1['data_samples']
        inputs1, data_samples1 = self.preprocess(inputs1, data_samples1, training)
        data1['inputs'] = inputs1
        data1['data_samples'] = data_samples1

        inputs2, data_samples2 = data2['inputs'], data2['data_samples']
        inputs2, data_samples2 = self.preprocess(inputs2, data_samples2, training)
        data2['inputs'] = inputs2
        data2['data_samples'] = data_samples2

        inputs3, data_samples3 = data3['inputs'], data3['data_samples']
        inputs3, data_samples3 = self.preprocess(inputs3, data_samples3, training)
        data3['inputs'] = inputs3
        data3['data_samples'] = data_samples3

        # print('data:', data)
        # print('data1:', data1)
        # print('data2:', data2)
        # print('data3:', data3)
        # print('data:', data.shape), data1, data2, data3
        # data = 'sawad' 

        return data, data1, data2, data3

    def preprocess(self,
                   inputs: List[torch.Tensor],
                   data_samples: SampleList,
                   training: bool = False) -> Tuple:
        # --- Pad and stack --
        batch_inputs = stack_batch(inputs)

        if self.format_shape == 'MIX2d3d':
            if batch_inputs.ndim == 4:
                # print('44444444')  # 无输出
                format_shape, view_shape = 'NCHW', (-1, 1, 1)
            else:
                # print('55555555')  # 无输出
                format_shape, view_shape = 'NCTHW', None
        else:
            # print('6666666')  # 先输出
            format_shape, view_shape = self.format_shape, None

        # ------ To RGB ------
        if self.to_rgb:
            if format_shape == 'NCHW':
                # print('7777777')  # 无输出
                batch_inputs = batch_inputs[..., [2, 1, 0], :, :]
            elif format_shape == 'NCTHW':
                # print('8888888')  # 无输出
                batch_inputs = batch_inputs[..., [2, 1, 0], :, :, :]
            else:
                # print('999999')  # 无输出
                raise ValueError(f'Invalid format shape: {format_shape}')

        # -- Normalization ---
        if self._enable_normalize:
            if view_shape is None:
                # print('10101010')  # 后输出
                batch_inputs = (batch_inputs - self.mean) / self.std
            else:
                # print('11 11 11 11')  # 无输出
                mean = self.mean.view(view_shape)
                std = self.std.view(view_shape)
                batch_inputs = (batch_inputs - mean) / std
        elif self.to_float32:
            # print('12121212') # 无输出
            batch_inputs = batch_inputs.to(torch.float32)

        # ----- Blending -----
        if training and self.blending is not None:
            # print('13131313') # 无输出
            batch_inputs, data_samples = self.blending(batch_inputs,
                                                       data_samples)

        return batch_inputs, data_samples
