# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from mmcv.transforms import BaseTransform, to_tensor
from mmengine.structures import InstanceData, LabelData

from mmaction.registry import TRANSFORMS
from mmaction.structures import ActionDataSample

import psutil


@TRANSFORMS.register_module()
class PackActionInputs(BaseTransform):
    """Pack the input data for the recognition.

    PackActionInputs first packs one of 'imgs', 'keypoint' and 'audios' into
    the `packed_results['inputs']`, which are the three basic input modalities
    for the task of rgb-based, skeleton-based and audio-based action
    recognition, as well as spatio-temporal action detection in the case
    of 'img'. Next, it prepares a `data_sample` for the task of action
    recognition (only a single label of `torch.LongTensor` format, which is
    saved in the `data_sample.gt_labels.item`) or spatio-temporal action
    detection respectively. Then, it saves the meta keys defined in
    the `meta_keys` in `data_sample.metainfo`, and packs the `data_sample`
    into the `packed_results['data_samples']`.

    Args:
        meta_keys (Sequence[str]): The meta keys to saved in the
            `metainfo` of the `data_sample`.
            Defaults to ``('img_shape', 'img_key', 'video_id', 'timestamp')``.
    """

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_labels': 'labels',
    }

    def __init__(
        self,
        collect_keys: Optional[Tuple[str]] = None,
        meta_keys: Sequence[str] = ('img_shape', 'img_key', 'video_id',
                                    'timestamp')
    ) -> None:
        self.collect_keys = collect_keys
        self.meta_keys = meta_keys

    def transform(self, results_all: Dict) -> Dict:
        """The transform function of :class:`PackActionInputs`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """

        memory_info1 = psutil.virtual_memory()
        used_memory1 = memory_info1.used
        print("Used Memory1:", used_memory1)

        results = results_all[0]
        results1 = results_all[1]
        results2 = results_all[2]
        results3 = results_all[3]

        packed_results = dict()
        packed_results1 = dict()
        packed_results2 = dict()
        packed_results3 = dict()

        if self.collect_keys is not None:
            # print('11111')
            packed_results['inputs'] = dict()
            for key in self.collect_keys:
                # print('22222')
                packed_results['inputs'][key] = to_tensor(results[key])
        else:
            if 'imgs' in results:
                # rint('33333')
                imgs = results['imgs']
                packed_results['inputs'] = to_tensor(imgs)
            elif 'heatmap_imgs' in results:
                heatmap_imgs = results['heatmap_imgs']
                packed_results['inputs'] = to_tensor(heatmap_imgs)
            elif 'keypoint' in results:
                keypoint = results['keypoint']
                packed_results['inputs'] = to_tensor(keypoint)
            elif 'audios' in results:
                audios = results['audios']
                packed_results['inputs'] = to_tensor(audios)
            elif 'text' in results:
                text = results['text']
                packed_results['inputs'] = to_tensor(text)
            else:
                raise ValueError(
                    'Cannot get `imgs`, `keypoint`, `heatmap_imgs`, '
                    '`audios` or `text` in the input dict of '
                    '`PackActionInputs`.')
        
        data_sample = ActionDataSample()
        # print('data_sample:', data_sample)

        if 'gt_bboxes' in results:
            instance_data = InstanceData()
            for key in self.mapping_table.keys():
                instance_data[self.mapping_table[key]] = to_tensor(
                    results[key])
            data_sample.gt_instances = instance_data

            if 'proposals' in results:
                data_sample.proposals = InstanceData(
                    bboxes=to_tensor(results['proposals']))

        if 'label' in results:
            # print('44444')
            label_data = LabelData()
            label_data.item = to_tensor(results['label'])
            data_sample.gt_labels = label_data

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        # print('packed_results:', packed_results) # feature
        # packed_results = 'eeee'
        # print('results1:', results1) # feature
        if 'imgs' in results1:
            # print('33333')
            imgs1 = results1['imgs']
            packed_results1['inputs'] = to_tensor(imgs1)
        data_sample1 = ActionDataSample()
        if 'label' in results1:
            # print('44444')
            label_data1 = LabelData()
            label_data1.item = to_tensor(results['label'])
            data_sample1.gt_labels = label_data1

        img_meta1 = {k: results1[k] for k in self.meta_keys if k in results1}
        data_sample1.set_metainfo(img_meta1)
        packed_results1['data_samples'] = data_sample1

        if 'imgs' in results2:
            # print('33333')
            imgs2 = results2['imgs']
            packed_results2['inputs'] = to_tensor(imgs2)
        data_sample2 = ActionDataSample()
        if 'label' in results2:
            # print('44444')
            label_data2 = LabelData()
            label_data2.item = to_tensor(results['label'])
            data_sample2.gt_labels = label_data2

        img_meta2 = {k: results2[k] for k in self.meta_keys if k in results2}
        data_sample2.set_metainfo(img_meta2)
        packed_results2['data_samples'] = data_sample2

        if 'imgs' in results3:
            # print('33333')
            imgs3 = results3['imgs']
            packed_results3['inputs'] = to_tensor(imgs3)
        data_sample3 = ActionDataSample()
        if 'label' in results3:
            # print('44444')
            label_data3 = LabelData()
            label_data3.item = to_tensor(results['label'])
            data_sample3.gt_labels = label_data3

        img_meta3 = {k: results3[k] for k in self.meta_keys if k in results3}
        data_sample3.set_metainfo(img_meta3)
        packed_results3['data_samples'] = data_sample3

        del results, results1, results2, results3
        del data_sample1, data_sample2, data_sample3

        memory_info3 = psutil.virtual_memory()
        used_memory3 = memory_info3.used
        print("Used Memory3:", used_memory3)

        return packed_results, packed_results1, packed_results2, packed_results3

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(collect_keys={self.collect_keys}, '
        repr_str += f'meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PackLocalizationInputs(BaseTransform):

    def __init__(self, keys=(), meta_keys=('video_name', )):
        self.keys = keys
        self.meta_keys = meta_keys

    def transform(self, results):
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_samples' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'raw_feature' in results:
            raw_feature = results['raw_feature']
            packed_results['inputs'] = to_tensor(raw_feature)
        elif 'bsp_feature' in results:
            packed_results['inputs'] = torch.tensor(0.)
        else:
            raise ValueError(
                'Cannot get "raw_feature" or "bsp_feature" in the input '
                'dict of `PackActionInputs`.')

        data_sample = ActionDataSample()
        for key in self.keys:
            if key not in results:
                continue
            if key == 'gt_bbox':
                instance_data = InstanceData()
                instance_data[key] = to_tensor(results[key])
                data_sample.gt_instances = instance_data
            elif key == 'proposals':
                instance_data = InstanceData()
                instance_data[key] = to_tensor(results[key])
                data_sample.proposals = instance_data
            else:
                raise NotImplementedError(
                    f"Key '{key}' is not supported in `PackLocalizationInputs`"
                )

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class Transpose(BaseTransform):
    """Transpose image channels to a given order.

    Args:
        keys (Sequence[str]): Required keys to be converted.
        order (Sequence[int]): Image channel order.
    """

    def __init__(self, keys, order):
        self.keys = keys
        self.order = order

    def transform(self, results):
        """Performs the Transpose formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        for key in self.keys:
            results[key] = results[key].transpose(self.order)
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'keys={self.keys}, order={self.order})')


@TRANSFORMS.register_module()
class FormatShape(BaseTransform):
    """Format final imgs shape to the given input_format.

    Required keys:
        - imgs (optional)
        - heatmap_imgs (optional)
        - num_clips
        - clip_len

    Modified Keys:
        - imgs (optional)
        - input_shape (optional)

    Added Keys:
        - heatmap_input_shape (optional)

    Args:
        input_format (str): Define the final data format.
        collapse (bool): To collapse input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Defaults to False.
    """

    def __init__(self, input_format: str, collapse: bool = False) -> None:
        self.input_format = input_format
        self.collapse = collapse
        if self.input_format not in [
                'NCTHW', 'NCHW', 'NCHW_Flow', 'NCTHW_Heatmap', 'NPTCHW'
        ]:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def transform(self, results_all: Dict) -> Dict:
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        results = results_all[0]
        results1 = results_all[1]
        results2 = results_all[2]
        results3 = results_all[3]

        if not isinstance(results['imgs'], np.ndarray):
            # print('1111') # 输出
            results['imgs'] = np.array(results['imgs'])

        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * T
        if self.collapse:
            # print('2222')
            assert results['num_clips'] == 1

        if self.input_format == 'NCTHW':
            
            if 'imgs' in results:
                # print('3333')  # 输出
                imgs = results['imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                if isinstance(clip_len, dict):
                    clip_len = clip_len['RGB']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x H x W x C
                imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['imgs'] = imgs
                results['input_shape'] = imgs.shape

            if 'heatmap_imgs' in results:
                # print('4444')
                imgs = results['heatmap_imgs']
                num_clips = results['num_clips']
                clip_len = results['clip_len']
                # clip_len must be a dict
                clip_len = clip_len['Pose']

                imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
                # N_crops x N_clips x T x C x H x W
                imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
                # N_crops x N_clips x C x T x H x W
                imgs = imgs.reshape((-1, ) + imgs.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results['heatmap_imgs'] = imgs
                results['heatmap_input_shape'] = imgs.shape

        elif self.input_format == 'NCTHW_Heatmap':
            # print('5555')
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x T x C x H x W
            imgs = np.transpose(imgs, (0, 1, 3, 2, 4, 5))
            # N_crops x N_clips x C x T x H x W
            imgs = imgs.reshape((-1, ) + imgs.shape[2:])
            # M' x C x T x H x W
            # M' = N_crops x N_clips
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NCHW':
            # print('6666')
            imgs = results['imgs']
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            # M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NCHW_Flow':
            # print('7777')
            num_imgs = len(results['imgs'])
            assert num_imgs % 2 == 0
            n = num_imgs // 2
            h, w = results['imgs'][0].shape
            x_flow = np.empty((n, h, w), dtype=np.float32)
            y_flow = np.empty((n, h, w), dtype=np.float32)
            for i in range(n):
                x_flow[i] = results['imgs'][2 * i]
                y_flow[i] = results['imgs'][2 * i + 1]
            imgs = np.stack([x_flow, y_flow], axis=-1)

            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x T x H x W x C
            imgs = np.transpose(imgs, (0, 1, 2, 5, 3, 4))
            # N_crops x N_clips x T x C x H x W
            imgs = imgs.reshape((-1, imgs.shape[2] * imgs.shape[3]) +
                                imgs.shape[4:])
            # M' x C' x H x W
            # M' = N_crops x N_clips
            # C' = T x C
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        elif self.input_format == 'NPTCHW':

            # print('8888')
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = results['imgs']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x T
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W
            results['imgs'] = imgs
            results['input_shape'] = imgs.shape

        if self.collapse:
            # print('9999')
            assert results['imgs'].shape[0] == 1
            results['imgs'] = results['imgs'].squeeze(0)
            results['input_shape'] = results['imgs'].shape

        if not isinstance(results1['imgs'], np.ndarray):
            # print('1111')
            results1['imgs'] = np.array(results1['imgs'])
        
        if self.input_format == 'NCTHW':
            
            if 'imgs' in results1:
                # print('3333')  # 输出
                imgs1 = results1['imgs']
                num_clips1 = results1['num_clips']
                clip_len1 = results1['clip_len']
                if isinstance(clip_len1, dict):
                    clip_len1 = clip_len1['RGB']

                imgs1 = imgs1.reshape((-1, num_clips1, clip_len1) + imgs1.shape[1:])
                # N_crops x N_clips x T x H x W x C
                imgs1 = np.transpose(imgs1, (0, 1, 5, 2, 3, 4))
                # N_crops x N_clips x C x T x H x W
                imgs1 = imgs1.reshape((-1, ) + imgs1.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results1['imgs'] = imgs1
                results1['input_shape'] = imgs1.shape

        if not isinstance(results2['imgs'], np.ndarray):
            # print('1111')
            results2['imgs'] = np.array(results2['imgs'])
        
        if self.input_format == 'NCTHW':
            
            if 'imgs' in results2:
                # print('3333')  # 输出
                imgs2 = results2['imgs']
                num_clips2 = results2['num_clips']
                clip_len2 = results2['clip_len']
                if isinstance(clip_len2, dict):
                    clip_len2 = clip_len2['RGB']

                imgs2 = imgs2.reshape((-1, num_clips2, clip_len2) + imgs2.shape[1:])
                # N_crops x N_clips x T x H x W x C
                imgs2 = np.transpose(imgs2, (0, 1, 5, 2, 3, 4))
                # N_crops x N_clips x C x T x H x W
                imgs2 = imgs2.reshape((-1, ) + imgs2.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results2['imgs'] = imgs2
                results2['input_shape'] = imgs2.shape

        if not isinstance(results3['imgs'], np.ndarray):
            # print('1111')
            results3['imgs'] = np.array(results3['imgs'])
        
        if self.input_format == 'NCTHW':
            
            if 'imgs' in results3:
                # print('3333')  # 输出
                imgs3 = results3['imgs']
                num_clips3 = results3['num_clips']
                clip_len3 = results3['clip_len']
                if isinstance(clip_len3, dict):
                    clip_len3 = clip_len3['RGB']

                imgs3 = imgs3.reshape((-1, num_clips3, clip_len3) + imgs3.shape[1:])
                # N_crops x N_clips x T x H x W x C
                imgs3 = np.transpose(imgs3, (0, 1, 5, 2, 3, 4))
                # N_crops x N_clips x C x T x H x W
                imgs3 = imgs3.reshape((-1, ) + imgs3.shape[2:])
                # M' x C x T x H x W
                # M' = N_crops x N_clips
                results3['imgs'] = imgs3
                results3['input_shape'] = imgs3.shape

        return results, results1, results2, results3

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


@TRANSFORMS.register_module()
class FormatAudioShape(BaseTransform):
    """Format final audio shape to the given input_format.

    Required keys are ``audios``, ``num_clips`` and ``clip_len``, added or
    modified keys are ``audios`` and ``input_shape``.

    Args:
        input_format (str): Define the final imgs format.
    """

    def __init__(self, input_format: str) -> None:
        self.input_format = input_format
        if self.input_format not in ['NCTF']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def transform(self, results: dict) -> dict:
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        audios = results['audios']
        # clip x sample x freq -> clip x channel x sample x freq
        clip, sample, freq = audios.shape
        audios = audios.reshape(clip, 1, sample, freq)
        results['audios'] = audios
        results['input_shape'] = audios.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str


@TRANSFORMS.register_module()
class FormatGCNInput(BaseTransform):
    """Format final skeleton shape.

    Required Keys:

        - keypoint
        - keypoint_score (optional)
        - num_clips (optional)

    Modified Key:

        - keypoint

    Args:
        num_person (int): The maximum number of people. Defaults to 2.
        mode (str): The padding mode. Defaults to ``'zero'``.
    """

    def __init__(self, num_person: int = 2, mode: str = 'zero') -> None:
        self.num_person = num_person
        assert mode in ['zero', 'loop']
        self.mode = mode

    def transform(self, results: Dict) -> Dict:
        """The transform function of :class:`FormatGCNInput`.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        keypoint = results['keypoint']
        if 'keypoint_score' in results:
            keypoint = np.concatenate(
                (keypoint, results['keypoint_score'][..., None]), axis=-1)

        cur_num_person = keypoint.shape[0]
        if cur_num_person < self.num_person:
            pad_dim = self.num_person - cur_num_person
            pad = np.zeros(
                (pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == 'loop' and cur_num_person == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]

        elif cur_num_person > self.num_person:
            keypoint = keypoint[:self.num_person]

        M, T, V, C = keypoint.shape
        nc = results.get('num_clips', 1)
        assert T % nc == 0
        keypoint = keypoint.reshape(
            (M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)

        results['keypoint'] = np.ascontiguousarray(keypoint)
        return results

    def __repr__(self) -> str:
        repr_str = (f'{self.__class__.__name__}('
                    f'num_person={self.num_person}, '
                    f'mode={self.mode})')
        return repr_str
