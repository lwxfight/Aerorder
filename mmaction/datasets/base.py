# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Callable, List, Optional, Union

import torch
from mmengine.dataset import BaseDataset

from mmaction.utils import ConfigType

import os
import random

class BaseActionDataset(BaseDataset, metaclass=ABCMeta):
    """Base class for datasets.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict, optional): Path to a directory where
            videos are held. Defaults to None.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Defaults to False.
        num_classes (int, optional): Number of classes of the dataset, used in
            multi-class datasets. Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Defaults to 0.
        modality (str): Modality of data. Support ``RGB``, ``Flow``, ``Pose``,
            ``Audio``. Defaults to ``RGB``.
    """

    # view1_list = []
    
    # view1_path = '/home/public/Hidden_Intention/Zhouz/dataset/list/Drone/view1/train.txt'
    # # view2_file = os.listdir(view2_path)
    # with open(view1_path, 'r') as view1_files:
    #     for view1_file in view1_files:
    #         view1_dic = {}
    #         # print('view2_file:', view2_file)
    #         file_name, label = view1_file.split(' ')
    #         view1_dic['filename'] = file_name
    #         view1_dic['label'] = label.strip()

    #         view1_list.append(view1_dic)

    # view2_list = []
    
    # view2_path = '/home/public/Hidden_Intention/Zhouz/dataset/list/Drone/view2/train.txt'
    # # view2_file = os.listdir(view2_path)
    # with open(view2_path, 'r') as view2_files:
    #     for view2_file in view2_files:
    #         view2_dic = {}
    #         # print('view2_file:', view2_file)
    #         file_name, label = view2_file.split(' ')
    #         view2_dic['filename'] = file_name
    #         view2_dic['label'] = label.strip()

    #         view2_list.append(view2_dic)
 

    # view3_list = []

    # view3_path = '/home/public/Hidden_Intention/Zhouz/dataset/list/Drone/view3/train.txt'
    # # view2_file = os.listdir(view2_path)
    # with open(view3_path, 'r') as view3_files:
    #     for view3_file in view3_files:
    #         view3_dic = {}
    #         file_name, label = view3_file.split(' ')
    #         view3_dic['filename'] = file_name
    #         view3_dic['label'] = label.strip()
    #         view3_list.append(view3_dic)
    # # print('view3_list:', view3_list)
    
    
    NTU_list = []

    NTU_path = '/home/public/Hidden_Intention/Zhouz/dataset/NTU/list/train.txt'
    # view2_file = os.listdir(view2_path)
    with open(NTU_path, 'r') as NTU_files:
        for NTU_file in NTU_files:
            NTU_dic = {}
            # print('view2_file:', view2_file)
            file_name, label = NTU_file.split(' ')
            NTU_dic['filename'] = file_name
            NTU_dic['label'] = label.strip()

            NTU_list.append(NTU_dic)

    view1_list = []
    
    view1_path = '/home/public/Hidden_Intention/Zhouz/dataset/MOD/view_list/viewa/train.txt'
    # view2_file = os.listdir(view2_path)
    with open(view1_path, 'r') as view1_files:
        for view1_file in view1_files:
            view1_dic = {}
            # print('view2_file:', view2_file)
            file_name, label = view1_file.split(' ')
            view1_dic['filename'] = file_name
            view1_dic['label'] = label.strip()

            view1_list.append(view1_dic)

    view2_list = []
    
    view2_path = '/home/public/Hidden_Intention/Zhouz/dataset/MOD/view_list/viewb/train.txt'
    # view2_file = os.listdir(view2_path)
    with open(view2_path, 'r') as view2_files:
        for view2_file in view2_files:
            view2_dic = {}
            # print('view2_file:', view2_file)
            file_name, label = view2_file.split(' ')
            view2_dic['filename'] = file_name
            view2_dic['label'] = label.strip()

            view2_list.append(view2_dic)
 

    view3_list = []

    view3_path = '/home/public/Hidden_Intention/Zhouz/dataset/MOD/view_list/viewc/train.txt'
    # view2_file = os.listdir(view2_path)
    with open(view3_path, 'r') as view3_files:
        for view3_file in view3_files:
            view3_dic = {}
            file_name, label = view3_file.split(' ')
            view3_dic['filename'] = file_name
            view3_dic['label'] = label.strip()
            view3_list.append(view3_dic)
    # print('view3_list:', view3_list)

    def __init__(self,
                 ann_file: str,
                 pipeline: List[Union[ConfigType, Callable]],
                 data_prefix: Optional[ConfigType] = dict(prefix=''),
                 test_mode: bool = False,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 0,
                 modality: str = 'RGB',
                 **kwargs) -> None:
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        super().__init__(
            ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index."""
        # print('idx:', idx)
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality
        data_info['start_index'] = self.start_index

        data_info_copy1 = data_info.copy()
        data_info_copy2 = data_info.copy()
        data_info_copy3 = data_info.copy()
        
        if self.multi_class:
            onehot = torch.zeros(self.num_classes)
            onehot[data_info['label']] = 1.
            data_info['label'] = onehot
        # print('data_info:', data_info) # {'filename': '/home/public/Hidden_Intention/Zhouz/dataset/Drone/train/S1_clapping_HD.mp4',
                                       #'label': 1, 'sample_idx': 0, 'modality': 'RGB', 'start_index': 0}
        # data_info = [{'filename': '/home/zhouzhuo/scratch/dataset/Drone/train/S1_claHD', 'label': 1}, {'filename': '/home/zhuo/scratch/dataset/Drone/train/S1_hittingBottle_toLeft_HD', 'label': 2}]
        view_label = data_info['label']
        # print('view_label:', view_label)

        NTU_label_samples =  [sample for sample in self.NTU_list if sample['label'] == str(view_label)]
        # print('view2_label_samples:', view2_label_samples)
        if len(NTU_label_samples) > 0:
            # print('22222')
            random_sample = random.choice(NTU_label_samples)
            # print('random_sample:', random_sample)
            data_info_NTU = random_sample
            # print('data_info2--:', data_info2)
            data_info_NTU['modality'] = self.modality
            data_info_NTU['start_index'] = self.start_index
        else:
            data_info_NTU = data_info_copy1

        # print('self.view2_list:', self.view2_list)
        view1_label_samples =  [sample for sample in self.view1_list if sample['label'] == str(view_label)]
        # print('view2_label_samples:', view2_label_samples)
        if len(view1_label_samples) > 0:
            # print('22222')
            random_sample = random.choice(view1_label_samples)
            # print('random_sample:', random_sample)
            data_info1 = random_sample
            # print('data_info2--:', data_info2)
            data_info1['modality'] = self.modality
            data_info1['start_index'] = self.start_index
        else:
            data_info1 = data_info_copy1
        

        view2_label_samples =  [sample for sample in self.view2_list if sample['label'] == str(view_label)]
        # print('view2_label_samples:', view2_label_samples)
        if len(view2_label_samples) > 0:
            # print('22222')
            random_sample = random.choice(view2_label_samples)
            # print('random_sample:', random_sample)
            data_info2 = random_sample
            data_info2['modality'] = self.modality
            data_info2['start_index'] = self.start_index
            # print('data_info2--:', data_info2)
        else:
            data_info2 = data_info_copy2
        
        # print('data_info2:', data_info2)

        view3_label_samples =  [sample for sample in self.view3_list if sample['label'] == str(view_label)]
        # print('view2_label_samples:', view2_label_samples)
        if len(view3_label_samples) > 0:
            # print('22222')
            random_sample = random.choice(view3_label_samples)
            # print('random_sample:', random_sample)
            data_info3 = random_sample
            # print('data_info2--:', data_info2)
            data_info3['modality'] = self.modality
            data_info3['start_index'] = self.start_index
        else:
            data_info3 = data_info_copy3  

        # print('data_info:', data_info)
        # print('data_info1:', data_info1)
        # print('data_info2:', data_info2)
        # print('data_info3:', data_info3)

        
        # print('view1_name:', view1_name) # /home/public/Hidden_Intention/Zhouz/dataset/Drone/train/S1_punching_toLeft_sideView_HD.mp4
        # return data_info, data_info1, data_info2, data_info3
        return data_info, data_info_NTU, [], []
