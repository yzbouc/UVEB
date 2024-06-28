import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data

from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.flow_util import dequantize_flow
from basicsr.utils.registry import DATASET_REGISTRY
from PIL import Image

@DATASET_REGISTRY.register()
class GOPRODataset(data.Dataset):

    def __init__(self, opt):
        super(GOPRODataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']
        self.num_half_frames = opt['num_frame'] // 2

        self.keys = []
        if opt['test_mode']:
            with open(opt['meta_info_file_test'], 'r') as fin:
                for line in fin:
                    folder, frame_num, _ = line.split(' ')
                    self.keys.extend([f'{folder}/{i:06d}/{frame_num}' for i in range(int(frame_num))])
        else:
            with open(opt['meta_info_file_train'], 'r') as fin:
                for line in fin:
                    folder, frame_num, _ = line.split(' ')
                    self.keys.extend([f'{folder}/{i:06d}/{frame_num}' for i in range(int(frame_num))])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']                               ###############################################################################################
        key = self.keys[index]
        clip_name, frame_name, frame_num = key.split('/')  # key example: 000/000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)

        # ensure not exceeding the borders
        center_frame_idx = int(frame_name)
        start_frame_idx = center_frame_idx - self.num_half_frames * interval
        end_frame_idx = center_frame_idx + self.num_half_frames * interval
        while (start_frame_idx < 0) or (end_frame_idx > int(frame_num)-1):
            center_frame_idx = random.randint(0, int(frame_num)-1)
            start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = center_frame_idx + self.num_half_frames * interval
        frame_name = f'{center_frame_idx:06d}'
        neighbor_list = list(range(start_frame_idx, end_frame_idx + 1, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the GT frame (as the center frame)
        if self.is_lmdb:
            img_gt_path = f'{clip_name}/{frame_name}'
        else:
            img_gt_path = self.gt_root / clip_name / f'{frame_name}.png'
        img_bytes = self.file_client.get(img_gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # get the neighboring LQ frames
        img_lqs = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:06d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:06d}.png'
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

        ##############################################################################################################################
        #w,h = Image.open(str(img_lq_path)).size
        #w_1 = (int(w/1024 > 1) +1)*512
        
        #print("*************************************w_1",w_1)
        #gt_size = w_1
        ####################################################################################################################################

        # randomly crop
        img_gt, img_lqs = paired_random_crop(img_gt, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.append(img_gt)
        img_results = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_lqs = torch.stack(img_results[0:-1], dim=0)
        img_gt = img_results[-1]

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gt, 'key': key}

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class GOPRORecurrentDataset(data.Dataset):

    def __init__(self, opt):
        super(GOPRORecurrentDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame']

        self.keys = []
        if opt['test_mode']:
            with open(opt['meta_info_file_test'], 'r') as fin:
                for line in fin:
                    folder, frame_num, _ = line.split(' ')
                    self.keys.extend([f'{folder}/{i:06d}/{frame_num}' for i in range(int(frame_num))])
        else:
            with open(opt['meta_info_file_train'], 'r') as fin:
                for line in fin:
                    #folder, frame_num, _ = line.split(' ')
                    folder, frame_num, _, raw_score, gt_score = line.split(' ')                       
                    self.keys.extend([f'{folder}/{raw_score}/{gt_score}/{i:06d}/{frame_num}' for i in range(int(frame_num))])

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.is_lmdb = False
        if self.io_backend_opt['type'] == 'lmdb':
            self.is_lmdb = True
            self.io_backend_opt['db_paths'] = [self.lq_root, self.gt_root]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']

        # temporal augmentation configs
        self.interval_list = opt.get('interval_list', [1])
 
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join(str(x) for x in self.interval_list)
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_size = self.opt['gt_size']                                 ###############################################################################################
        key = self.keys[index]
        clip_name, raw_score, gt_score, frame_name, frame_num = key.split('/')  # key example: 000/000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)
        
        ###########################################################################
 
        #添加分数信息
        #folder可以得到视频名称信息，对应分数表得到该视频的的分数
        #如果视频分数差小且原始分数小于一定值，则该分数舍弃 
        #每个卡加载两个视频，用公式计算该视频置信分数，然后返回到loss那里乘起来
        



        a = 2*(float(gt_score)-float(raw_score))/float(raw_score)+float(gt_score)/float(100)
        #print("*****************a",a,clip_name)
        #############################################################################################################
        
        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        if start_frame_idx > int(frame_num) - self.num_frame:
            start_frame_idx = random.randint(0, int(frame_num) - self.num_frame)
        end_frame_idx = start_frame_idx + self.num_frame

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring LQ and GT frames
        img_lqs = []
        img_gts = []
        for neighbor in neighbor_list:
            if self.is_lmdb:
                img_lq_path = f'{clip_name}/{neighbor:06d}'
                img_gt_path = f'{clip_name}/{neighbor:06d}'
            else:
                img_lq_path = self.lq_root / clip_name / f'{neighbor:06d}.png'
                img_gt_path = self.gt_root / clip_name / f'{neighbor:06d}.png'
            ##############################################################################################################################
            #w,h = Image.open(str(img_lq_path)).size
            #w_1 = (int(w/1024 > 1) +1)*512
            
            #print("*************************************w_1",w_1)
            #gt_size = w_1
            ####################################################################################################################################
            # get LQ
            img_bytes = self.file_client.get(img_lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)
            img_lqs.append(img_lq)

            # get GT
            img_bytes = self.file_client.get(img_gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            img_gts.append(img_gt)
        
        # randomly crop
        img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size, scale, img_gt_path)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        img_results = augment(img_lqs, self.opt['use_flip'], self.opt['use_rot'])

        img_results = img2tensor(img_results)
        img_gts = torch.stack(img_results[len(img_lqs) // 2:], dim=0)
        img_lqs = torch.stack(img_results[:len(img_lqs) // 2], dim=0)

        # img_lqs: (t, c, h, w)
        # img_gts: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'a':a,'clip_name':clip_name}

    def __len__(self):
        return len(self.keys)
