import torch
from collections import Counter
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm
import numpy as np
import time

from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel
from torchvision import transforms
import torch.nn.functional as F
import cv2

@MODEL_REGISTRY.register()
class VideoRecurrentModel(VideoBaseModel):

    def __init__(self, opt):
        super(VideoRecurrentModel, self).__init__(opt)
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for differnet lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name or 'raft' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'raft' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        super(VideoRecurrentModel, self).optimize_parameters(current_iter)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None
        # initialize self.metric_results
        # It is a dict: {
        #    'folder1': tensor (num_frame x len(metrics)),
        #    'folder2': tensor (num_frame x len(metrics))
        # }
        if with_metrics and not hasattr(self, 'metric_results'):
            self.metric_results = {}
            num_frame_each_folder = Counter(dataset.data_info['folder'])
            for folder, num_frame in num_frame_each_folder.items():
                self.metric_results[folder] = torch.zeros(
                    num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')

        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')
        # Will evaluate (num_folders + num_pad) times, but only the first
        # num_folders results will be recorded. (To avoid wait-dead)
        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']

            # compute outputs
            val_data['lq'].unsqueeze_(0)
            val_data['gt'].unsqueeze_(0)
            self.feed_data(val_data)
            val_data['lq'].squeeze_(0)
            val_data['gt'].squeeze_(0)

            self.test()
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            if 'gt' in visuals:
                del self.gt
            torch.cuda.empty_cache()

            if self.center_frame_only:
                visuals['result'] = visuals['result'].unsqueeze(1)
                if 'gt' in visuals:
                    visuals['gt'] = visuals['gt'].unsqueeze(1)

            # evaluate
            if i < num_folders:
                for idx in range(visuals['result'].size(1)):
                    result = visuals['result'][0, idx, :, :, :]
                    print("result", result.shape,type(result))
                    lq = visuals['lq'][0, idx, :, :, :]
                    print("**********",lq.shape)
                    img_w, img_h = lq.shape[2], lq.shape[1]
                    #result = F.interpolate(result, size=(img_h, img_w), mode='nearest')
                    result_img = tensor2img([result])  # uint8, bgr
                    result_img = cv2.resize(result_img,(img_w,img_h))
                    print("result_img",result_img.shape)
                    if 'gt' in visuals:
                        gt = visuals['gt'][0, idx, :, :, :]
                        gt_img = tensor2img([gt])  # uint8, bgr

                    if save_img:
                       # if self.opt['is_train']:
                        #    raise NotImplementedError('saving image is not supported during training.')
                        #else:
                        if self.center_frame_only:  # vimeo-90k
                            clip_ = val_data['lq_path'].split('/')[-3]
                            seq_ = val_data['lq_path'].split('/')[-2]
                            name_ = f'{clip_}_{seq_}'
                            img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                f"{name_}_{self.opt['name']}.png")
                        else:  # others
                            img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder,
                                                f"{idx:08d}_{self.opt['name']}.png")
                        # image name only for REDS dataset
                    imwrite(result_img, img_path)

                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            metric_data = dict(img1=result_img, img2=gt_img)
                            result = calculate_metric(metric_data, opt_)
                            self.metric_results[folder][idx, metric_idx] += result

                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def test(self):
        n = self.lq.size(1)
        self.net_g.eval()

        flip_seq = self.opt['val'].get('flip_seq', False)
        self.center_frame_only = self.opt['val'].get('center_frame_only', False)

        if flip_seq:
            with torch.no_grad():
                output_sum = 0.0
                for k in range(0, 8):
                    print(k)
                    if k < 4:
                        output_sum += self.net_g(self.lq.rot90(k, [3, 4]).cuda()).rot90(-k, [3, 4]).detach().cpu()
                    else:
                        output_sum += self.net_g(self.lq.rot90(k, [3, 4]).flip(1).cuda()).flip(1).rot90(-k, [3, 4]).detach().cpu()
                self.output = 0.125 * output_sum

            # with torch.no_grad():
            #     output_list = []
            #     for k in range(0, 8):
            #         print(k)
            #         if k < 4:
            #             output_list.append(self.net_g(self.lq.rot90(k, [3, 4]).cuda()).rot90(-k, [3, 4]).detach().cpu())
            #         else:
            #             output_list.append(self.net_g(self.lq.rot90(k, [3, 4]).flip(1).cuda()).flip(1).rot90(-k, [3, 4]).detach().cpu())
            #     self.output = torch.median(torch.cat(output_list, dim=0), dim=0)[0].unsqueeze(dim=0)
        else:
            with torch.no_grad():
                # start = torch.cuda.Event(enable_timing=True)
                # end = torch.cuda.Event(enable_timing=True)
                # start.record()
                #self.output = self.net_g(self.lq)             ############################################################
                #with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
                print("self.lq", self.lq.shape)
                img_w, img_h = self.lq.shape[4], self.lq.shape[3]
                new_width, new_height = img_w//16*16, img_h//16*16
                self.lq1 = self.lq.squeeze(0)
                #self.lq = F.interpolate(self.lq, size=(new_height, new_width), mode='trilinear',align_corners=False)
                self.lq1 = transforms.Resize((img_h//16*16, img_w//16*16))(self.lq1)
                self.lq1 = self.lq1.unsqueeze(0)
                print(self.lq1.shape)
                self.output,self.output_small = self.net_g(self.lq1)
                #print(prof.table())
                #prof.export_chrome_trace('./our_profile.json')               
                # end.record()
                # torch.cuda.synchronize()
                # print("inference time:", start.elapsed_time(end))

        if self.center_frame_only:
            self.output = self.output[:, n // 2, :, :, :]

        self.net_g.train()
