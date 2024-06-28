import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from torch.nn import functional as F
from pytorch_msssim import ms_ssim

@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('hem_opt'):
            self.cri_hem = build_loss(train_opt['hem_opt']).to(self.device)
        else:
            self.cri_hem = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'a' in data:
            self.a = data['a']
            #print("*****************************************self.a",self.a)
        
        #self.clip_name = data['clip_name']
    def optimize_parameters(self, current_iter):
        # with torch.cuda.amp.autocast(enabled=True):
        self.output,self.ouput_small= self.net_g(self.lq)
        if len(self.output.size()) == 4:
            b, c, h, w = self.output.size()
        else:
            b, t, c, h, w = self.output.size()

        l_total = 0
        loss_dict = OrderedDict()
        self.gt_small = F.interpolate(self.gt[:b,t//2], scale_factor=0.25, mode='bilinear', align_corners=False)
        #print("*****************************************self.gt",self.gt)
        #print("*****************************************self.gt_small",self.gt_small)
        #print("*****************************************self.output",self.output)
        #print("*****************************************self.output_small",self.output_small)
        #########################################################################
        #output和gt分成两个，各自计算loss
        
        #######################################################################
        # pixel loss
        l_pix1 = 0
        l_pix2 = 0

        if self.cri_pix:
            #print("ouput",self.output.shape)
            #print("gt",self.gt.shape)
            #print("ouput", type(self.output))
            #print("gt", type(self.gt))
            #print("*****************************************a_loss",self.a.item(),self.clip_name)
            
            #l_pix1 = self.a.item()*self.cri_pix(self.output, self.gt)
            #l_pix1 = self.cri_pix(self.output, self.gt)
            #print("self.cri_pix(self.output, self.gt)",self.cri_pix(self.output, self.gt))
            pix_all=[]
            for i in range(b):
                l_pix1 = self.cri_pix(self.output[i], self.gt[i])
                #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",self.a[i])
                l_pix2 = self.cri_pix(self.ouput_small[i], self.gt_small[i])
                pix_all.append(l_pix1+l_pix2)
            l_pix = sum(pix_all)
                
            
            #l_pix2 = self.a.item()*self.cri_pix(self.output_small, self.gt_small)
            #l_pix2 = self.cri_pix(self.output_small, self.gt_small)
            ###mind that btchw            

           
           # l_pix =l_pix
            #ms-ssim_loss
            #ms=[]
           # l_ms1=0
            #l_ms2=0
            #for i in range(b):
                #ms[i] =(1-ms_ssim(self.output[i],self.gt[i],data_range=1,size_average=True))*self.a[i]
                #l_ms1+=ms[i]  #output[i]=n*3*hw
                #l_ms2+=(1-ms_ssim(self.output_small,self.gt_small,data_range=1,size_average=True))*self.a[t//2]
            #l_ms =l_ms1+l_ms2                     
            #if need ms-ssim-loss  ? total+=? ?cri_pix calculate all frame pix loss？ is this L1? make sure and check it 
            # l_total+=l_pix+l_ms 
            l_total+=l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            self.gt=self.gt.squeeze(0)
            l_percep1, l_style1 = self.a.item()*self.cri_perceptual(self.output, self.gt)
            l_percep2, l_style2 = self.a.item()*self.cri_pix(self.output_small, self.gt_small)
            l_percep = l_percep1+l_percep2
            l_style = l_style1+l_style2
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
                
            
                        
        # hem loss
        if self.cri_hem:
            l_hem = self.a.item()*self.cri_hem(self.output.view(-1, c, h, w), self.gt.view(-1, c, h, w))
            l_total += l_hem
            loss_dict['l_hem'] = l_hem

        # fft loss
        if self.cri_fft:
            l_fft = self.a.item()*self.cri_fft(self.output.view(-1, c, h, w), self.gt.view(-1, c, h, w))
            l_total += l_fft
            loss_dict['l_fft'] = l_fft
        ##############################################################
        #置信分数*loss
        ##################################################################
        # l_total = l_total / 4.0
        l_total.backward()
        # if current_iter % 4 == 0:
            # torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()
        self.optimizer_g.zero_grad()
        # self.scaler.scale(l_total).backward()
        # self.scaler.step(self.optimizer_g)
        # self.scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output,self.ouput_small = self.net_g_ema(self.lq)           ########################################################
        else:
            self.net_g.eval()
            with torch.no_grad():
                #self.output = self.net_g(self.lq)           #############################################################传入get_current_visuals
                self.output,self.ouput_small = self.net_g(self.lq)
            # with torch.no_grad():
            #     output_list = 0.0
            #     output_f = self.net_g(self.lq)
            #     output_list += output_f
            #     output_f = self.net_g(torch.flip(self.lq, (-2,)))
            #     output_f = torch.flip(output_f, (-2,))
            #     output_list += output_f
            #     self.output = output_list / 2.0
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
