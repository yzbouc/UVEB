
> [**UVEB: A Large-scale Benchmark and Baseline Towards Real-World
Underwater Video Enhancement**]()  
> Yaofeng Xie1 Lingwei Kong2 Kai Chen2 Ziqiang Zheng3 Xiao Yu2 Zhibin Yu1,2,† Bing Zheng1,2

1College of Electronic Engineering,Ocean University of China

2Key Laboratory of Ocean Observation and Information of Hainan Province,
Sanya Oceanographic Institution, Ocean University of China

3Department of Computer Science and Engineering,The Hong Kong University of Science and Technology

† corresponding author: yuzhibin@ouc.edu.cn; Project website: https://github.com/yzbouc/UVEB

This repository contains the official implementation and dataset of the CVPR2024 paper "UVEB: A Large-scale Benchmark and Baseline Towards Real-World
Underwater Video Enhancement"


> Learning-based underwater image enhancement (UIE)
methods have made great progress. However, the lack of
large-scale and high-quality paired training samples has
become the main bottleneck hindering the development of
UIE. The inter-frame information in underwater videos can
accelerate or optimize the UIE process. Thus, we constructed the first large-scale high-resolution underwater
video enhancement benchmark (UVEB) to promote the development of underwater vision. It contains 1,308 pairs
of video sequences and more than 453,000 high-resolution
with 38% Ultra-High-Definition (UHD) 4K frame pairs.
UVEB comes from multiple countries, containing various
scenes and video degradation types to adapt to diverse and
complex underwater environments. We also propose the first
supervised underwater video enhancement method, UVENet. UVE-Net converts the current frame information into
convolutional kernels and passes them to adjacent frames
for efficient inter-frame information exchange. By fully utilizing the redundant degraded information of underwater
videos, UVE-Net completes video enhancement better. Experiments show the effective network design and good performance of UVE-Net.

## Dataset display
> ![Alt text](/Pictures/1.png)
> ![Alt text2](/Pictures/2.png)
## Method
> ![Alt text2](/Pictures/8.png)
> 
## Update
- **2024.07.07:** The quality score information of the videos has been released.
- The score information for the train dataset was stored in meta_info_our_video_quality_score_GT.txt. The score information for the test dataset was stored in meta_info_our_video_quality_scoretest_GT.txt.
- The format of each row of information is: video name, total frames, resolution, video quality score before enhancement, and video quality score after enhancement. For example : cv_30 241 (2064,3840,3) 45.93 54.73
- **2024.07.03:** Download website for supplementary materials of the paper: https://openaccess.thecvf.com/content/CVPR2024/supplemental/Xie_UVEB_A_Large-scale_CVPR_2024_supplemental.pdf
- Download website for the paper:  https://openaccess.thecvf.com/content/CVPR2024/papers/Xie_UVEB_A_Large-scale_Benchmark_and_Baseline_Towards_Real-World_Underwater_Video_CVPR_2024_paper.pdf
- **2024.06.25:** Code Release.
- **2024.06.17:** The UVEB dataset has been published.
- Test dataset is aviable at https://pan.baidu.com/s/1qPLYZ7tHJm2YeNqh8wxK1Q.  Extraction code: 1234
- Train dataset is aviable at https://pan.baidu.com/s/1A-Z6kbYAiCy95d_DLrdDBg. Exreaction code:1234
- For researchers outside of China, the full dataset can be aviable at https://terabox.com/s/1Mmz6ZAUv6h2GZFKZybD5mg.
- **2024.02.27:** Accepted by CVPR 2024!
## Dependencies
- Linux (Tested on Ubuntu 18.04)
- Python 3 (Recommend to use Anaconda)
- PyTorch 1.10.1: ```conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge```
- Install dependent packages :```pip install -r requirements.txt```
- Install UVE-Net :```python setup.py develop```
## Get Started
#### Pretrained models
- Models are available in ```'UVE-Net/pretrained/large_net_g.pth'```
#### Dataset Organization Form
If you prepare your own dataset, please follow the following form like UEVB:
```
|--dataset  
    |--blur  
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
            :
        |--video n
    |--gt
        |--video 1
            |--frame 1
            |--frame 2
                ：  
        |--video 2
        	:
        |--video n
```
#### Training
Attention:When using the small model code, please copy the small model code ```deblur_arch_small.py``` to ```deblur_arch.py``` and change the class name ```class Deblur_small (nn. Module):``` to ```class Deblur (nn. Module):```
- Download training dataset like above form.
- Run the following commands:
```
Single GPU
python basicsr/train.py -opt options/train/Deblur/train_Deblur_GOPRO.yml
Multi-GPUs
python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/Deblur/train_Deblur_GOPRO.yml --launcher pytorch
```
#### Testing
- Models are available in ```'UVE-Net/experiments/'.```
- Organize your dataset(UVEB) like the above form.
- Run the following commands:
```
python basicsr/test.py -opt options/test/Deblur/test_Deblur_GOPRO.yml
cd results
```
## Experimental Results
#### Results on UVEB
> ![Alt text](/Pictures/3.png)
> > ![Alt text](/Pictures/4.png)
#### Results on UIEB
> ![Alt text](/Pictures/5.png)
> 
> UVE-Net did not use the UIEB dataset for training, and this result was obtained by testing the full UIEB dataset.
#### Results on CUDIE
> ![Alt text](/Pictures/6.png)
## Citation
```
@inproceedings{xie2024uveb,
  title={UVEB: A Large-scale Benchmark and Baseline Towards Real-World Underwater Video Enhancement},
  author={Xie, Yaofeng and Kong, Lingwei and Chen, Kai and Zheng, Ziqiang and Yu, Xiao and Yu, Zhibin and Zheng, Bing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={22358--22367},
  year={2024}
}
```
