
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
- [ ] Code Release.
- **2024.06.17:** The UVEB dataset has been published. 
- **2024.02.27:** Accepted by CVPR 2024!
- Test dataset is aviable at https://pan.baidu.com/s/1qPLYZ7tHJm2YeNqh8wxK1Q.  Extraction code: 1234
- Train dataset is aviable at https://pan.baidu.com/s/1A-Z6kbYAiCy95d_DLrdDBg. Exreaction code:1234
- For researchers outside of China, the full dataset can be aviable at https://terabox.com/s/1Mmz6ZAUv6h2GZFKZybD5mg.
## Dependencies
- Linux (Tested on Ubuntu 18.04)
- Python 3 (Recommend to use Anaconda)
- PyTorch 1.10.1: ```conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge```
- Install dependent packages :```pip install -r requirements.txt```
- Install DSTNet :```python setup.py develop```
## Get Started
#### Pretrained models
- Models are available in 'UVE-Net/pretrained/large_net_g.pth'
#### Dataset Organization Form
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
