
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
update later
## Update
- [ ] Data Release.
- [ ] Code Release.
- **2024.02.27:** Accepted by CVPR 2024!
- Test dataset is aviable at [https://pan.baidu.com/s/14dl6K_KKi5of8aafhLqOXA](https://pan.baidu.com/s/1SJgSEbN_JCL1Tw1mwV2fAg)  extract code: 1234
- Train dataset is aviable at https://pan.baidu.com/s/1WFfK0Oxj9Mf4g7SOCVSqMg. exreact code:1234
- All paired data has uploaded to Baidu Cloud Disk.
We are trying to upload a copy of dataset to the Terabox cloud drive for researchers outside of China to download and use the data conveniently.
## Experimental Results
#### Results on UVEB
> ![Alt text](/Pictures/3.png)
> > ![Alt text](/Pictures/4.png)
#### Results on UIEB
> ![Alt text](/Pictures/5.png)
#### Results on CUDIE
> ![Alt text](/Pictures/6.png)
