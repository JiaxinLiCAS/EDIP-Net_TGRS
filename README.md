# EDIP-Net_TGRS
Enhanced Deep Image Prior for Unsupervised Hyperspectral Image Super-resolution, TGRS. (PyTorch)

# $\color{red}{有问题直接添加 我的微信(WeChat): BatAug，欢迎交流与合作}$

## 本人还提出了其余多个开源的高光谱-多光谱超分融合代码，可移步至[GitHub主页下载](https://github.com/JiaxinLiCAS) 

[Jiaxin Li](https://www.researchgate.net/profile/Li-Jiaxin-20), [Ke Zheng](https://www.researchgate.net/profile/Ke-Zheng-9), [Lianru Gao](https://scholar.google.com/citations?hl=en&user=f6OnhtcAAAAJ), [Zhu Han](https://scholar.google.com/citations?user=AtmD3QUAAAAJ&hl=zh-CN), [Zhi Li](https://ieeexplore.ieee.org/author/37085683916), and [Jocelyn Chanussot](https://scholar.google.com/citations?user=6owK2OQAAAAJ&hl=zh-CN&oi=ao)

Our paper is accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS). 

文章可在这里下载🖼️[**PDF**](./Imgs/EDIP-Net.pdf)，The final version can be downloaded in  🖼️[**PDF**](./Imgs/EDIP-Net.pdf) 


这是我的[谷歌学术](https://scholar.google.com/citations?user=aSPDpmgAAAAJ&hl=zh-CN)和[ResearchGate](https://www.researchgate.net/profile/Jiaxin-Li-lijiaxin?ev=hdr_xprf)，More information can be found in my [Google Scholar Citations](https://scholar.google.com/citations?user=aSPDpmgAAAAJ&hl=zh-CN) and my [ResearchGate](https://www.researchgate.net/profile/Jiaxin-Li-lijiaxin?ev=hdr_xprf)

<img src="./Imgs/fig1.png" width="666px"/>

**Fig.1.** Overall Pipeline of proposed method, abbreviated as EDIP-Net, for the task of unsupervised hyperspectral image super-resolution.

## 文件结构 Directory structure
<img src="./Imgs/fig2.png" width="200px"/>

**Fig.2.** Directory structure. There are four folders and one main.py file in EDIP-Net_TGRS-main.

### checkpoints
这个文件夹用于储存训练中的所有结果，这里给出了TG数据的示例。如果你直接运行main.py,将会在`TGSF12_band240_S1_0.001_2000_2000_S2_0.004_2000_2000_S3_0.004_7000_7000`文件夹中生成以下的这些文件

This folder is used to store the results and a folder named `TGSF12_band240_S1_0.001_2000_2000_S2_0.004_2000_2000_S3_0.004_7000_7000` is given as an example.

- `BlindNet.pth` is the trained parameters of Stage One. 第一阶段网络训练好的参数

- `estimated_lr_msi.mat` is the estimated LrMSI in Stage One. 第一阶段估计到的LrMSI

- `estimated_psf_srf.mat` is the estimated PSF and SRF. 第一阶段估计到的PSF和SRF

- `gt_lr_msi.mat` is the gt lr_msi. LrMSI的真值

- `hr_msi.mat` and `lr_hsi.mat`  are simulated results as the input of our method. 由输入的TG数据仿真得到的LrHSI和HrMSI

- `opt.txt` is the configuration of our method. 存储本次实验的所有配置，包括超参数以及数据名称等，由model里的config.py决定
  
- `Out_fhsi_S2.mat` and `Out_fhsi_S3.mat` is the estimated HrHSI from S2 and S3 in the stream of lrhsi. 由hsi分支在第二和第三阶段生成的HrHSI估计结果

- `Out_fmsi_S2.mat` and `Out_fmsi_S3.mat` is the estimated HrHSI from S2 and S3 in the stream of hrmsi. 由msi分支在第二和第三阶段生成的HrHSI估计结果

- `psf_gt.mat` and  `srf_gt.mat` are the GT PSF and SRF. PSF 和 SRF的真值

- `srf_Out_S4.mat` is the final estimation of our method. 本方法最终估计的结果

- `Stage1.txt` is the training accuracy of Stage One.第1阶段的精度

- `Stage2.txt` is the training accuracy of Stage Two.第2阶段的精度
  
- `Stage3.txt` is the training accuracy of Stage Three.第3阶段的精度

- `Stage4.txt` is the training accuracy of Stage Four.第4阶段的精度
  
### data
This folder is used to store the ground true HSI and corresponding spectral response of multispectral imager, aiming to generate the simulated inputs. The TianGong-1 HSI data and spectral response of WorldView 2 multispectral imager are given as an example here.

这里给出了一个示例。EDIP-Net文件里的TG文件夹是TG数据的真值，spectral_response是用来仿真HrMSI的光谱响应函数。

### model
This folder consists of ten .py files, including 
- `__init__.py`

- `config.py`: all the hyper-parameters can be adjusted here. 本方法所有需要调整的参数，包含数据读取地址以及模型超参数等

- `dip.py`: the stage three. 对应第三阶段

- `evaluation.py`: to evaluate the metrics. 评价指标计算

- `network_s2.py`: the network used in the Stage Two. 阶段二所需要的网络模型

- `network_s3.py`: the network used in the Stage Three. 阶段三所需要的网络模型

- `read_data.py`: read and simulate data. 读取数据和仿真数据

- `select.py`: generate the final result from Stage three. 对阶段三的两个输出进行融合

- `spectral_up.py`: the network in the Stage Two. 对应阶段二

- `srf_psf_layer.py`: the network in the Stage One. 对应阶段一

### main
- `main.py`: main.py 运行该文件，生成目标图像

## 如何运行我们的代码 How to run our code
- Requirements: codes of networks were tested using PyTorch 1.9.0 version (CUDA 11.4) in Python 3.8.10 on Windows system.

- Parameters: all the parameters need fine-tunning can be found in `config.py`. 本方法所有需要调整的参数都在此.py中

- Data: put your HSI data and MSI spectral reponse in `./data/EDIP-Net/TG` and `./data/EDIP-Net/spectral_response`, respectively. The TianGong-1 HSI data and spectral response of WorldView 2 multispectral imager are given as an example here.

  将你的高光谱数据以及用于仿真HrMSI的光谱响应放到对应文件夹中，这里用TG数据作为示例

- Run: just simply run `main.py` after adjusting the parameters in `config.py`.
  在对应文件夹放置你的数据后，调整 `config.py`后的参数，即可运行`main.py`

- Results: one folder named `TGSF12_band260_S1_0.001_3000_3000_S2_0.004_2000_2000_S3_0.004_7000_7000` will be generated once `main.py` is run and all the results will be stored in the new folder.
  当你运行本代码后，将会生成` TGSF12_band260_S1_0.001_3000_3000_S2_0.004_2000_2000_S3_0.004_7000_7000` 文件夹，里面存储所有结果


## 如何联系我们 Contact
遇到任何问题，包括但不限于代码调试、数据仿真、运行结果等，随时添加
$\color{red}{我的微信(WeChat): BatAug，欢迎交流与合作}$

If you encounter any bugs while using this code, please do not hesitate to contact us. lijiaxin203@mails.ucas.ac.cn
