# 3DSRnet
Official repository of 3DSRnet (ICIP2019)

We provide the training and test code along with the trained weights and the dataset (train+test) used for 3DSRnet.

If you find this repository useful, please consider citing our paper.

**Reference**:  
> Soo Ye Kim, Jeongyeon Lim, Taeyoung Na, Munchurl Kim. Video Super-Resolution Based on 3D-CNNS with Consideration of Scene Change.
*IEEE International Conference on Image Processing*, 2019.

**Extended Paper**:
> Soo Ye Kim, Jeongyeon Lim, Taeyoung Na, Munchurl Kim. 3DSRnet: Video Super-resolution using 3D Convolutional Neural Networks.
arXiv: 1812.09079 [link](https://arxiv.org/abs/1812.09079)

### Requirements
Our code is implemented using MatConvNet. (MATLAB required)

Appropriate installations of MatConvNet is *necessary* via the [official website](http://www.vlfeat.org/matconvnet/).  
Detailed instructions on installing MatConvNet can be found [here](http://www.vlfeat.org/matconvnet/install/).

The 3D convolution layer is implemented based on *pengsun*'s mex implementation in [GitHub](https://github.com/pengsun/MexConv3D).  
MexConv3D **must** be installed prior to executing any of the provided source code.

The code was tested under the following setting:  
* MATLAB 2017a  
* MatConvNet 1.0-beta25  
* CUDA 9.0, 10.0  
* cuDNN 7.1.4  
* NVIDIA TITAN Xp GPU

## Test code
### Quick Start
1. Download the source code in a directory of your choice \<source_path\>.
2. Download the test dataset (Vid4) from [this link](https://drive.google.com/file/d/16_rbLVFPObQc275yVeaM_Rg1TqvVa4CB) and place the 'test' folder in **\<source_path\>/data**
3. Place the files in **\<source_path\>/+dagnn/** to **\<MatConvNet\>/matlab/+dagnn**
4. Run **test.m**

### Description
We provide the pre-trained weights for the x2, x3 and x4 models in **\<source_path\>/net**.  
The test dataset (Vid4) can be downloaded from [here](https://drive.google.com/file/d/16_rbLVFPObQc275yVeaM_Rg1TqvVa4CB).
With test.m, the pre-trained models can be evaluated on the Vid4 benchmark.

**Remarks**
- You can change the SR scale factor (2, 3 or 4) by modifying the 'scale' parameter in the initial settings.
- You can change the video sequence by modifying the 'sequence_name' parameter in the initial settings.
- When you run this code, evaluation will be performed on PSNR and the .png prediction files will be saved in **\<source_path\>/pred/**

## Training code
### Quick Start
1. Download the source code in a directory of your choice \<source_path\>.
2. Download the train dataset from [here](https://drive.google.com/file/d/1Lav83JHZCNYInNbpf70CvTgDdBvCalhm) and place the 'train' folder in **\<source_path\>/data**
3. Place the files in **\<source_path\>/+dagnn/** to **\<MatConvNet\>/matlab/+dagnn**
4. Run **train.m**

### Description
The 3D-CNN model is specified in **net.m**.
The train dataset can be downloaded from [here](https://drive.google.com/file/d/1Lav83JHZCNYInNbpf70CvTgDdBvCalhm).  

**Remarks**
- You can change the SR scale factor (2, 3 or 4) by modifying the 'scale' parameter.
- The trained weights will be saved in **\<source_path\>/net/net_x***scale*

## Contact
Please contact me via email (sooyekim@kaist.ac.kr) for any problems regarding the released code.  
Note: We plan to provide the source code for the scene change module in the near future.
