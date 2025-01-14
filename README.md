# Adaptive Deep Joint Source-Channel Coding for One-to-Many Wireless Image Transmission

This is the code for paper "Adaptive Deep Joint Source-Channel Coding for One-to-Many Wireless Image Transmission". The model is implemented with PyTorch.

# Abstruct

Deep learning based joint source-channel coding (DJSCC) has recently made significant progress and emerged as a potential solution for future wireless communications. However, there are still several crucial issues that necessitate further in-depth exploration to enhance the efficiency of DJSCC, such as channel quality adaptability, bandwidth adaptability, and the delicate balance between efficiency and complexity. This work proposes an adaptive deep joint source-channel coding scheme (ADMIT) tailored for one-to-many wireless transmission scenarios. First, to effectively improve transmission performance, neighboring attention is introduced as the backbone for the proposed MDJSCC method. Second,  a channel quality adaptive module(CQAM) is designed based on multi-scale feature fusion, which seamlessly adapts to fluctuating channel conditions across a wide range of channel signal-to-noise ratios (CSNRs). Third, to be precisely tailored to different bandwidth resources, the channel gained adaptive module (CGAM) dynamically adjusts the significance of individual channels within the latent space, which ensures seamless varying bandwidth accommodation with a single model through bandwidth adaptation and symbol completion. Additionally, to mitigate the imbalance of loss across multiple bandwidth ratios during the training process, the gradient normalization (GradNorm) based training strategy is leveraged to ensure adaptive loss balancing. The extensive experimental results demonstrate that the proposed method significantly enhances transmission performance while maintaining relatively low computational complexity.

# Pipline

<div align="center">
  <img src="Sys_model2.png" width="85%">
</div>

<div align="center">
  <img src="fig_network_overview.png" width="85%">
</div>

# Usage

## Requirements

```
git clone https://github.com/llsurreal919/ADMIT

pip install -r requirements.txt
```

## Dataset

The test dataset can be downloaded from [kaggle](https://www.kaggle.com/datasets/drxinchengzhu/kodak24) .

The training dataset can be downloaded from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) .

## Training

We will release the tutorial soon.

## Testing

Pre-trained models can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1o7aqd5OgAIltr8NK6tmF-jkeq7xmc1HS/view?usp=sharing).

Example usage:

    python test_dyna_kodak.py

# Acknowledgement

The style of coding is borrowed from [Dynamic_JSCC](https://github.com/mingyuyng/Dynamic_JSCC) and partially built upon the [Neighborhood Attention Transformer](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer). We thank the authors for sharing their codes.

# Contact

If you have any question, please contact me (He Ziyang) via heziyang@bupt.edu.cn.
