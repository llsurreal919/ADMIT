# Dynamic_JSCC

This is the code for paper "High Efficiency Deep Joint Source-Channel Coding for One-to-Many Wireless Image Transmission". The model is implemented with PyTorch. 


# Usage
 

## Training
Example usage:
    
    python train_flicker.py --gpu_ids 0 --band 12 --batch_size 64

    python train_flicker.py --gpu_ids 0 --band 10 --batch_size 64 --pretrain --test_dir pre_12_0.375


## Testing

Pre-trained models can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1A6jIpwJs2BjoWd7Hycs1LV-Cos_UowEH/view).

Example usage:
    
    python test_dyna_kodak.py

    
## Acknowledgement
The style of coding is borrowed from [Dynamic_JSCC](https://github.com/mingyuyng/Dynamic_JSCC) and partially built upon the [Neighborhood Attention Transformer](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer). We thank the authors for sharing their codes.

## Contact
If you have any question, please contact me (He Ziyang) via s210131061@stu.cqupt.edu.cn.