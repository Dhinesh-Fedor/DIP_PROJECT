# A Lightweight Hybrid CNN-Transformer Model for Smoke Removal in Laparoscopic	Surgery Images 
Mohammed Ashlab N - 23MIA1088 | Dhinesh Kumar M - 23MIA1102 | Srishreyan S - 23MIA1081

# Abstract

Laparoscopic surgeries are highly dependent on visual clarity, but the generation of surgical smoke during procedures such as cauterization and ablation often obscures the surgeon’s view and degrades image quality. To overcome this issue, we propose a Progressive Frequency-Aware Network (PFAN)—a lightweight hybrid deep learning architecture that integrates Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) within a Generative Adversarial Network (GAN) framework. The proposed model progressively extracts and fuses high-frequency local and low-frequency global features in the frequency domain to restore clear, artifact-free laparoscopic images. PFAN introduces two novel components: the Multi-scale Bottleneck-Inverting (MBI) Block for multi-scale high-frequency extraction and the Locally-Enhanced Axial Attention Transformer (LAT) Block for global low-frequency feature representation. The model is trained using synthetic smoke images generated via the Blender engine to simulate realistic laparoscopic smoke. Experimental evaluation on the Cholec80 dataset demonstrates that PFAN achieves superior quantitative results in PSNR, SSIM, and CIEDE2000 metrics compared to state-of-the-art approaches such as Pix2Pix, CycleGAN, and DCP, while maintaining only 629K parameters, enabling real-time deployment on resource-constrained surgical systems.


## Overall Architecture
<img width="580" height="278" alt="image" src="https://github.com/user-attachments/assets/48c40e67-2711-427c-a530-8e7a8abeaeac" />



## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/Dhinesh-Fedor/DIP_PROJECT.git
cd DIP_PROJECT
```

- Install [PyTorch](http://pytorch.org) and 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.


### Dataset

![dataset](/3.png)

We used images from the [Cholec80](http://camma.u-strasbg.fr/datasets) dataset and sampled 1,500 images at 20-second intervals from videos, selecting 660 representative smoke-free images. As detailed above, we added synthetic random smoke, yielding 660 image pairs.

Composite Images [[Google Drive]](https://drive.google.com/file/d/1n_-AzgmQcUWt9VMkTvuZnewJXNHCB7_o/view?usp=drive_link)


### Train/Test
- Download the Composite Images dataset
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- To log training progress and test images to W&B dashboard, set the `--use_wandb` flag with train and test script
- Train a model(`sh ./scripts/train_pfan.sh`)::
```bash
python train.py --dataroot ./datasets/composite --name pfan
```

- Test the model (`bash ./scripts/test_pfan.sh`):
```bash
#!./scripts/test_pfan.sh
python test.py --dataroot ./datasets/composite --name pfan
```
- The test results will be saved to a html file here: `./results/pfan/test_latest/index.html`. You can find more scripts at `scripts` directory.


## Representative Results
![representive_results](2.png)

## Acknowledgments
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
