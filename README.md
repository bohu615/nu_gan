# nu_gan
This repository contains the Python implementation for our paper 

Unsupervised Learning for Cell-level Visual Representation in Histopathology Images with Generative Adversarial Networks 
Bo Hu#,Ye Tang#,Eric I-Chao Chang, Yubo Fan, Maode Lai and Yan Xu* (* corresponding author; # equal contribution)

[*arxiv*](https://arxiv.org/abs/1711.11317) | [*IEEE*](https://ieeexplore.ieee.org/document/8402089).  

Requirements
=================
* [*Pytorch*](https://github.com/pytorch/pytorch)
* [*HistomicsTK*](https://github.com/DigitalSlideArchive/HistomicsTK)

Usage
=================

* Unsupervised Cell-level Classification:
```shell
python nu_gan.py --task 'cell_representation'
```

* Unsupervised Image-level Classification:
```shell
python nu_gan.py --task 'image_classification'
```

* Neuclei Segmentation:
```shell
python nu_gan.py --task 'cell_segmentation'
```

DATA
=================
uploading...
