# XDT-CXR: Investigating Cross-Disease Transferability in Zero-Shot Binary Classification of Chest X-Rays 

[Umaima Rahman](https://scholar.google.com/citations?user=nTZQZ0EAAAAJ&hl=en)<sup>1</sup>, [Abhishek Basu](https://iabh1shekbasu.github.io/)<sup>1</sup>, [Muhammad Uzair Khattak](https://muzairkhattak.github.io)<sup>1</sup>, [Aniq Ur Rahman](https://eng.ox.ac.uk/people/aniq-ur-rahman/)<sup>2</sup>

<sup>1</sup> Mohamed Bin Zayed University of Artificial Intelligence  
<sup>2</sup> Oxford University

---

## Table of Contents
- [Overview](#overview)
  - [Paper Highlights](#paper-highlights)
  - [Method Overview](#method-overview)
- [🚀 Release](#rocket-release)
- [Prerequisites](#prerequisites)
  - [Hardware](#hardware)
  - [Environment](#environment)
  - [Datasets](#datasets)
- [Run XDT-CXR](#run-xdt-cxr)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Overview

This repository provides the official PyTorch implementation of our paper **"XDT-CXR: Investigating Cross-Disease Transferability in Zero-Shot Binary Classification of Chest X-Rays"**.

### Paper Highlights
- **Title:** XDT-CXR: Investigating Cross-Disease Transferability in Zero-Shot Binary Classification of Chest X-Rays
- **Authors:** Umaima Rahman, Abhishek Basu, Muhammad Uzair Khattak, Aniq Ur Rahman
- **Conference:** Machine Learning for Healthcare, 2024
- **[arXiv Preprint](https://arxiv.org/abs/2408.11493v1)**

### Method Overview
* The embedding space of a vision encoder is leveraged with a kernel transformation to distinguish between diseased and non-diseased classes.
* The model is trained as a binary classifier on one pulmonary disease to perform zero-shot classification on another novel pulmonary disease.

<p align="center">
  <img src="utils/Overview_method.png" alt="XDT-CXR Method Overview" width="600">
</p>

## :rocket: Release

* **(August 21, 2024)**
  * Paper uploaded on arXiv.
* **(July 4, 2024)**
  * Paper accepted at Machine Learning for Healthcare, 2024.

## Prerequisites

### Hardware
This implementation is designed for a single GPU with 24GB of memory.

### Environment 
The code is tested on PyTorch **1.x**.

### Datasets 

We suggest downloading all datasets to a root directory (`${DATA_ROOT}`) and renaming the directory of each dataset as suggested in `${ID_to_DIRNAME}` in `./data/datautils.py`. This organization will allow you to evaluate multiple datasets within the same run. If this approach is not feasible, you can evaluate different datasets separately and change the `${DATA_ROOT}` accordingly in the bash script.

We have used four publicly available Chest X-Ray (CXR) datasets in our experiments:

- **[Guangzhou-PN (G)]():** Pneumonia dataset published in Kermany et al. (2018), consisting of 5,232 pediatric CXR images from patients aged one to five years at Guangzhou Women and Children’s Medical Center, China. The dataset includes 1,349 normal and 3,883 pneumonia x-rays (2,538 bacterial pneumonia and 1,345 viral pneumonia).
  
- **[Montgomery-TB (M)]():** Montgomery dataset from Jaeger et al. (2014), a collection of 138 posterior-anterior CXR images collected as part of the tuberculosis control program by the Department of Health and Human Services of Montgomery County, MD, USA. The collection includes 80 normal and 58 abnormal x-rays showing manifestations of tuberculosis.
  
- **[Shenzhen-TB (S)]():** Shenzhen Hospital dataset from Jaeger et al. (2014), consisting of 662 CXR images collected as part of routine care at the No.3 Hospital in Shenzhen, Guangdong Province, China. The dataset includes 326 normal and 336 abnormal X-rays.
  
- **[Covid (C)]():** Covid dataset from Rahman et al. (2021), comprising 1,383 test CXR images, making it the largest public COVID-positive database. We divided this set into equal numbers of images in each class, positive (COVID) and negative (non-COVID).

## Run XDT-CXR

To run the XDT-CXR model, follow the instructions below:

```bash
# Example command to run the model
python train.py --dataset_path ${DATA_ROOT} --config config.yaml
```

## Citation
If you find our code useful or our work relevant, please consider citing: 
```
@article{rahman2024xdt,
  title={XDT-CXR: Investigating Cross-Disease Transferability in Zero-Shot Binary Classification of Chest X-Rays},
  author={Rahman, Umaima and Basu, Abhishek and Khattak, Muhammad Uzair and Rahman, Aniq Ur},
  journal={arXiv preprint arXiv:2408.11493},
  year={2024}
}
```

## Acknowledgements
We thank the authors of [TPT](https://github.com/azshue/TPT/) and [DeYO](https://github.com/Jhyun17/DeYO) for their open-source implementation and instructions on data preparation.
