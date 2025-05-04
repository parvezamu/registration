
Enhanced Deep Learning Model for Multi-Modal Brain Image Registration

This repository contains an advanced brain image registration framework based on deep learning, attention mechanisms, and deformation constraints. It is specifically designed for multi-modal neuroimaging registration, particularly in stroke analysis using the MICCAI ISLES 2018 dataset.

Project Overview

Brain image registration is a critical step in medical image analysis, aligning different imaging modalities to a common anatomical space. This repository provides:

- A U-Net-based registration model with channel attention and multi-branch deformation estimation.
- Integration with the IIT Human Brain Atlas for anatomical correspondence matching.
- Optimized patch-based registration with 5-fold cross-validation.
- Evaluation using MSE, SSIM, Dice Coefficient, and Jacobian Determinants.

Features

- Multi-Modal Registration: CT, CBF, CBV, Tmax, and MTT images.
- Deep Learning-Based: U-Net with multi-branch deformation model.
- Channel Attention Mechanism: Enhances feature relevance.
- Patch-Based Processing: Efficient registration with anatomical precision.
- Cross-Validation Support: Robust generalization.
- Adaptable Framework: Easily extendable to other neuroimaging datasets.

Dataset

The model is tested on the [MICCAI ISLES 2018 dataset](https://www.isles-challenge.org/):
- 103 stroke cases (63 training, 40 testing)
- Modalities: CT, CBF, CBV, MTT, Tmax
- Pre-aligned to MNI space
- DWI-based manual lesion segmentation as ground truth

Atlas used:
- [IIT Human Brain Atlas](https://www.nitrc.org/projects/iit/)
- Gray matter probability maps
- High-resolution template for anatomical alignment

Installation

```bash
pip install -r requirements.txt
```

Requirements
- Python 3.8+
- TensorFlow 2.11+
- NumPy, SciPy, scikit-learn
- ITK, SimpleITK
- Spektral (optional)
- h5py

Usage

Run Registration

```bash
python vs6.py --dataset /path/to/dataset/ \
              --template /path/to/template.nii.gz \
              --output /path/to/output/ \
              --weights /path/to/weights/ \
              --use-ensemble
```

Command Line Arguments

| Argument         | Description                              |
|------------------|------------------------------------------|
| `--dataset`      | Path to the dataset folder               |
| `--template`     | Path to the IIT brain template           |
| `--output`       | Directory for saving registered results  |
| `--weights`      | Path to pre-trained model weights        |
| `--use-ensemble` | Enable 5-fold ensemble inference         |

Each patient folder must include:
- `CT_CBF`, `CT_CBV`, `CT_MTT`, `CT_Tmax` (.nii.gz)

Model Architecture

1. Encoder  
- 4-level feature extractor with residual + dropout
- Instance normalization and ReLU activation

2. Channel Attention  
- Dual-branch attention with sigmoid gating
- Emphasizes informative features

3. Multi-Branch Deformation Estimation  
- Five parallel branches: Main, CBF, CBV, Tmax, MTT  
- Weighted combination to yield smooth deformation field

Loss Functions

- MSE – voxel-wise intensity similarity  
- SSIM – structural similarity  
- Dice Coefficient – segmentation overlap (optional)  
- Jacobian Determinant Regularization – topology preservation  
- Smoothness Loss – penalizes abrupt deformation changes  

Performance Metrics

| Modality | SSIM ↑         | MSE ↓             |
|----------|----------------|-------------------|
| CBF      | 0.6934 ± 0.0562 | 0.016857 ± 0.0087 |
| CBV      | 0.6958 ± 0.0542 | 0.016966 ± 0.0087 |
| Tmax     | 0.6692 ± 0.0606 | 0.024300 ± 0.0103 |
| MTT      | 0.6572 ± 0.0630 | 0.030049 ± 0.0116 |

- Mean Deformation: 1.1143 pixels  
- Folding Percentage: 0.00% (anatomically valid)

Evaluation Summary

- SSIM: 0.6789 average
- MSE: 0.022043
- Zero-folding percentage (valid topology)
- Highest SSIM in CBV and CBF
- Robust generalization via 5-fold ensemble

Key Takeaways

- Channel attention improves anatomical alignment.
- Multi-branch design allows modality-aware registration.
- Modular design supports other imaging contexts.

References

1. [ISLES 2018 Challenge](https://www.isles-challenge.org/)  
2. [IIT Human Brain Atlas](https://www.nitrc.org/projects/iit/)  
3. Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation (MICCAI 2015)

Contributing

Open issues and pull requests are welcome!  
For major changes, use GitHub Discussions first.

📄 License

This project is licensed under the MIT License.

Contact

- 📧 Email: [parvezamu@gmail.com](mailto:parvezamu@gmail.com)  
- 🌐 GitHub: [https://github.com/parvezamu](https://github.com/parvezamu)
