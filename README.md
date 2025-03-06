
Enhanced Deep Learning Model for Multi-Modal Brain Image Registration
This repository contains an advanced brain image registration framework based on deep learning, attention mechanisms, and deformation constraints. It is specifically designed for multi-modal neuroimaging registration, particularly in stroke analysis using MICCAI ISLES 2018 dataset.



Project Overview
Brain image registration is a critical step in medical image analysis, aligning different imaging modalities to a common anatomical space. This repository provides:
- A U-Net-based registration model with channel attention and multi-branch deformation estimation.
- Integration with the IIT Human Brain Atlas for anatomical correspondence matching.
- Optimized patch-based registration with 5-fold cross-validation.
- Evaluation using MSE, SSIM, Dice Coefficient, and Jacobian Determinants.



Features
Multi-Modal Registration: Works with CT, CBF, CBV, Tmax, and MTT images.  
Deep Learning-Based: Uses U-Net with a multi-branch deformation model.  
Channel Attention Mechanism: Improves feature representation for better alignment.  
Patch-Based Processing: Reduces computational complexity while preserving details.  
Cross-Validation Support: Ensures robust generalization across datasets.  
Scalable to Other Neuroimaging Datasets: Can be adapted for different modalities.  



Dataset
The model is tested on MICCAI ISLES 2018 dataset:
- 103 stroke cases (63 training, 40 testing)
- Multi-modal scans: CT, CBF, CBV, MTT, Tmax
- Pre-aligned using Montreal Neurological Institute (MNI) Atlas
- Manual lesion segmentation using DWI as ground truth
- The dataset is available at: [MICCAI ISLES 2018](https://www.isles-challenge.org/)

Additionally, the IIT Human Brain Atlas is used for anatomical alignment:
- Gray matter probability maps from IIT Atlas
- Provides voxel-wise estimations of anatomical structures
- Available at: [IIT Brain Atlas](https://www.nitrc.org/projects/iit/)



Installation
Before running the model, install the required dependencies:
```bash
pip install -r requirements.txt
```
 Requirements
- Python 3.8+
- TensorFlow 2.11+
- Scikit-learn, NumPy, SciPy
- ITK, SimpleITK (for medical image processing)
- Spektral (for potential graph-based models)
- HDF5 (for efficient data storage)



 🔧 Usage
 Command to Run Registration
```bash
python vs6.py --dataset /path/to/dataset/ \
              --template /path/to/template.nii.gz \
              --output /path/to/output/ \
              --weights /path/to/weights/ \
              --use-ensemble

 Command Line Arguments
| Argument         | Description  |
|-----------------|-------------|
| `--dataset`     | Path to the dataset folder |
| `--template`    | Path to the IIT brain template |
| `--output`      | Directory for saving results |
| `--weights`     | Path to pre-trained model weights |
| `--use-ensemble`| Enable ensemble model inference |


 Model Architecture
 1️Encoder
- Extracts multi-scale features using convolutional layers and instance normalization.
- Implements spatial dropout (0.3) for regularization.
- Uses residual addition fusion for feature refinement.

 2️Channel Attention
- Enhances relevant anatomical features.
- Uses global average pooling and a dual-branch dense layer network.
- Applied through sigmoid activation for feature reweighting.

  Multi-Branch Deformation Estimation
- Five parallel branches: Main, CBF, CBV, Tmax, MTT.
- Each branch predicts spatial deformations for specific brain structures.
- Deformation predictions are weighted and combined.

Loss Function & Optimization
The model optimizes using:
- Mean Squared Error (MSE) for intensity-based alignment.
- Structural Similarity Index (SSIM) to ensure perceptual alignment.
- Dice Coefficient (DSC) to compare structural overlap.
- Jacobian Determinant Regularization to prevent unrealistic deformations.
- Smoothness Loss to penalize abrupt transformations.

 Performance Metrics
| Modality | SSIM (↑) | MSE (↓) |
|----------|---------|---------|
| CBF      | 0.6934 ± 0.0562 | 0.016857 ± 0.008662 |
| CBV      | 0.6958 ± 0.0542 | 0.016966 ± 0.008654 |
| Tmax     | 0.6692 ± 0.0606 | 0.024300 ± 0.010339 |
| MTT      | 0.6572 ± 0.0630 | 0.030049 ± 0.011611 |

Mean Deformation: 1.1143 pixels  
Folding Percentage: 0.00% (Ensures anatomically valid transformations)  



Evaluation & Results
The model achieves:
- SSIM = 0.6789 (Average across modalities)
- MSE = 0.022043
- Zero-folding percentage (Ensures anatomically consistent transformations)
- Robust deformation constraints, avoiding extreme distortions

Key Findings
- The highest registration accuracy is observed for CBV and CBF modalities.
- Multi-branch deformation estimation significantly improves registration.
- Channel attention enhances the ability to align fine-grained anatomical structures.



References
1. MICCAI ISLES 2018 Challenge: [https://www.isles-challenge.org/](https://www.isles-challenge.org/)
2. IIT Human Brain Atlas: [https://www.nitrc.org/projects/iit/](https://www.nitrc.org/projects/iit/)
3. Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.



Contributing
Feel free to open an issue or submit a pull request to improve this repository.  
For major contributions, please discuss the proposed changes in GitHub Discussions.



 License
This project is open-source under the MIT License.



Contact
For any questions or collaborations, please reach out via:
📧 Email: [parvezamu@gmail.com]  
🌐 GitHub: [https://github.com/parvezamu](https://github.com/parvezamu)  

