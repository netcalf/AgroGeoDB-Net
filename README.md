# AgroGeoDB-Net 🚜📡  
---

## 🔍 Overview
AgroGeoDB-Net is a geometry-aware spatiotemporal deep-learning framework for **point-wise field–road classification** from GNSS trajectories.  
It addresses three long-standing challenges in agricultural trajectory analysis:

- **Severe class imbalance** (field ≫ road)  
- **Spatial noise & sampling inconsistency**  
- **Long-range temporal dependencies**  

This repo provides the official implementation, datasets, and experimental results for the paper  
*“AgroGeoDB-Net: A DBSCAN-Guided Augmentation and Geometric-Similarity Regularised Framework for GNSS Field–Road Classification in Precision Agriculture”*.

---

## 🏗️ Architecture at a Glance
AgroGeoDB-Net consists of three tightly coupled components:

### 1. **Density-Aware Local Interpolator (DALI)**
- DBSCAN-based road clustering  
- Road-fragment interpolation guided by spatial density & directional coherence  
- Effectively oversamples minority road points  

### 2. **Motion–Spatial Descriptor (14-D)**
Each GNSS point is converted into a feature vector combining:
- Motion dynamics: speed/acceleration/heading/curvature  
- Local density & directional coherence  
- Multi-scale spacing descriptors  

### 3. **Density-Regularised VAE + Residual BiLSTM**
- VAE learns geometry-aligned latent embeddings  
- Residual BiLSTM captures bidirectional long-range temporal structure  
- Linear head outputs per-point field/road label  

---

## ✨ Key Features
- **DBSCAN-guided data augmentation** for road segments  
- **14-dim handcrafted descriptors** + learned deep representations  
- **Density-weighted focal loss + density-regularised KL divergence**  
- **Residual BiLSTM** stabilises sequence modelling  
- Strong **cross-dataset generalisation** (Harvester & Tractor)

---
## 📂 Datasets

- **Wheat / Paddy / Corn / Tractor**  
  https://github.com/Agribigdata/dataset_code

- **Harvester**  
  https://github.com/AgriMachineryBigData/Field-road_mode_mining
