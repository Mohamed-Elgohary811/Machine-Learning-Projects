# N-BaIoT IoT Botnet Attack Classification
 
Multi-class classification of IoT botnet traffic using the [N-BaIoT dataset](https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset), combining a **Transformer encoder (FT-Transformer style)** for representation learning with **XGBoost** for final classification.
 
## Overview
 
The goal is to predict the exact traffic class for network flows collected from 9 real IoT devices:
 
- `benign`
- `mirai.ack`, `mirai.scan`, `mirai.syn`, `mirai.udp`, `mirai.udpplain`
- `gafgyt.combo`, `gafgyt.junk`, `gafgyt.scan`, `gafgyt.tcp`, `gafgyt.udp`
**11 classes total**, using the 115 statistical traffic features provided by the dataset.
 
## Pipeline
 
1. **Data acquisition** — download the N-BaIoT dataset from Kaggle.
2. **Loading & labeling** — load benign, Mirai, and Gafgyt traffic files for all 9 devices and assign class labels (2 devices have no Mirai traffic).
3. **Preprocessing** — drop missing/duplicate rows, label-encode targets, split into train/val/test (70/15/15, stratified), and standard-scale features.
4. **Transformer training** — train a tabular Transformer (FT-Transformer style: per-feature tokenization + `[CLS]` token) as an 11-class classifier over the 115 features, with early stopping and LR scheduling.
5. **Embedding extraction** — extract `[CLS]` token embeddings from the trained Transformer for train/val/test sets.
6. **Feature fusion** — concatenate the learned embeddings with the original 115 scaled features.
7. **XGBoost training** — train an XGBoost multiclass classifier (`multi:softprob`) on the combined feature set.
8. **Evaluation** — accuracy, weighted/macro precision, recall, F1, macro ROC-AUC, and feature importance analysis.
## Model Architecture
 
- **Feature Tokenizer**: each of the 115 scalar features is projected into its own learned `d_model`-dimensional embedding (per-feature linear layer).
- **Transformer Encoder**: multi-head self-attention over feature tokens + a learnable `[CLS]` token, `d_model=64`, `4 heads`, `3 layers`.
- **XGBoost**: `n_estimators=400`, `max_depth=6`, `learning_rate=0.05`, trained on original features + Transformer embeddings.
## Requirements
 
```
torch
numpy
pandas
scikit-learn
xgboost
joblib
matplotlib
seaborn
opendatasets
requests
```
 
This notebook was built for **Google Colab** and mounts Google Drive for storing models, embeddings, and logs. To run it elsewhere, replace the Google Drive mounting cell with a local project directory and skip the Kaggle `opendatasets` download step if you already have the data locally.
 
## Dataset
 
- Source: [N-BaIoT Dataset on Kaggle](https://www.kaggle.com/datasets/mkashifn/nbaiot-dataset)
- You'll need a Kaggle account/API token to download it via `opendatasets`.
- Place/extract the CSVs under `data/nbaiot-dataset/` (or update `DATA_DIR` in the notebook).
## Usage
 
1. Open `N-BaIoT_IoT_Botnet_Attack_Classification.ipynb` in Google Colab (or Jupyter).
2. Run the cells in order:
   - Mount Drive / set up project folders
   - Download and load the dataset
   - Preprocess and split the data
   - Train the Transformer
   - Extract embeddings and train XGBoost
   - Review evaluation metrics and feature importance plots
3. Trained artifacts (Transformer weights, scaler, label encoder, embeddings, XGBoost model) are saved to the configured project directory for reuse.
## Outputs
 
- `best_fttransformer_model.pth` — trained Transformer weights
- `train/val/test_embeddings.npy` — extracted `[CLS]` embeddings
- Saved `StandardScaler` and `LabelEncoder` (via `joblib`)
- Trained XGBoost model (`.json`)
- Training curves (loss/accuracy) and top-20 feature importance plot
## Results
 
The final XGBoost model trained on **original features + Transformer embeddings** is evaluated with accuracy, weighted/macro precision-recall-F1, and macro-averaged ROC-AUC. See the notebook's evaluation section for exact numbers on your run.
 
## License
 
Add a license of your choice (e.g., MIT) here.
