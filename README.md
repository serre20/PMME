# PMME: Spatio-Temporal Few-shot Learning via Pattern Matching with Memory Enhancement

This project uses code from previous papers for MTPB and STGP, which are organized as follows:

- **[MTPB](https://github.com/zhyliu00/MTPB)**: Uses the official MTPB code provided by their original papers, located in the `MTPB` folder. 
- **[STGP](https://github.com/hjf1997/STGP)**: Uses the official STGP code provided by their original papers, located in the `STGP` folder. 
- **PMME and Other Baselines**: The code for PMME and other baselines is included in the `PMME_and_Others` folder.

This repository does not provide implementations of STGFSL, TransGTR, GPD, or UniST. Their results, used for comparison in our paper, are cited from the **[MTPB paper](https://dl.acm.org/doi/full/10.1145/3727622)** (published in ACM TKDD 2025).

## Data Preparation
The dataset is available at [Google Drive](https://drive.google.com/drive/folders/1UrKTgR27YmP9PjJ-FWv4SCDH3zUxtc5R?usp=share_link).
Before running the code, **please unzip `data.zip` and place the data files in the designated directories** within each folder (`MTPB`, `STGP`, and `PMME_and_Others`).  
Make sure to follow any additional data placement instructions inside each folder's `README` or comments.

## Main Dependencies

Install the main dependencies with:

```bash
pip install torch==2.6.0
pip install torch-geometric==2.6.1
pip install gpytorch==1.14
pip install PyYAML==6.0.2
pip install geomloss==0.2.6
pip install einops==0.8.1
