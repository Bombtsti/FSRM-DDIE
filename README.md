# FSRM-DDIE

## Environment

Create and activate virtual env using conda:

```
conda create -n "env_name" python=3.7
conda activate "env_name"
```

Install Pytorch and Torch_geometric:

```
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

pip install torch_scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch_sparse==0.6.17 -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install torch_geometric=2.3.1
```

Other python packages installed from the requirments.txt:

```
pip install -r requirements.txt
```

## Dataset

The DengDDIE dataset is available at: [https://drive.google.com/file/d/1K-5lQI2Dy5dyp2SrY8-QoaWGm0fgyVNP/view?usp=sharing](https://drive.google.com/file/d/1K-5lQI2Dy5dyp2SrY8-QoaWGm0fgyVNP/view?usp=sharing)

The DrugBank dataset is uploaded at the METADDIEdata folder.


## Train
Please put FSRM-DDIE and METADDIEdata in the same folder. 
Before running the script file, please tune the hyperparameter.

```
python main.py
```
