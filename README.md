# cs182-project
Group project repo for cs182 

### Environment setup

Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install), then run the following commands
```bash
# 1. Create a conda environment
conda env create -f environments.yaml
conda activate cs182-rep

```
[optional] Add the following to the end of your .bashrc file to enter this environment automatically upon shell creation: `conda activate cs182-rep`

# Steps to Running Code

## Step 1: Download the fineweb dataset
```bash
export DATA_DIR = "absolute/path/to/data"
python3 process_data.py --name fineweb1B
```

## Step 2: Login to wandb
Run `wandb login` to login to your wandb account.


## Step 3: Run a sweep
e.g.
```bash
wandb sweep sweeps/phase2-muon-pe-mod-all.yaml
```

# Reproducing Plots

## Download data [if you want to use your own sweep data]
Download your wanb sweep data to your local computer
`download-data.ipynb`

## Generate plots [use our included data]
Map the name of the sweep you care -> sweep_id 

To create the plots, run this notebook:
`analyze-data.ipynb`

