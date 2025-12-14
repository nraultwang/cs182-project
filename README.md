# cs182-project
Group project repo for cs182 

### Environment setup

Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install), then run the following commands
```bash
# 1. Create a conda environment
conda env create -f environments.yaml
conda activate cs182-rep

```
3. [optional] Add the following to the end of your .bashrc file to enter this environment automatically upon shell creation: `conda activate cs182-rep`

## Step 1: Download data

os.getenv file -> looks for data_dir -> where users install the data.
```bash
export DATA_DIR = "absolute/path/to/data"
python3 process_data.py --name fineweb1B
```

## Step 2: Login to wandb

## Step 3: Run a sweep

# Reproducing Plots

## Download data
Download your wanb sweep data to your local computer
`download-data.ipynb`

## Generate plots
Map the name of the sweep you care -> sweep_id 

Three places: 
1. 
2. 
3. 
To create the plots 
`analyze-data.ipynb`

