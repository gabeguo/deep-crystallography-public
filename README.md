# Deep Powder Crystallography
Source code to accompany our upcoming paper, *Towards End-to-End Electron Density Field Generation for Powder Crystallography: A Deep Learning Approach* by Gabe Guo, Judah Goldfeder, Ling Lan, Aniv Ray, Albert Hanming Yang, Boyuan Chen, Simon JL Billinge, Hod Lipson.

## Project Objective
Given x-ray diffraction pattern and partial chemical composition information, reconstruct 3D electron density function. AKA crystallography.

# Reproducibility

**Important:** Will need to change all datapaths starting in `/home/gabeguo/` to the corresponding filepaths on your system.

## Environment
Python 3.10.10
```
pip install -r requirements.txt
```

## Dataset

### Source
Data can be downloaded from the [Materials Project](https://next-gen.materialsproject.org/materials).

 Also need to change API_KEY in `download_charge_densities.py`.
```
cd download_data
bash download_data.sh
```

### Obtain Stable Crystals

```
cd download_data
bash record_stable_chemical_info.sh
```

### Data Split to Train, Val, Test

```
cd download_data
bash run_split_dataset.sh
```

### Get Formula, Crystal System, and Spacegroup Info

```
cd download_data
bash record_chemical_aux_info.sh
```

### Remove Duplicate Materials
```
cd download_data
bash remove_duplicate_materials.sh
```

### Splits

For reproducibility, we supply the mpids of all the crystals used in training, validation, and testing in `data_split_reproducibility_info`. Our own attempts to replicate the study indicate that the Materials Project dataset adds and removes some materials over time -- thus, results may vary from what is listed in the paper.
See [data_split_reproducibility_info](data_split_reproducibility_info).

## Training Model

```
cd multiXRD_withFormula/new_experimental_scripts
CUDA_VISIBLE_DEVICES=x train_base_model.sh
CUDA_VISIBLE_DEVICES=x train_formula_ablation.sh
```

## Testing Model

```
cd multiXRD_withFormula/new_experimental_scripts
CUDA_VISIBLE_DEVICES=x test_base_model.sh
CUDA_VISIBLE_DEVICES=x test_formula_ablation.sh
```

