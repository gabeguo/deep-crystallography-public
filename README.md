# Deep Powder Crystallography
Source code to accompany our upcoming paper, *Towards End-to-End Structure Determination from X-Ray Diffraction Data Using Deep Learning* by Gabe Guo, Judah Goldfeder, Ling Lan, Aniv Ray, Albert Hanming Yang, Boyuan Chen, Simon JL Billinge, Hod Lipson.

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
You may edit the `seed` argument if you like, in `run_split_dataset.sh`.
```
cd download_data
bash run_split_dataset.sh
```

*Note:* Our own attempts to replicate the study indicate that the Materials Project dataset adds and removes some materials over time &mdash; thus, results may vary slightly from what is listed in the paper. For reproducibility, we supply the mpids of all the crystals used in training, validation, and testing in [data_split_reproducibility_info](data_split_reproducibility_info). 

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

## Training Model
```
cd multiXRD_withFormula/new_experimental_scripts
CUDA_VISIBLE_DEVICES=x bash train_base_model.sh
CUDA_VISIBLE_DEVICES=x bash train_formula_ablation.sh
```

## Testing Model
```
cd multiXRD_withFormula/new_experimental_scripts
CUDA_VISIBLE_DEVICES=x bash test_base_model.sh
CUDA_VISIBLE_DEVICES=x bash test_formula_ablation.sh
```

