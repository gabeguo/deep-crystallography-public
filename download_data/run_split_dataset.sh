for system in "cubic" "trigonal"
do
    # move everything to train folder first (ensures reproducibility)
    python split_dataset.py \
        --charge_data_src /home/gabeguo/data/replicate_crystallography/${system}_split/charges \
        --xrd_data_src /home/gabeguo/data/replicate_crystallography/${system}_split/xrds \
        --charge_data_dst /home/gabeguo/data/replicate_crystallography/${system}_split/charges \
        --xrd_data_dst /home/gabeguo/data/replicate_crystallography/${system}_split/xrds \
        --stable_elems /home/gabeguo/data/replicate_crystallography/STABLE_material_info/${system^}_formulas.json \
        --train_percent 100 \
        --val_percent 0 \
        --seed 0
    # now do the split
    python split_dataset.py \
        --charge_data_src /home/gabeguo/data/replicate_crystallography/${system}_split/charges \
        --xrd_data_src /home/gabeguo/data/replicate_crystallography/${system}_split/xrds \
        --charge_data_dst /home/gabeguo/data/replicate_crystallography/${system}_split/charges \
        --xrd_data_dst /home/gabeguo/data/replicate_crystallography/${system}_split/xrds \
        --stable_elems /home/gabeguo/data/replicate_crystallography/STABLE_material_info/${system^}_formulas.json \
        --train_percent 80 \
        --val_percent 10 \
        --seed 0
done
