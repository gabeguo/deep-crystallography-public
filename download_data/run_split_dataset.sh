for system in "cubic" "trigonal"
do
    python split_dataset.py \
        --charge_data_src /home/gabeguo/data/replicate_crystallography/${system}/charges \
        --xrd_data_src /home/gabeguo/data/replicate_crystallography/${system}/xrds \
        --charge_data_dst /home/gabeguo/data/replicate_crystallography/${system}_split/charges \
        --xrd_data_dst /home/gabeguo/data/replicate_crystallography/${system}_split/xrds \
        --stable_elems /home/gabeguo/data/replicate_crystallography/STABLE_material_info/${system^}_formulas.json \
        --train_percent 80 \
        --val_percent 10
done
