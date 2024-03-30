cd ../
for system in "Tetragonal" "Orthorhombic" "Hexagonal" "Monoclinic" "Triclinic"
do
    for model in "Cubic" "Trigonal"
    do
        python3 new_visualize_predictions.py \
            --num_x 50 \
            --num_y 50 \
            --num_z 50 \
            --model_path /home/gabeguo/old_crystallography_results/new_results/models/base_${model}.pt \
            --charge_data_dir /home/gabeguo/old_crystallography_results/charge_data_npy/test \
            --xrd_data_dir /home/gabeguo/old_crystallography_results/xrd_data_tensor__moka_crka/test \
            --mp_id_to_formula /home/gabeguo/old_crystallography_results/crystal_systems_all/${system}_formulas.json \
            --mp_id_to_lattice /home/gabeguo/old_crystallography_results/crystal_systems_all/${system}_lattice_vectors.json \
            --mp_id_to_spacegroup /home/gabeguo/old_crystallography_results/crystal_systems_all/${system}_space_groups.json \
            --max_num_crystals 10 \
            --num_channels 1 \
            --num_conv_blocks 4 \
            --num_formula_blocks 3 \
            --num_lattice_blocks 0 \
            --num_spacegroup_blocks 0 \
            --num_regressor_blocks 4 \
            --num_freq 128 \
            --dropout_prob 0 \
            --test_all_rotations \
            --num_trials 5 \
            --results_folder /home/gabeguo/old_crystallography_results/new_results/test_results/PRELIM_EXPLORATION_${system}_${model}Model
    done
done