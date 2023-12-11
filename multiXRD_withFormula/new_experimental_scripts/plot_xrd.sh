cd ../
for system in "Trigonal" "Cubic"
do
    python3 new_visualize_predictions.py \
        --num_x 2 \
        --num_y 2 \
        --num_z 2 \
        --model_path /home/gabeguo/data/crystallography_paper_version/new_results/models/base_${system}.pt \
        --charge_data_dir /home/gabeguo/data/crystallography_paper_version/charge_data_npy/test \
        --xrd_data_dir /home/gabeguo/data/crystallography_paper_version/xrd_data_tensor__moka_crka/test \
        --mp_id_to_formula /home/gabeguo/data/crystallography_paper_version/crystal_systems_all/${system}_formulas_NO_DUPLICATES.json \
        --mp_id_to_lattice /home/gabeguo/data/crystallography_paper_version/crystal_systems_all/${system}_lattice_vectors.json \
        --mp_id_to_spacegroup /home/gabeguo/data/crystallography_paper_version/crystal_systems_all/${system}_space_groups_NO_DUPLICATES.json \
        --max_num_crystals 500 \
        --num_channels 1 \
        --num_conv_blocks 4 \
        --num_formula_blocks 3 \
        --num_lattice_blocks 0 \
        --num_spacegroup_blocks 0 \
        --num_regressor_blocks 4 \
        --num_freq 128 \
        --dropout_prob 0 \
        --test_all_rotations \
        --num_trials 0 \
        --results_folder /home/gabeguo/data/crystallography_paper_version/new_results/test_results/xrds_${system} \
        --plot_only_xrd
done