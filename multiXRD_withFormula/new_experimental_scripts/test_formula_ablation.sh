cd ../
for system in "Trigonal" "Cubic"
do
    # formula without mass ratios (all elements, one dropped element)
    for num_exclude in 0 1
    do
        python3 new_visualize_predictions.py \
            --num_x 50 \
            --num_y 50 \
            --num_z 50 \
            --model_path /home/gabeguo/data/crystallography_paper_version/new_results/models/noElementalRatios_${num_exclude}Dropped_${system}.pt \
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
            --num_trials 3 \
            --ignore_elemental_ratios \
            --unstandardized_formula \
            --num_excluded_elems $num_exclude \
            --results_folder /home/gabeguo/data/crystallography_paper_version/new_results/test_results/noElementalRatios_${num_exclude}Dropped_${system}
    done
    python3 new_visualize_predictions.py \
        --num_x 50 \
        --num_y 50 \
        --num_z 50 \
        --model_path /home/gabeguo/data/crystallography_paper_version/new_results/models/noFormula_${system}.pt \
        --charge_data_dir /home/gabeguo/data/crystallography_paper_version/charge_data_npy/test \
        --xrd_data_dir /home/gabeguo/data/crystallography_paper_version/xrd_data_tensor__moka_crka/test \
        --mp_id_to_formula /home/gabeguo/data/crystallography_paper_version/crystal_systems_all/${system}_formulas_NO_DUPLICATES.json \
        --mp_id_to_lattice /home/gabeguo/data/crystallography_paper_version/crystal_systems_all/${system}_lattice_vectors.json \
        --mp_id_to_spacegroup /home/gabeguo/data/crystallography_paper_version/crystal_systems_all/${system}_space_groups_NO_DUPLICATES.json \
        --max_num_crystals 500 \
        --num_channels 1 \
        --num_conv_blocks 4 \
        --num_formula_blocks 0 \
        --num_lattice_blocks 0 \
        --num_spacegroup_blocks 0 \
        --num_regressor_blocks 4 \
        --num_freq 128 \
        --dropout_prob 0 \
        --test_all_rotations \
        --num_trials 3 \
        --results_folder /home/gabeguo/data/crystallography_paper_version/new_results/test_results/noFormula_${system}
    python3 new_visualize_predictions.py \
        --num_x 50 \
        --num_y 50 \
        --num_z 50 \
        --model_path /home/gabeguo/data/crystallography_paper_version/new_results/models/formulaAblationBaseline_${system}.pt \
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
        --num_trials 3 \
        --results_folder /home/gabeguo/data/crystallography_paper_version/new_results/test_results/formulaAblationBaseline_${system}
done