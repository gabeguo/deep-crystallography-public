# Usage: CUDA_VISIBLE_DEVICES=x bash run_basic_cubic_tests.sh
# This is redundant, but I created it because for some reason, the original combined train-test script froze

cd ../

CRYSTAL_SYSTEM=Cubic

for MODEL_VERSION in base l2Loss
do

    OUTPUT_FOLDER=/data/therealgabeguo/crystallography/paper_results/${CRYSTAL_SYSTEM}_${MODEL_VERSION}
    MODEL_PATH=$OUTPUT_FOLDER/${CRYSTAL_SYSTEM}_${MODEL_VERSION}_model.pt
    RESULTS_PATH=$OUTPUT_FOLDER/${CRYSTAL_SYSTEM}_${MODEL_VERSION}_visualizations

    python3 new_visualize_predictions.py \
        --num_x 50 \
        --num_y 50 \
        --num_z 50 \
        --model_path $MODEL_PATH \
        --charge_data_dir /data/therealgabeguo/crystallography/charge_data_npy/test \
        --xrd_data_dir /data/therealgabeguo/crystallography/xrd_data_tensor__moka_crka/test \
        --mp_id_to_formula /data/therealgabeguo/crystallography/crystal_systems_all/${CRYSTAL_SYSTEM}_formulas.json \
        --mp_id_to_lattice /data/therealgabeguo/crystallography/crystal_systems_all/${CRYSTAL_SYSTEM}_lattice_vectors.json \
        --mp_id_to_spacegroup /data/therealgabeguo/crystallography/crystal_systems_all/${CRYSTAL_SYSTEM}_space_groups.json \
        --max_num_crystals 500 \
        --num_channels 2 \
        --num_conv_blocks 4 \
        --num_formula_blocks 3 \
        --num_lattice_blocks 0 \
        --num_spacegroup_blocks 0 \
        --num_regressor_blocks 4 \
        --num_freq 256 \
        --dropout_prob 0 \
        --test_all_rotations \
        --results_folder $RESULTS_PATH

done