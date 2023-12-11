# Usage: CUDA_VISIBLE_DEVICES=x bash generate_base_models.sh

cd ../

for CRYSTAL_SYSTEM in Cubic Trigonal
do
    OUTPUT_FOLDER=/data/therealgabeguo/crystallography/paper_results/${CRYSTAL_SYSTEM}_base
    MODEL_PATH=$OUTPUT_FOLDER/${CRYSTAL_SYSTEM}_base_model.pt
    RESULTS_PATH=$OUTPUT_FOLDER/${CRYSTAL_SYSTEM}_base_visualizations

    python3 train_loop.py \
        --num_x 5 \
        --num_y 5 \
        --num_z 5 \
        --num_x_val 25 \
        --num_y_val 25 \
        --num_z_val 25 \
        --patience 840 \
        --log_interval 30 \
        --val_log_interval 10 \
        --batch_size 128 \
        --test_batch_size 8192 \
        --adam \
        --warm_restart \
        --lr 5e-4 \
        --min_lr 1e-7 \
        --momentum 0.9 \
        --weight_decay 0 \
        --gamma 2 \
        --lr_step 70 \
        --discrim_adam \
        --discrim_warm_restart \
        --discrim_lr 1e-4 \
        --discrim_min_lr 1e-8 \
        --discrim_momentum 0.9 \
        --discrim_weight_decay 0 \
        --discrim_gamma 2 \
        --discrim_lr_step 100 \
        --lambda_adv 0 \
        --lambda_sim 0 \
        --latent_sigma 1e6 \
        --augmentation \
        --xrd_noise 1e-3 \
        --xrd_shift 3 \
        --formula_noise 1e-3 \
        --max_num_crystals 50000 \
        --max_num_val_crystals 250 \
        --charge_data_dir /data/therealgabeguo/crystallography/charge_data_npy \
        --xrd_data_dir /data/therealgabeguo/crystallography/xrd_data_tensor__moka_crka \
        --alt_train_charge_folder /data/therealgabeguo/crystallography/unstable_charge_densities \
        --alt_train_xrd_folder /data/therealgabeguo/crystallography/unstable_xrd_tensor__moka_crka \
        --model_path $MODEL_PATH \
        --epochs 840 \
        --num_channels 2 \
        --num_conv_blocks 4 \
        --num_regressor_blocks 4 \
        --num_formula_blocks 3 \
        --num_lattice_blocks 0 \
        --num_spacegroup_blocks 0 \
        --num_freq 256 \
        --freq_sigma 5 \
        --dropout_prob 0 \
        --results_folder /data/therealgabeguo/crystallography/paper_results/run_logs \
        --mp_id_to_formula /data/therealgabeguo/crystallography/crystal_systems_all/${CRYSTAL_SYSTEM}_formulas.json \
        --mp_id_to_lattice /data/therealgabeguo/crystallography/crystal_systems_all/${CRYSTAL_SYSTEM}_lattice_vectors.json \
        --mp_id_to_spacegroup /data/therealgabeguo/crystallography/crystal_systems_all/${CRYSTAL_SYSTEM}_space_groups.json \
        --wandb_project crystallography_paper_results

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