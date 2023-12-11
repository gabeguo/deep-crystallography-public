# Download trigonal
python3 download_charge_densities.py \
    --output_dst_charge /home/gabeguo/data/replicate_crystallography/trigonal/charges \
    --output_dst_xrd /home/gabeguo/data/replicate_crystallography/trigonal/xrds \
    --sample_freq 50 \
    --min_theta 0 \
    --max_theta 180 \
    --wave_sources 'MoKa' \
    --crystal_system 'Trigonal'
# Download cubic
python3 download_charge_densities.py \
    --output_dst_charge /home/gabeguo/data/replicate_crystallography/cubic/charges \
    --output_dst_xrd /home/gabeguo/data/replicate_crystallography/cubic/xrds \
    --sample_freq 50 \
    --min_theta 0 \
    --max_theta 180 \
    --wave_sources 'MoKa' \
    --crystal_system 'Cubic'