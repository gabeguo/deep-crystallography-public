for system in "Triclinic" "Monoclinic" "Orthorhombic" "Tetragonal" "Trigonal" "Hexagonal" "Cubic"
do
    python3 download_charge_densities.py \
        --output_dst_charge /home/gabeguo/data/replicate_crystallography/STABLE_material_info \
        --output_dst_xrd dummy \
        --crystal_system "${system}" \
        --record_id_to_spacegroup \
        --record_id_to_lattice \
        --record_id_to_formula \
        --stability 1
done
