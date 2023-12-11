for crystal_system in "Cubic" "Trigonal"
do
    python remove_duplicates.py \
        --old_mp_id_to_formula /home/gabeguo/data/replicate_crystallography/material_info/${crystal_system}_formulas.json \
        --old_mp_id_to_spacegroup /home/gabeguo/data/replicate_crystallography/material_info/${crystal_system}_space_groups.json \
        --new_mp_id_to_formula /home/gabeguo/data/replicate_crystallography/material_info/${crystal_system}_formulas_NO_DUPLICATES.json \
        --new_mp_id_to_spacegroup /home/gabeguo/data/replicate_crystallography/material_info/${crystal_system}_space_groups_NO_DUPLICATES.json
done