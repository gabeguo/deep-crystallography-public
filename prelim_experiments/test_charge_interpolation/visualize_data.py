import numpy as np
from pyrho.pgrid import PGrid
from pyrho.vis.scatter import get_scatter_plot

from pymatgen.io.vasp import Chgcar
from pyrho.charge_density import ChargeDensity

for item in [('val', 572), ('train', 2861), ('train', 576), ('val', 284), ('val', 5694), ('test', 1189), ('test', 113)]:
    print(item)
    charge_density = ChargeDensity.from_file(
        "/data/therealgabeguo/crystallography/charge_data_split/{}/CHGCAR_mp-{}.vasp".format(item[0], item[1])
    )
    print('\t', charge_density.grid_shape)
    plt = get_scatter_plot(charge_density.pgrids['total'].grid_data, lat_mat=charge_density.pgrids['total'].lattice, plotter='plotly')
    #plt.title('{}-{} original'.format(item[0], item[1]))
    plt.show()
    # print(charge_density.grid_shape)
    # print(charge_density.normalized_data['total'])

    new_charge_density = np.load('/data/therealgabeguo/crystallography/charge_data_npy/{}/CHGCAR_mp-{}.npy'.format(item[0], item[1]))
    plt = get_scatter_plot(new_charge_density, lat_mat=charge_density.pgrids['total'].lattice, plotter='plotly')
    #plt.title('{}-{} downsampled'.format(item[0], item[1]))
    plt.show()