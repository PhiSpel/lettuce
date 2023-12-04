import lettuce as lt
import torch
import numpy as np
import pandas as pd
from scipy import interpolate
from matplotlib import pyplot as plt
from liftdragcoefficient import FullwayBounceBackBoundary


class Garden(lt.Obstacle):
    def __init__(self, shape, **args):
        self.args = args
        # LETTUCE PARAMETERS #
        self.lattice = self.get_lattice()
        super(Garden, self).__init__(shape, reynolds_number=args["Re"], mach_number=args["Ma"],
                                   lattice=self.lattice, domain_length_x=1,
                                   char_length=1, char_velocity=1)
        self.mask = self.mask_flow(shape, **self.args)

    def get_lattice(self):
        if self.args["no_cuda"] == 1:
            print("Not using CUDA, but CPU.")
            return lt.Lattice(lt.D2Q9, torch.device("cpu"), use_native=False)
        else:
            print("Using CUDA.")
            return lt.Lattice(lt.D2Q9, torch.device("cuda"), use_native=False)

    def mask_flow(self, shape=(10000, 5000), debug=False, show=False, **args):
        Re = args['Re']
        Ma = args['Ma']
        # read data
        nx, ny = shape
        if debug:
            csv_data_np = np.genfromtxt('data/out_rough.csv', delimiter=', ', dtype=float)
            fig, ax = plt.subplots()
            ax.scatter(csv_data_np[:, 0], csv_data_np[:, 1])
            plt.show()
        csv_data = pd.read_csv('data/out_rough.csv', names=['x', 'y'], dtype=float)
        if debug:
            fig2, ax2 = plt.subplots()
            ax2.scatter(csv_data['x'], csv_data['y'])
            plt.show()
        # csv_data = np.reshape(csv_data, (int(len(csv_data) / 2), 2)).transpose()

        x_data = np.array(csv_data.sort_values('x')['x'])
        y_data = -np.array(csv_data.sort_values('x')['y'])

        # mapping data to the grid
        x_interp = np.linspace(x_data.min(), x_data.max(), nx)  # [0 ... 5.05]
        y_interp = interpolate.interp1d(x_data, y_data)(x_interp)  # .interp1d object

        # setting y data in a 2D grid to compare with flow.grid[1]
        y_mapped = np.array([y_interp]).transpose()

        # scaling y-data to bind with bottom and fix ratio while scaling x
        y_mapped -= y_data.min()
        self.domain_length_x = x_data.max() - x_data.min()
        dx = self.domain_length_x / nx
        dy = dx
        domain_length_y = dy * ny
        y_mapped *= 0.2 * domain_length_y / (y_data.max() - y_data.min())

        x, y = self.grid
        mask = (y < y_mapped)
        if debug or show:
            fig3, ax3 = plt.subplots()
            ax3.imshow(mask.transpose(), origin="lower", cmap='gray_r')
            plt.show()
        return mask

    @property
    def boundaries(self):
        obstacle = FullwayBounceBackBoundary(self.mask, self.units.lattice)
        # obstacle = lt.BounceBackBoundary(self.mask, self.units.lattice)
        return [
            lt.EquilibriumBoundaryPU(  # inlet
                np.abs(self.grid[0]) < 1e-6, self.units.lattice, self.units,
                self.units.characteristic_velocity_pu * self._unit_vector()
            ),
            lt.EquilibriumBoundaryPU(  # top
                np.abs(self.grid[1]) >= (self.grid[1].max() - 1e-6), self.units.lattice, self.units,
                self.units.characteristic_velocity_pu * self._unit_vector()
            ),
            lt.EquilibriumOutletP(self.units.lattice, self._unit_vector().tolist()),
            obstacle
        ]