import os

import lettuce as lt
import torch
import numpy as np
import pandas as pd
from scipy import interpolate
from matplotlib import pyplot as plt
from liftdragcoefficient import FullwayBounceBackBoundary


class Garden(lt.Obstacle):
    def __init__(self, **args):
        self.y0 = None
        self.y1 = None
        self.args = args
        # LETTUCE PARAMETERS #
        self.lattice = self.get_lattice()
        super(Garden, self).__init__((args["nx"], args["ny"]), reynolds_number=args["Re"], mach_number=args["Ma"],
                                   lattice=self.lattice, domain_length_x=1,
                                   char_length=1, char_velocity=1)
        self.mask = self.mask_flow(**self.args)

    def get_lattice(self):
        if self.args["no_cuda"] == 1:
            print("Not using CUDA, but CPU.")
            return lt.Lattice(lt.D2Q9, torch.device("cpu"), use_native=False)
        else:
            print("Using CUDA.")
            return lt.Lattice(lt.D2Q9, torch.device("cuda"), use_native=False)

    def mask_flow(self, **args):
        # read data
        nx, ny = (args['nx'], args['ny'])
        csv_data = pd.read_csv(os.getcwd() + '/data/out.csv', names=['x', 'y'], dtype=float)
        if args["debug"]:
            fig2, ax2 = plt.subplots()
            ax2.scatter(csv_data['x'], csv_data['y'])
            ax2.axis('equal')
            plt.show()
        # csv_data = np.reshape(csv_data, (int(len(csv_data) / 2), 2)).transpose()

        x_data = np.array(csv_data.sort_values('x')['x'])
        y_data = np.array(csv_data.sort_values('x')['y'])

        # mapping data to the grid
        x_interp = np.linspace(x_data.min(), x_data.max(), nx)  # [xmin ... xmax]
        y_interp = interpolate.interp1d(x_data, y_data)(x_interp)  # .interp1d object

        # setting y data in a 2D grid to compare with flow.grid[1]
        y_mapped = np.array([y_interp]).transpose() - y_data.min()
        dx = 1 / (x_interp.max() - x_interp.min())
        dy = dx
        y_mapped *= dy
        y_mapped += 1e-6

        # scaling y-data to bind with bottom and fix ratio while scaling x
        # y_mapped -= y_data.min()
        # domain_length_x = x_data.max() - x_data.min()
        # dx = domain_length_x / nx
        # dy = dx
        # height = y_data.max() - y_data.min()
        # y_mapped /= dy

        x, y = self.grid
        mask = (y < y_mapped)
        self.y0 = y_mapped[1]
        self.y1 = y_mapped[-1]
        if args["debug"]:
            fig4, ax4 = plt.subplots()
            ax4.scatter(x_interp, y_mapped)
            ax4.axis('equal')
            plt.show()
            fig5, ax5 = plt.subplots()
            ax5.scatter(x, y)
            ax5.axis('equal')
            plt.show()
        if args["debug"] or args["show"]:
            fig3, ax3 = plt.subplots()
            ax3.imshow(mask.transpose(), origin="lower", cmap='gray_r')
            plt.show()
        return mask

    @property
    def boundaries(self):
        obstacle = FullwayBounceBackBoundary(self.mask, self.units.lattice)
        x, y = self.grid
        # obstacle = lt.BounceBackBoundary(self.mask, self.units.lattice)
        return [
            lt.EquilibriumBoundaryPU(  # inlet
                (np.abs(x) < 1e-6) * (y > self.y0),
                self.units.lattice, self.units, self.units.characteristic_velocity_pu * self._unit_vector()
            ),
            lt.EquilibriumBoundaryPU(  # top
                np.abs(y) >= (y.max() - 1e-6), self.units.lattice, self.units,
                self.units.characteristic_velocity_pu * self._unit_vector()
            ),
            lt.EquilibriumBoundaryPU(  # outlet
                (np.abs(x) >= (x.max() - 1e-6)) * (y > self.y1),
                self.units.lattice, self.units, self.units.characteristic_velocity_pu * self._unit_vector()
            ),
            obstacle
        ]
