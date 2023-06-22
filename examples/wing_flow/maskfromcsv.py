import os

from lettuce import Obstacle
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


class Naca(Obstacle):
    def __init__(self, wing_name, shape, lattice, **args):
        args["domain_length_x"]: float
        args["x_wing_nose"]: float
        args["x_wing_tail"]: float
        args["wing_length"]: float
        args["chord_length"]: float
        args["vchar"]: float
        args["Ma"]: float

        super(Naca, self).__init__(shape, reynolds_number=args["Re"], mach_number=args["Ma"],lattice=lattice,
                                   domain_length_x=args["domain_length_x"], char_length=args["chord_length"],
                                   char_velocity=args["vchar"])
        x, y = self.grid
        self.mask = self.mask_from_csv(x, y, wing_name, **args)

    def mask_from_csv(self, x, y, wing_name, **args):
        mask_shape = np.shape(x)
        nx1, ny1 = mask_shape
        dx = args["domain_length_x"] / nx1  ## i.e. resolution
        n_wing_nose = int(args["x_wing_nose"] // dx)  ## first grid point with wing
        n_wing_tail = int(args["x_wing_tail"] // dx)  ## first grid point with wing
        # n_wing_nose = int(nx1//5)    # wing starts after 1/5 of domain length
        # n_wing_tail = int(nx1*3//5)  # wing goes until 3/5 of domain length
        n_wing_height = int(ny1 // 2)  # wing sits at middle of domain length

        # read wing data from http://airfoiltools.com/plotter/index
        surface_data = np.genfromtxt(os.getcwd() + '/' + wing_name + '.csv', delimiter=",")[9:, :]
        surface_data = surface_data[:np.min(np.where(np.isnan(surface_data)[:, 1])), :]
        zero_row = np.where(surface_data[:, 0] == 0)[0][0]
        x_data_top, y_data_top = surface_data[:zero_row, :].transpose()
        x_data_bottom, y_data_bottom = surface_data[zero_row:, :].transpose()

        # calculate scaling factor
        # x_wing_nose = x[n_wing_nose,0]
        # x_wing_tail = x[n_wing_tail,0]
        # available_length_x = x_wing_tail - x_wing_nose
        available_length_n = n_wing_tail - n_wing_nose
        actual_wing_length_x = max(max(x_data_top), max(x_data_bottom))
        scaling_factor = args["wing_length"] / actual_wing_length_x

        # scale wing data to fit domain restrictions
        x_data_top *= scaling_factor
        x_data_bottom *= scaling_factor
        y_data_top *= scaling_factor
        y_data_bottom *= scaling_factor

        # mapping data to the grid
        x_data_interp = np.linspace(0, args["wing_length"], available_length_n)  # [0 ... 5.05]
        y_data_top_interp = interpolate.interp1d(x_data_top, y_data_top, fill_value="extrapolate")(
            x_data_interp)  # .interp1d object
        y_data_bottom_interp = interpolate.interp1d(x_data_bottom, y_data_bottom, fill_value="extrapolate")(
            x_data_interp)  # .interp1d object

        # shifting the wing up by half the grid
        y_wing_height = y[0, n_wing_height]
        y_data_top_interp += y_wing_height
        y_data_bottom_interp += y_wing_height

        # setting y data in a 2D grid to compare with flow.grid[1]
        y_data_top_mapped = np.zeros(mask_shape)
        y_data_top_mapped[n_wing_nose:n_wing_tail, :] = np.array([y_data_top_interp]).transpose()
        y_data_bottom_mapped = np.zeros(mask_shape)
        y_data_bottom_mapped[n_wing_nose:n_wing_tail, :] = np.array([y_data_bottom_interp]).transpose()

        # creating mask
        bool_mask = (y < y_data_top_mapped) & (y > y_data_bottom_mapped)
        return bool_mask

