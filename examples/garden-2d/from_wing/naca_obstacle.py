import os
from lettuce import Obstacle
import lettuce as lt
import numpy as np
from scipy import interpolate
import torch
from pyevtk.hl import imageToVTK
from liftdragcoefficient import FullwayBounceBackBoundary

class Naca(Obstacle):
    def __init__(self, **args):
        self.filename_base = args["filename_base"]
        self.t_pre = args["wing_length"] / args["vchar"] * args["n_stream_pre"]
        self.Re_pre = args["Re_pre"]
        self.Ma_pre = 0.1
        self.shape = args["shape"]
        self.wing_name = args["name"]
        self.liftdrag = args["liftdrag"]
        self.args = args

        self.pre_p, self.pre_u = self.calculate_pre()

        # LETTUCE PARAMETERS #
        self.lattice = self.get_lattice()
        super(Naca, self).__init__(self.shape, reynolds_number=args["Re"], mach_number=args["Ma"],
                                   lattice=self.lattice, domain_length_x=args["domain_length_x"],
                                   char_length=args["wing_length"], char_velocity=args["vchar"])
        self.mask = self.mask_from_csv(self.wing_name, self.grid, **self.args)
        # point_dict["p"] = self.lattice.convert_to_numpy(p[0, ...])
        # for d in range(self.lattice.D):
        #     self.point_dict[f"u{'xyz'[d]}"] = self.lattice.convert_to_numpy(u[d, ...])

    def get_lattice(self):
        if self.args["no_cuda"] == 1:
            print("Not using CUDA, but CPU.")
            return lt.Lattice(lt.D3Q27, torch.device("cpu"), use_native=False)
        else:
            print("Using CUDA.")
            return lt.Lattice(lt.D3Q27, torch.device("cuda"), use_native=False)

    def calculate_pre(self):
        lattice = self.get_lattice()
        pre_flow = Obstacle(self.shape, reynolds_number=self.Re_pre, mach_number=self.Ma_pre,
                            lattice=lattice, domain_length_x=self.args["domain_length_x"])
        n_pre = int(pre_flow.units.convert_time_to_lu(self.t_pre))
        # run a bit with low Re
        print('Doing', n_pre, 'steps with Re =', self.Re_pre, 'before actual run. ', end="")
        pre_flow.mask = self.mask_from_csv(self.wing_name, pre_flow.grid, **self.args)
        point_dict = dict()
        point_dict["mask"] = pre_flow.mask.astype(int)
        imageToVTK(
            path=f"{self.filename_base}_mask",
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 1.0),
            cellData=None,
            pointData=point_dict,
            fieldData=None,
        )
        ndim = len(self.shape)
        if ndim == 2:
            collision = lt.KBCCollision2D(lattice, pre_flow.units.relaxation_parameter_lu)
        elif ndim == 3:
            collision = lt.KBCCollision3D(lattice, pre_flow.units.relaxation_parameter_lu)
        else:
            collision = lt.BGKCollision(lattice, pre_flow.units.relaxation_parameter_lu)
        simulation = lt.Simulation(pre_flow, lattice, collision, lt.StandardStreaming(lattice))
        print("Pre-time in pu: {:.4f}".format(pre_flow.units.convert_time_to_pu(n_pre)), "s")
        simulation.initialize_f_neq()
        simulation.reporters.append(lt.VTKReporter(lattice, pre_flow, interval=self.args["nreport_pre"],
                                                   filename_base=self.filename_base + 'pre'))
        simulation.step(n_pre)
        p = simulation.flow.units.convert_density_lu_to_pressure_pu(simulation.lattice.rho(simulation.f))
        u = simulation.flow.units.convert_velocity_to_pu(lattice.u(simulation.f))
        if ndim == 3:
            nx, ny, nz = self.shape
            u1 = (np.random.rand(nx, ny, nz) - 0.5) * 2
            u2 = (np.random.rand(nx, ny, nz) - 0.5) * 2
            u3 = (np.random.rand(nx, ny, nz) - 0.5) * 2
            u = u.cpu() + np.array([u1, u2, u3]) * np.invert(pre_flow.mask)
        u_test = lattice.u(simulation.f).cpu().mean().item()
        if not u_test == u_test:
            print("Pre-run crashed!")
        return p, u

    def initial_solution(self, x):
        return self.pre_p, self.pre_u

    def mask_from_csv(self, wing_name, grid, **args):
        x, y, *z = grid
        mask_shape = np.shape(x)
        n_wing_height = int(mask_shape[1] // 2)  # wing sits at middle of domain length

        # read wing data from http://airfoiltools.com/plotter/index
        surface_data = np.genfromtxt(os.path.dirname(os.path.realpath(__file__)) + '/' + wing_name + '.csv',
                                     delimiter=",")[9:, :]
        surface_data = surface_data[:np.min(np.where(np.isnan(surface_data)[:, 1])), :]
        zero_row = np.where(surface_data[:, 0] == 0)[0][0]
        x_data_top, y_data_top = surface_data[:zero_row, :].transpose()
        x_data_bottom, y_data_bottom = surface_data[zero_row:, :].transpose()

        # calculate scaling factor
        available_length_n = args["n_wing_tail"] - args["n_wing_nose"]
        actual_wing_length_x = max(max(x_data_top), max(x_data_bottom))
        scaling_factor = args["wing_length"] / actual_wing_length_x

        # scale wing data to fit domain restrictions
        x_data_top *= scaling_factor
        x_data_bottom *= scaling_factor
        y_data_top *= scaling_factor
        y_data_bottom *= scaling_factor

        # rescale if wing is too large vertically (should be less than half of the domain
        y_wing_height = max(x_data_top) - min(y_data_bottom)
        domain_height = np.max(y)
        if y_wing_height >= 0.5 * domain_height:
            scaling_factor = domain_height / y_wing_height
            x_data_top *= scaling_factor
            x_data_bottom *= scaling_factor
            y_data_top *= scaling_factor
            y_data_bottom *= scaling_factor

        # mapping data to the grid
        x_data_interp = np.linspace(0, args["wing_length"], available_length_n)  # [0 ... 5.05]
        y_data_top_interp = interpolate.interp1d(x_data_top, y_data_top, fill_value="extrapolate")(
            x_data_interp)
        y_data_bottom_interp = interpolate.interp1d(x_data_bottom, y_data_bottom, fill_value="extrapolate")(
            x_data_interp)

        # shifting the wing up by half the grid
        if y.ndim == 2:
            y_wing_height = y[0, n_wing_height]
            y_data_top_interp += y_wing_height
            y_data_bottom_interp += y_wing_height

            # setting y data in a 2D grid to compare with flow.grid[1]
            y_data_top_mapped = np.zeros(mask_shape)
            y_data_top_mapped[args["n_wing_nose"]:args["n_wing_tail"], :] = np.array([y_data_top_interp]).transpose()
            y_data_bottom_mapped = np.zeros(mask_shape)
            y_data_bottom_mapped[args["n_wing_nose"]:args["n_wing_tail"], :] = np.array(
                [y_data_bottom_interp]).transpose()
        elif y.ndim == 3:
            y_wing_height = y[0, n_wing_height, 0]
            y_data_top_interp += y_wing_height
            y_data_bottom_interp += y_wing_height

            # setting y data in a 2D grid to compare with flow.grid[1]
            y_data_top_mapped = np.zeros(mask_shape)
            y_data_bottom_mapped = np.zeros(mask_shape)
            for iz in range(mask_shape[2]):
                y_data_top_mapped[args["n_wing_nose"]:args["n_wing_tail"], :, iz] += np.array(
                    [y_data_top_interp]).transpose()
                y_data_bottom_mapped[args["n_wing_nose"]:args["n_wing_tail"], :, iz] += np.array(
                    [y_data_bottom_interp]).transpose()
        else:
            assert ValueError("Wrong dimensions, must be 2 or 3.")
            return

        # creating mask
        bool_mask = (y < y_data_top_mapped) & (y > y_data_bottom_mapped)
        return bool_mask

    @property
    def boundaries(self):
        if self.liftdrag:
            obstacle = FullwayBounceBackBoundary(self.mask, self.units.lattice)
        else:
            obstacle = lt.BounceBackBoundary(self.mask, self.units.lattice)
        return [
            lt.EquilibriumBoundaryPU(
                np.abs(self.grid[0]) < 1e-6, self.units.lattice, self.units,
                self.units.characteristic_velocity_pu * self._unit_vector()
            ),
            lt.EquilibriumOutletP(self.units.lattice, self._unit_vector().tolist()),
            obstacle
        ]
