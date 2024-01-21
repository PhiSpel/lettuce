import numpy as np
from lettuce.unit import UnitConversion
from lettuce.util import append_axes
from lettuce.boundary import EquilibriumBoundaryPU, \
    BounceBackBoundary, HalfwayBounceBackBoundary, FullwayBounceBackBoundary, EquilibriumOutletP, AntiBounceBackOutlet, \
    InterpolatedBounceBackBoundary, InterpolatedBounceBackBoundary_compact_v1, InterpolatedBounceBackBoundary_compact_v2, \
    SlipBoundary, FullwayBounceBackBoundary_compact, HalfwayBounceBackBoundary_compact_v1, HalfwayBounceBackBoundary_compact_v2, \
    HalfwayBounceBackBoundary_compact_v3
import torch


class ObstacleCylinder:
    """
        add description here: unified version of 2D and 3D cylinder flow
        refined version of flow/obstacle.py, for cylinder-flow in 2D or 3D. The dimensions will be assumed from
        lattice.D

        Flow:
        - inflow (EquilibriumBoundaryPU) at x=0, outflow (EquilibriumOutletP) at x=xmax
        - further boundaries depend on parameters:
            - lateral (y direction): periodic, no-slip wall, slip wall
            - lateral (z direction): periodic (only if lattice.D==3)
        - obstacle: cylinder obstacle centered at (y_lu/2, y_LU/2), with radius, uniform symmetry in z-direction
            - obstacle mask has to be set externally?
        - boundary condition for obstacle can be chosen: hwbb, fwbb, ibb1
        - initial pertubation (trigger Von Kármán vortex street for Re>46) can be initialized in y and z direction
        - initial velocity can be 0, u_char or a parabolic profile (parabolic if lateral_walls = "bounceback")
        - inlet/inflow velocity can be uniform u_char or parabolic

        Parameters:
        ----------
        <to fill>
        ----------
    """
    def __init__(self, shape, reynolds_number, mach_number, lattice, char_length_pu, char_velocity_pu=1,
                 lateral_walls=None, bc_type='ibb1c2', perturb_init=False, u_init=0, x_offset=0, y_offset=0):
        # shape of the domain (2D or 3D):
        if len(shape) != lattice.D:
            raise ValueError(f"{lattice.D}-dimensional lattice requires {lattice.D}-dimensional `shape`")
        if len(shape) == 2:
            self.shape = (int(shape[0]), int(shape[1]))
        elif len(shape) == 3:
            self.shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        else:
            print("WARNING: shape is not 2- or 3-dimensional...(!)")
        #self.shape = shape

        self.char_length_pu = char_length_pu  # characteristic length

        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=char_length_pu,  # TODO: change this to char_length_lu
            characteristic_length_pu=char_length_pu,
            characteristic_velocity_pu=char_velocity_pu  # reminder: u_char_lu = Ma * cs_lu = Ma * 1/sqrt(3)
        )

        # flow and boundary settings
        self.perturb_init = perturb_init  # toggle: introduce asymmetry in initial solution to trigger v'Karman Vortex Street
        self.u_init = u_init  # toggle: initial solution velocity profile type
        self.lateral_walls = lateral_walls  # toggle: lateral walls to be bounce back (bounceback), slip wall (slip) or periodic (periodic)
        self.bc_type = bc_type  # toggle: bounce back algorithm: halfway (hwbb) or fullway (fwbb)

        x, y = self.grid

        # initialize masks (init with zeros)
        self.solid_mask = torch.zeros(self.shape, dtype=torch.bool)  # marks all solid nodes (obstacle, walls, ...)
        self.in_mask = torch.zeros(x.shape, dtype=torch.bool)  # marks all inlet nodes
        self.out_mask = torch.zeros(x.shape, dtype=torch.bool)  # marks all outlet nodes
        # self.wall_mask = torch.zeros_like(self.solid_mask)  # marks lateral (top+bottom) walls
        self._obstacle_mask = torch.zeros_like(self.solid_mask)  # marks all obstacle nodes (for fluid-solid-force_calc.)

        # cylinder geometry in LU (1-based indexing!)
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.radius = char_length_lu / 2
        self.y_pos = self.shape[1] / 2 + 0.5 + self.y_offset  # y_position of cylinder-center in 1-based indexing
        self.x_pos = self.y_pos + self.x_offset  # keep symmetry of cylinder in x and y direction

        xyz = tuple(torch.linspace(1, n, n) for n in self.shape)  # Tupel of index-lists (1-n (one-based!))
        if self.units.lattice.D == 2:
            x_lu, y_lu = torch.meshgrid(*xyz, indexing='ij')  # meshgrid of x-, y-index
        elif self.units.lattice.D == 3:
            x_lu, y_lu, z_lu = torch.meshgrid(*xyz, indexing='ij')  # meshgrid of x-, y- and z-index
        else:
            raise ValueError("WARNING: something went wrong in LU-gird-index generation, lattice.D must be 2 or 3!")

        condition = np.sqrt((x_lu - self.x_pos) ** 2 + (y_lu - self.y_pos) ** 2) < self.radius
        self.obstacle_mask[torch.where(condition)] = 1
        self.solid_mask[torch.where(condition)] = 1

        # indexing doesn't need z-Index for 3D, everything is broadcasted along z!
        if self.lateral_walls == 'garden':
            self.in_mask = (np.abs(x) < 1e-6) * (y > y_mapped[0])
            self.out_mask = (np.abs(x) >= (x.max() - 1e-6)) * (y > self.y_mapped[-1])
        else:  # if lateral_wals == 'periodic', no walls
            self.in_mask[0, :] = True  # inlet on the left (x=0)
            self.out_mask[-1, :] = True  # outlet on the right (x=xmax)

    def initial_solution(self, x: torch.Tensor):
        p = torch.zeros_like(x[0], dtype=torch.float)[None, ...]
        u_max_pu = self.units.characteristic_velocity_pu * self._unit_vector()
        u_max_pu = append_axes(u_max_pu, self.units.lattice.D)
        self.solid_mask[torch.where(self.obstacle_mask)] = 1  # This line is needed, because the obstacle_mask.setter does not define the solid_mask properly (see above) #OLD
        ### initial velocity field: "u_init"-parameter
        # 0: uniform u=0
        # 1: uniform u=1 or parabolic (depends on lateral_walls -> bounceback => parabolic; slip, periodic => uniform)
        u = ~self.solid_mask * u_max_pu
        if self.u_init == 0:
            u = u * 0  # uniform u=0
        else:
            if self.lateral_walls == 'bounceback':  # parabolic along y, uniform along x and z (similar to poiseuille-flow)
                ny = self.shape[1]  # number of gridpoints in y direction
                ux_factor = torch.zeros(ny)  # vector for one column (u(x=0))
                # multiply parabolic profile with every column of the velocity field:
                y_coordinates = torch.linspace(0, ny, ny)
                ux_factor[1:-1] = - y_coordinates[1:-1] * (y_coordinates[1:-1] - ny) * 1 / (ny / 2) ** 2
                if self.units.lattice.D == 2:
                    u = torch.einsum('k,ijk->ijk', ux_factor, u)
                elif self.units.lattice.D == 3:
                    u = torch.einsum('k,ijkl->ijkl', ux_factor, u)
            else:  # lateral_walls == periodic or slip
                # initiale velocity u_PU=1 on every fluid node
                u = (1 - self.solid_mask) * u_max_pu

        ### perturb initial velocity field-symmetry (in y and z) to trigger 'von Karman' vortex street
        if self.perturb_init:  # perturb initial solution in y
            # overlays a sine-wave on the second column of nodes x_lu=1 (index 1)
            ny = x[1].shape[1]
            if u.max() < 0.5 * self.units.characteristic_velocity_pu:
                # add perturbation for small velocities
                #OLD 2D: u[0][1] += np.sin(np.linspace(0, ny, ny) / ny * 2 * np.pi) * self.units.characteristic_velocity_pu * 1.0
                amplitude_y = np.sin(np.linspace(0, ny, ny) / ny * 2 * np.pi) * self.units.characteristic_velocity_pu * 0.1
                if self.units.lattice.D == 2:
                    u[0][1] += amplitude_y
                elif self.units.lattice.D == 3:
                    nz = x[2].shape[2]
                    plane_yz = np.ones_like(u[0, 1])  # plane of ones
                    u[0][1] = np.einsum('y,yz->yz', amplitude_y, plane_yz)  # plane of amplitude in y
                    amplitude_z = np.sin(np.linspace(0, nz, nz) / nz * 2 * np.pi) * self.units.characteristic_velocity_pu * 0.1  # amplitude in z
                   # print("amplitude y:", amplitude_y.shape)
                   # print("u[0][1]:", u[0][1].shape)
                   # print("amplitude z:", amplitude_z.shape)
                    # factor = 1 + np.sin(np.linspace(0, nz, nz) / nz * 2 * np.pi) * 0.3  # pertubation in z-direction
                    u[0][1] += np.einsum('z,yz->yz', amplitude_z, plane_yz)
            else:
                # multiply scaled down perturbation if velocity field is already near u_char
                #OLD 2D: u[0][1] *= 1 + np.sin(np.linspace(0, ny, ny) / ny * 2 * np.pi) * 0.3
                factor = 1 + np.sin(np.linspace(0, ny, ny) / ny * 2 * np.pi) * 0.1
                if self.units.lattice.D == 2:
                    u[0][1] *= factor
                elif self.units.lattice.D == 3:
                    nz = x[2].shape[1]
                    plane_yz = np.ones_like(u[0, 1, :, :])
                    u[0][1] = np.einsum('y,yz->yz', factor, u[0][1])
                    factor = 1 + np.sin(np.linspace(0, nz, nz) / nz * 2 * np.pi) * 0.1  # pertubation in z-direction
                    u[0][1] = np.einsum('z,yz->yz', factor, u[0][1])
        return p, u

    @property
    def obstacle_mask(self):
        return self._obstacle_mask

    @obstacle_mask.setter
    def obstacle_mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == self.shape
        self._obstacle_mask = m.astype(bool)
        # self.solid_mask[np.where(self._obstacle_mask)] = 1  # (!) this line is not doing what it should! solid_mask is now defined in the initial solution (see below)!

    @property
    def grid(self):
        # THIS IS NOT USED AT THE MOMENT. QUESTION: SHOULD THIS BE ONE- OR ZERO-BASED? Indexing or "node-number"?
        xyz = tuple(self.units.convert_length_to_pu(torch.linspace(0, n, n)) for n in self.shape)  # tuple of lists of x,y,(z)-values/indices
        return torch.meshgrid(*xyz, indexing='ij')  # meshgrid of x-, y- (und z-)values/indices

    @property
    def boundaries(self):
        # inlet ("left side", x[0],y[1:-1], z[:])
        inlet_boundary = EquilibriumBoundaryPU(  # inlet
            self.in_mask,
            self.units.lattice, self.units, self.units.characteristic_velocity_pu * self._unit_vector()
        )

        # # outlet ("right side", x[-1],y[:], (z[:]))
        # if self.units.lattice.D == 2:
        #     outlet_boundary = EquilibriumOutletP(self.units.lattice, [1, 0])  # outlet in positive x-direction
        # else: # self.units.lattice.D == 3:
        #     outlet_boundary = EquilibriumOutletP(self.units.lattice, [1, 0, 0])  # outlet in positive x-direction

        outlet_boundary = EquilibriumBoundaryPU(  # outlet
                self.out_mask,
                self.units.lattice, self.units, self.units.characteristic_velocity_pu * self._unit_vector()
            )

        # obstacle (for example: obstacle "cylinder" with radius centered at position x_pos, y_pos) -> to be set via obstacle_mask.setter
        obstacle_boundary = None
        # (!) the obstacle_boundary should alway be the last boundary in the list of boundaries to correctly calculate forces on the obstacle
        if self.bc_type == 'ibb1' or self.bc_type == 'IBB1':
            obstacle_boundary = InterpolatedBounceBackBoundary(self.obstacle_mask, self.units.lattice,
                                                               x_center=(self.shape[1] / 2 - 0.5),
                                                               y_center=(self.shape[1] / 2 - 0.5), radius=self.radius)
        elif self.bc_type == 'ibb1c1':
            obstacle_boundary = InterpolatedBounceBackBoundary_compact_v1(self.obstacle_mask, self.units.lattice,
                                                               x_center=(self.shape[1] / 2 - 0.5),
                                                               y_center=(self.shape[1] / 2 - 0.5), radius=self.radius)
        elif self.bc_type == 'ibb1c2':
            obstacle_boundary = InterpolatedBounceBackBoundary_compact_v2(self.obstacle_mask, self.units.lattice,
                                                                      x_center=(self.shape[1] / 2 - 0.5),
                                                                      y_center=(self.shape[1] / 2 - 0.5),
                                                                      radius=self.radius)
        else:
            return ValueError("Invalid boundary")

        return [
            # TODO: add top boundary
            inlet_boundary,
            outlet_boundary,
            obstacle_boundary
        ]

    def _unit_vector(self, i=0):
        return np.eye(self.units.lattice.D)[i]