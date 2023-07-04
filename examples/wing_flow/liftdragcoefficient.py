import torch
from lettuce import Observable, BounceBackBoundary
import numpy as np


### NEW FULLWAY BOUNCE BACK BOUNDARY CONDITION (with force calculation)
class FullwayBounceBackBoundary:
    """Fullway Bounce-Back Boundary (with added force_on_boundary calculation)
    - fullway = inverts populations within two substeps
    - call() must be called after Streaming substep
    - calculates the force on the boundary:
        - calculation is done after streaming, but theoretically the force is evaluated based on the populations touching/crossing the boundary IN this streaming step
    """
    # based on Master-Branch "class BounceBackBoundary"
    # added option to calculate force on the boundary by Momentum Exchange Method

    def __init__(self, mask, lattice):
        self.mask = lattice.convert_to_tensor(mask)  # which nodes are solid
        self.lattice = lattice
        self.force_sum = torch.zeros_like(self.lattice.convert_to_tensor(self.lattice.stencil.e[0]))  # summed force vector on all boundary nodes, in D dimensions (x,y,(z))
        ### create f_mask, needed for force-calculation
        # ...(marks all fs which streamed into the boundary in prior streaming step)
        if self.lattice.D == 2:
            nx, ny = mask.shape  # domain size in x and y
            self.f_mask = np.zeros((self.lattice.Q, nx, ny), dtype=bool)
                # f_mask: [q, nx, ny], marks all fs on the boundary-border, which point into the boundary/solid
            self.force = np.zeros((nx, ny, 2))  # force in x and y on all individual nodes
            a, b = np.where(mask)
                # np.arrays: list of (a) x-coordinates and (b) y-coordinates in the boundary.mask
                # ...to enable iteration over all boundary/wall/object-nodes
            for p in range(0, len(a)):  # for all TRUE-nodes in boundary.mask
                for i in range(0, self.lattice.Q):  # for all stencil-directions c_i (lattice.stencil.e in lettuce)
                    try:  # try in case the neighboring cell does not exist (= an f pointing out of the simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1]]:
                            # if the neighbour of p is False in the boundary.mask, p is a solid node, neighbouring a fluid node:
                            # ...the direction pointing from the fluid neighbour to solid p is marked on the solid p
                            self.f_mask[self.lattice.stencil.opposite[i], a[p], b[p]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        if self.lattice.D == 3:  # like 2D, but in 3D...guess what...
            nx, ny, nz = mask.shape
            self.f_mask = np.zeros((self.lattice.Q, nx, ny, nz), dtype=bool)
            self.force = np.zeros((nx, ny, nz, 3))
            a, b, c = np.where(mask)
            for p in range(0, len(a)):
                for i in range(0, self.lattice.Q):
                    try:  # try in case the neighboring cell does not exist (an f pointing out of simulation domain)
                        if not mask[a[p] + self.lattice.stencil.e[i, 0], b[p] + self.lattice.stencil.e[i, 1], c[p] + self.lattice.stencil.e[i, 2]]:
                            self.f_mask[self.lattice.stencil.opposite[i], a[p], b[p], c[p]] = 1
                    except IndexError:
                        pass  # just ignore this iteration since there is no neighbor there
        self.f_mask = self.lattice.convert_to_tensor(self.f_mask)

    def __call__(self, f):
        # FULLWAY-BBBC: inverts populations on all boundary nodes

        # calc force on boundary:
        self.calc_force_on_boundary(f)

        # bounce (invert populations on boundary nodes)
        f = torch.where(self.mask, f[self.lattice.stencil.opposite], f)
        return f

    def make_no_collision_mask(self, f_shape):
        assert self.mask.shape == f_shape[1:]
        return self.mask

    def calc_force_on_boundary(self, f):
        # calculate force on boundary by momentum exchange method (MEA, MEM) according to Kruger et al., 2017, pp.215-217:
            # momentum (f_i*c_i - f_i_opposite*c_i_opposite = 2*f_i*c_i for a resting boundary) is summed for all...
            # ...populations pointing at the surface of the boundary
        tmp = torch.where(self.f_mask, f, torch.zeros_like(f))  # all populations f in the fluid region, which point at the boundary
        self.force_sum = 2 * torch.einsum('i..., id -> d', tmp, self.lattice.e)  # CALCULATE FORCE / v2.0 - M.Bille: dx_lu = dt_lu is allways 1 (!)
            # explanation for 2D:
                # sums forces in x and in y (and z) direction,
                # tmp: all f, that are marked in f_mask
                    # tmp.size: 9 x nx x ny (for 2D)
                # self.lattice.e: 9 x 2 (for 2D)
                # - the multiplication of f_i and c_i is down through the first dimension (q) = direction, indexname i
                # - the sign is given by the coordinates of the stencil-vectors (e[0 to 8] for 2D)
                # -> results in twodimensional output (index d) for x- and y-direction (for 2D)
                # "dx**self-lattice.D" = dx³ (3D) or dx² (2D) as prefactor, converting momentum density to momentum
                    # theoretically DELTA P (difference in momentum density) is calculated
                    # assuming smooth momentum transfer over dt, force can be calculated through: F= dP/dt
                    # ...that's why theoretically dividing by dt=dx=1 is necessary (BUT: c_i=1=dx/dt=1 so that can be omitted (v2.0) !)

        # calculate Force on all boundary nodes individually:
        if self.lattice.D == 2:
            self.force = 2 * torch.einsum('qxy, qd -> xyd', tmp, self.lattice.e)  # force = [x-coordinate, y-coodrinate, direction (0=x, 1=y)]
        if self.lattice.D == 3:
            self.force = 2 * torch.einsum('qxyz, qd -> xyzd', tmp, self.lattice.e)  # force = [x-coordinate, y-coodrinate, z-coodrinate, direction (0=x, 1=y, 2=z)]
        #print(torch.sum(self.force))
        #print(self.force_sum)


### NEW OBSERVABLES Coefficient of Drag and Lift (relying on force calculation (see above)
class LiftDragCoefficient(Observable):
    """The lift and drag coefficients of an obstacle, calculated using momentum exchange method (MEM, MEA) according to a
    modified version of M.Kliemank's Drag Coefficient Code

    calculates the density, gets the force in x and y direction on the obstacle boundary,
    calculates the coefficient of lift and drag
    """

    def __init__(self, lattice, flow, obstacle_boundary: FullwayBounceBackBoundary, area_x=None, area_y=None):
        super().__init__(lattice, flow)
        self.obstacle_boundary = obstacle_boundary
        if area_x is not None:
            self.area_lu_x = area_x * (self.flow.units.characteristic_length_lu/self.flow.units.characteristic_length_pu) ** (self.lattice.D-1) # crosssectional area of obstacle i LU (! lengthdimension in 2D -> area-dimension = self.lattice.D-1)
        else:
            self.area_lu_x = torch.sum(torch.any(obstacle_boundary.mask, axis=1))
        if area_y is not None:
            self.area_lu_y = area_y * (self.flow.units.characteristic_length_lu/self.flow.units.characteristic_length_pu) ** (self.lattice.D-1) # crosssectional area of obstacle i LU (! lengthdimension in 2D -> area-dimension = self.lattice.D-1)
        else:
            self.area_lu_y = torch.sum(torch.any(obstacle_boundary.mask, axis=1))

    def __call__(self, f):
        #rho = torch.mean(self.lattice.rho(f[:, 0, ...]))  # simple rho_mean, including the boundary region
        # rho_mean (excluding boundary region):
        rho_tmp = torch.where(self.obstacle_boundary.mask, self.lattice.convert_to_tensor(torch.nan), self.lattice.rho(f))
        rho = torch.nanmean(rho_tmp)
        #print(self.obstacle_boundary)
        #print(self.obstacle_boundary.force_sum[1])
        force_y_lu = self.obstacle_boundary.force_sum[1]  # get current force on obstacle in y direction
        #print(force_y_lu)
        lift_coefficient = force_y_lu / (0.5 * rho * self.flow.units.characteristic_velocity_lu ** 2 * self.area_lu_y)  # calculate lift_coefficient in LU
        #print(lift_coefficient)
        force_x_lu = self.obstacle_boundary.force_sum[0]  # get current force on obstacle in x direction
        drag_coefficient = force_x_lu / (0.5 * rho * self.flow.units.characteristic_velocity_lu ** 2 * self.area_lu_x)  # calculate drag_coefficient in LU
        lift_drag_coefficient = torch.tensor([lift_coefficient, drag_coefficient])
        return lift_drag_coefficient
