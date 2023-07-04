import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np

nreport = 200
Ma = 0.1
nx = 200
ny = 60
Re = 1000
nmax = 5000
domain_length_x = 10.1


class ObstacleSlip(lt.Obstacle):
    def __init__(self, shape, reynolds_number, mach_number, lattice, domain_length_x, wall):
        super(ObstacleSlip, self).__init__(shape, reynolds_number, mach_number, lattice, domain_length_x)
        self.wall = wall

    @property
    def boundaries(self):
        x, y = self.grid
        mask_wall = np.zeros(x.shape, dtype=bool)
        inflow = np.zeros(self.shape, dtype=bool)
        mask_wall[:, [0, -1]] = True
        inflow[0, :] = True
        mask_obstacle = ((x >= 1) & (x < 2) & (y >= x) & (y <= 2))
        inlet = lt.EquilibriumBoundaryPU(inflow, self.units.lattice, self.units,
                                         self.units.characteristic_velocity_pu * self._unit_vector())
        outlet = lt.EquilibriumOutletP(self.units.lattice, self._unit_vector().tolist())
        obstacle = lt.BounceBackBoundary(mask_obstacle, self.units.lattice)
        if self.wall == "slip":
            wall = lt.SlipBoundary(mask_wall, lattice=self.units.lattice, direction=1)
        elif self.wall == "stick":
            wall = lt.BounceBackBoundary(mask_wall, lattice=self.units.lattice)
        else:
            return [inlet, outlet, obstacle]
        return [inlet, outlet, wall, obstacle]


for wall in ["slip", "stick", "period"]:
    lattice = lt.Lattice(lt.D2Q9, device="cuda", use_native=False)
    flow = ObstacleSlip((nx, ny), Re, Ma, lattice, domain_length_x, wall)
    plt.imshow(flow.mask.T, origin="lower")
    collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = lt.StandardStreaming(lattice)
    simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=nreport,
                                               filename_base="./data/slipboundary/obstacle_" + wall))

    simulation.initialize_f_neq()
    mlups = simulation.step(num_steps=nmax)
    print("Performance in MLUPS:", mlups)
