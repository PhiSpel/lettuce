import os
import lettuce as lt
import torch
from time import time
from maskfromcsv import Naca
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

#########################
# ARGUMENT PARSING
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--outputdir", default=os.getcwd() + "/data", type=str, help="directory for output data")
parser.add_argument("--n_steps", default=50000, type=int, help="number of steps to simulate, overwritten by t_target")
parser.add_argument("--t_target", default=None, type=float, help="time in PU to simulate")
parser.add_argument("--n_stream", default=None, type=float, help="time in PU to simulate")
parser.add_argument("--Ma", default=0.1, type=float, help="Mach number")
parser.add_argument("--Re", default=2000, type=float, help="Reynolds number")
parser.add_argument("--no_cuda", default=False, type=bool, help="Set False to use CPU instead of Cuda")
parser.add_argument("--collision", default="bgk", help="collision operator (bgk, kbc, reg)")
parser.add_argument("--nreport", default=500, type=int, help="vtk report every n steps")
parser.add_argument("--ntest", default=2000, type=int, help="test for nans every n steps")
parser.add_argument("--name", default='NACA-0012-lowAOA', type=str, help="name of wing profile file")
parser.add_argument("--nx", default=600, type=int, help="lattice nodes in x-direction")
parser.add_argument("--ny", default=150, type=int, help="lattice nodes in y-direction")

args = vars(parser.parse_args())

outputdir = args["outputdir"]
Ma = args["Ma"]  # The speed of streaming
name = args["name"]

## APPLICATION ##
# turbine_diameter =
wing_length = 1  # 'depth' of airfoil profile
tempC = 10  # degrees celcius
# p = 14.5                   # air pressure
rho = 1.293  # kg/m³ air density
vchar = 5  # usually medium streaming velocity (may also be maximum velocity, around 1.5-times)
# large wind turbines produce maximum power at 15 m/s. This can be assumed to be streaming velocity around the centre
dt_pu = 1e-5  # this should allow up to 25,000 Hz

## DOMAIN ##
nx = args["nx"]  # number of lattice nodes in x-direction
ny = 150  # number of lattice nodes in y-direction
# shape = (nx, ny)            # domain shape
x_wing_nose = 1  # physical space before wing
x_wing_tail = 3  # physical space behind wing
chord_length = wing_length  # physical length of wing
domain_length_x = x_wing_nose + wing_length + x_wing_tail
dx = domain_length_x / nx  # i.e. resolution
n_wing_nose = int(x_wing_nose // dx)  # first grid point with wing
n_wing_tail = int(x_wing_tail // dx)  # first grid point with wing

## FLOW CHARACTERISTICS ##
# Re = 5e6
lchar = wing_length  # characteristic length in pu is obstacle length
temp = tempC + 273.15  # temperature in Kelvin
visc_dyn = 2.791e-7 * temp ** 0.7355  # dynamic viscosity of air
visc_kin = visc_dyn / rho  # kinematic viscosity of air
Re = vchar * lchar / visc_kin  # The type of streaming around the foil. Small (1.5m) 8e3,medium (3-5m) 2e5, large (>5m-150m) up to 5e6

## load setup to args
args["domain_length_x"] = domain_length_x
args["x_wing_nose"] = x_wing_nose
args["x_wing_tail"] = x_wing_tail
args["chord_length"] = chord_length
args["vchar"] = vchar
args["wing_length"] = wing_length

# number of steps depends on Mach number and resolution (?)
# nmax = 100000
if args["t_target"] is None and not args["n_stream"] is None:
    args["t_target"] = wing_length / vchar * args["n_stream"]  # simulate tmax-seconds

## SIMULATION PARAMETERS #
# how often to report (every n simulation steps)
nreport = args["nreport"]
# how often to test for nans (every n simulation steps)
ntest = args["ntest"]
# test for convergence and crash
test_iterations = True

## LETTUCE PARAMETERS ##
if args["no_cuda"] == 1:
    lattice = lt.Lattice(lt.D2Q9, torch.device("cpu"), use_native=False)
    print("Not using CUDA, but CPU.")
else:
    lattice = lt.Lattice(lt.D2Q9, torch.device("cuda"), use_native=False)
    print("Using CUDA.")


def setup_simulation(wing_name, file_name=None, re_number=Re, n_x=nx, n_y=ny, **args):
    t_target = args["t_target"]
    n_steps = args["n_steps"]

    if file_name is None:
        filename_base = outputdir + wing_name
    else:
        filename_base = outputdir + file_name
    shape = (n_x, n_y)
    flow = Naca(wing_name, shape, lattice, **args)
    tau = flow.units.relaxation_parameter_lu
    # collision operator
    if args["collision"] == "kbc":
        collision = lt.KBCCollision2D(lattice, tau)
    elif args["collision"] == "reg":
        collision = lt.RegularizedCollision(lattice, tau)
    elif args["collision"] == "bgk":
        collision = lt.BGKCollision(lattice, tau)
    else:
        assert ValueError("collision must be set to kbc, reg, or bgk")
        return
    simulation = lt.Simulation(flow, lattice, collision, lt.StandardStreaming(lattice))
    if t_target is not None:
        n_steps = flow.units.convert_time_to_lu(t_target)
        t_target = flow.units.convert_velocity_to_pu(n_steps)
    print("Doing up to ", "{:.2e}".format(n_steps), " steps.")
    print("Key paramters: run name:", file_name, ", chord length", chord_length, "[m], Re", "{:.2e}".format(re_number),
          "[1]")
    print("I will record every", nreport, "-th step, print every", ntest, "-th step\n",
          "1000 steps correspond to", t_target / n_steps * 1e3, "seconds.\n")

    # set up reporters
    energy = lt.IncompressibleKineticEnergy(lattice, flow)
    # energy_reporter_internal = lt.ObservableReporter(energy, interval=nreport, out=None)
    # simulation.reporters.append(energy_reporter_internal)
    simulation.reporters.append(lt.ObservableReporter(energy, interval=ntest))  # print energy
    simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=nreport, filename_base=filename_base))
    return simulation, energy, n_steps


def run_n_plot(simulation, energy, **args):
    # initialize simulation
    simulation.initialize_f_neq()
    if test_iterations:
        mlups = 0
        iterations = int(args["n_steps"] // ntest)
        for i in range(iterations):
            mlups += simulation.step(ntest)
            energy_new = energy(simulation.f).mean().item()
            print("avg MLUPS: ", mlups / (i + 1))
            if not energy_new == energy_new:
                print("CRASHED!")
                break
    else:
        mlups = simulation.step(args["n_steps"])
        print("MLUPS: ", mlups)
    return


run_name = name + '_ny' + str(ny) + '_Re' + str(Re) + '_Ma' + str(Ma)
t = time()
sim, ener, args["n_steps"] = setup_simulation(name, run_name, re_number=Re, n_x=4 * ny, n_y=ny)
run_n_plot(sim, ener, **args)
print(run_name, " took ", time() - t, " s\n")
