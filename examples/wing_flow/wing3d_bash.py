import os
import lettuce as lt
import torch
from time import time
from naca_obstacle import Naca
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

#########################
# ARGUMENT PARSING
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--outputdir", default=os.getcwd() + "/data/3D", type=str, help="directory for output data")
parser.add_argument("--n_steps", default=20000, type=int, help="number of steps to simulate, overwritten by t_target")
parser.add_argument("--nmax", default=100000, type=int, help="maximum number of steps to simulate, not overwritten")
parser.add_argument("--t_target", default=None, type=float, help="time in PU to simulate")
parser.add_argument("--n_stream", default=None, type=float, help="how often to stream past the profile with the given speed")
parser.add_argument("--nreport", default=500, type=int, help="vtk report every n steps")
parser.add_argument("--ntest", default=1000, type=int, help="test for nans every n steps")
parser.add_argument("--Ma", default=0.1, type=float, help="Mach number")
parser.add_argument("--Re", default=10000, type=float, help="Reynolds number, set 0 to calculate")
parser.add_argument("--no_cuda", default=0, type=bool, help="Set False to use CPU instead of Cuda")
parser.add_argument("--collision", default="kbc", help="collision operator (bgk, kbc, reg)")
parser.add_argument("--name", default='NACA-0012-lowAOA', type=str, help="name of wing profile file")
parser.add_argument("--ny", default=100, type=int, help="lattice nodes in y-direction")
parser.add_argument("--nx", default=None, type=int, help="lattice nodes in x-direction")
parser.add_argument("--nz", default=None, type=int, help="lattice nodes in z-direction")
parser.add_argument("--x_before", default=1, type=int, help="physical space before wing")
parser.add_argument("--x_behind", default=4, type=int, help="physical space behind wing")
parser.add_argument("--vchar", default=5, type=int,
                    help="usually medium streaming velocity (may also be maximum velocity, around 1.5-times) "
                         "large wind turbines produce maximum power at 15 m/s. "
                         "This can be assumed to be streaming velocity around the centre")
parser.add_argument("--wing_length", default=4, type=int, help="'depth' of airfoil profile")

args = vars(parser.parse_args())

## APPLICATION ##
# turbine_diameter =
tempC = 10  # degrees celcius
# p = 14.5                   # air pressure
rho = 1.293  # kg/m³ air density
dt_pu = 1e-5  # this should allow up to 25,000 Hz

## DOMAIN ##
args["domain_length_x"] = args["x_before"] + args["wing_length"] + args["x_behind"]
if args["nx"] is None:
    args["nx"] = int(args["domain_length_x"]*args["ny"]//2)
if args["nz"] is None:
    args["nz"] = int(args["ny"]//2)
args["dx"] = args["domain_length_x"] / args["nx"]  # i.e. resolution
args["n_wing_nose"] = int(args["x_before"] // args["dx"])  # first grid point with wing
args["n_wing_tail"] = int((args["x_before"] + args["wing_length"]) // args["dx"])  # last grid point with wing

## FLOW CHARACTERISTICS ##
# Re = 5e6
if args["Re"] == 0:
    lchar = args["wing_length"]  # characteristic length in pu is obstacle length
    temp = tempC + 273.15  # temperature in Kelvin
    visc_dyn = 2.791e-7 * temp ** 0.7355  # dynamic viscosity of air
    visc_kin = visc_dyn / rho  # kinematic viscosity of air
    args["Re"] = args["vchar"] * lchar / visc_kin  # The type of streaming around the foil. Small (1.5m) 8e3,medium (3-5m) 2e5, large (>5m-150m) up to 5e6

# number of steps depends on Mach number and resolution (?)
# nmax = 100000
if args["t_target"] is None and not args["n_stream"] is None:
    args["t_target"] = args["wing_length"] / args["vchar"] * args["n_stream"]  # simulate tmax-seconds
# test for convergence and crash
test_iterations = True

## LETTUCE PARAMETERS ##
if args["no_cuda"] == 1:
    lattice = lt.Lattice(lt.D3Q27, torch.device("cpu"), use_native=False)
    print("Not using CUDA, but CPU.")
else:
    lattice = lt.Lattice(lt.D3Q27, torch.device("cuda"), use_native=False)
    print("Using CUDA.")


def setup_simulation(**args):
    t_target = args["t_target"]
    n_steps = args["n_steps"]
    outputdir = args["outputdir"]
    Re = args["Re"]
    wing_name = args["name"]

    file_name = wing_name + '_ny' + str(args["ny"]) + "_Re{:.1e}".format(args["Re"]) + '_Ma' + str(args["Ma"])
    args["filename_base"] = outputdir + file_name
    shape = (args["nx"], args["ny"], args["nz"])
    flow = Naca(wing_name, shape, lattice, **args)
    tau = flow.units.relaxation_parameter_lu
    # collision operator
    if args["collision"] == "kbc":
        collision = lt.KBCCollision3D(lattice, tau)
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
    # print("Doing up to {:.0e}".format(n_steps), " steps.")
    print("Key paramters of ", file_name, ": {:.0e}".format(n_steps), "steps, chord length", args["wing_length"],
          "[m], Re {:.2e}".format(Re), "[1], Ma {:.2f}".format(args["Ma"]))
    print("I will record every", args["nreport"], "-th step, print every", args["ntest"], "-th step. ",
          "100 steps correspond to {:.2f}".format(t_target / n_steps * 1e2), "seconds.\nReports are in ", args["filename_base"])

    # set up reporters
    energy = lt.IncompressibleKineticEnergy(lattice, flow)
    # energy_reporter_internal = lt.ObservableReporter(energy, interval=nreport, out=None)
    # simulation.reporters.append(energy_reporter_internal)
    simulation.reporters.append(lt.ObservableReporter(energy, interval=args["ntest"]))  # print energy
    simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=args["nreport"], filename_base=args["filename_base"]))
    return simulation, energy, n_steps


def run_n_plot(simulation, energy, **args):
    n_steps = args["n_steps"]
    nmax = args["n_steps"]

    # initialize simulation
    simulation.initialize_f_neq()
    energy_test = energy(simulation.f).mean().item()
    if not energy_test == energy_test:
        print("Pre-run crashed!")
        return
    if test_iterations:
        mlups = 0
        it = 0
        i = 0
        while it <= nmax:
            i += 1
            it += args["ntest"]
            mlups += simulation.step(args["ntest"])
            energy_test = energy(simulation.f).mean().item()
            # print("avg MLUPS: ", mlups / (i + 1))
            if not energy_test == energy_test:
                print("CRASHED!")
                break
    else:
        mlups = simulation.step(n_steps)
        print("MLUPS: ", mlups)
    return


run_name = args["name"] + '_ny' + str(args["ny"]) + "_Re{:.1e}".format(args["Re"]) + "_Ma" + str(args["Ma"])
t = time()
sim, ener, args["n_steps"] = setup_simulation(**args)
run_n_plot(sim, ener, **args)
print(run_name, " took ", time() - t, " s")

# TODO: Append output to a csv file
