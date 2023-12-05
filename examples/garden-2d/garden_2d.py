import os
import lettuce as lt
import torch
from time import time
from mask_from_csv import Garden
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from liftdragcoefficient import LiftDragCoefficient

#########################
# ARGUMENT PARSING
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--outputdir", default=os.getcwd() + "/garden_data/", type=str, help="directory for output data")
parser.add_argument("--n_steps", default=50000, type=int, help="number of steps to simulate, overwritten by t_target")
parser.add_argument("--nmax", default=50000, type=int, help="maximum number of steps to simulate, not overwritten")
parser.add_argument("--t_target", default=None, type=float, help="time in PU to simulate")
parser.add_argument("--n_stream", default=None, type=float, help="how often to stream past the profile with the given speed")
parser.add_argument("--nreport", default=500, type=int, help="vtk report every n steps")
parser.add_argument("--ntest", default=1000, type=int, help="test for nans every n steps")
parser.add_argument("--Ma", default=0.1, type=float, help="Mach number")
parser.add_argument("--Re", default=2000, type=float, help="Reynolds number, set 0 to calculate")
parser.add_argument("--no_cuda", default=0, type=bool, help="Set False to use CPU instead of Cuda")
parser.add_argument("--collision", default="kbc", help="collision operator (bgk, kbc, reg)")
parser.add_argument("--ny", default=2000, type=int, help="lattice nodes in y-direction")
parser.add_argument("--nx", default=10000, type=int, help="lattice nodes in x-direction")

args = vars(parser.parse_args())
args["debug"] = True
args["show"] = True

# test for convergence and crash
test_iterations = True

## LETTUCE PARAMETERS ##
if args["no_cuda"] == 1:
    lattice = lt.Lattice(lt.D2Q9, torch.device("cpu"), use_native=False)
    print("Not using CUDA, but CPU.")
else:
    lattice = lt.Lattice(lt.D2Q9, torch.device("cuda"), use_native=False)
    print("Using CUDA.")


def setup_simulation(**args):
    t_target = args["t_target"]
    n_steps = args["n_steps"]
    outputdir = args["outputdir"]
    Re = args["Re"]

    file_name = "garden_Re{:.1e}".format(args["Re"]) + '_Ma' + str(args["Ma"])
    args["filename_base"] = outputdir + file_name
    flow = Garden(**args)
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
    # print("Doing up to {:.0e}".format(n_steps), " steps.")
    print("Key paramters of ", file_name, ": {:.0e}".format(n_steps), "steps, Re {:.2e}".format(Re), "[1], Ma {:.2f}".format(args["Ma"]))
    print("I will record every", args["nreport"], "-th step, print every", args["ntest"], "-th step. ",
          "100 steps correspond to {:.2f}".format(t_target / n_steps * 1e2), "seconds.\nReports are in ", args["filename_base"])

    # set up reporters
    energy = lt.IncompressibleKineticEnergy(lattice, flow)
    # energy_reporter_internal = lt.ObservableReporter(energy, interval=nreport, out=None)
    # simulation.reporters.append(energy_reporter_internal)
    simulation.reporters.append(lt.ObservableReporter(energy, interval=args["ntest"]))  # print energy
    simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=args["nreport"], filename_base=args["filename_base"]))
    drag_filename = args["outputdir"] + "/liftdrag.txt"
    print("Lift and drag coefficients are in", drag_filename)
    dragobservable = LiftDragCoefficient(lattice, flow, simulation._boundaries[
        -1])  # ! area A=2*r is in PU and 1-dimensional in 2D
    dragfile = open(drag_filename, "a")
    dragfile.write("Lift and Drag reporters.\nT, Ekin, c_L, c_D")
    dragreport = lt.ObservableReporter(dragobservable, out=dragfile)
    simulation.reporters.append(dragreport)
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
            energy_old = energy_test
            energy_test = energy(simulation.f).mean().item()
            # print("avg MLUPS: ", mlups / (i + 1))
            if not energy_test == energy_test:
                print("CRASHED!")
                break
            if abs(energy_test - energy_old) < 1e-6:
                print("Convergence!")
                break
    else:
        mlups = simulation.step(n_steps)
        print("MLUPS: ", mlups)
    return


run_name = "garden_Re{:.1e}".format(args["Re"]) + "_Ma" + str(args["Ma"])
t = time()
sim, ener, args["n_steps"] = setup_simulation(**args)
run_n_plot(sim, ener, **args)
print(run_name, " took ", time() - t, " s")

