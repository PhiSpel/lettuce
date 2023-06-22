import os

import lettuce as lt
import torch
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt
from time import time
from maskfromcsv import Naca
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

##################################################
# ARGUMENT PARSING
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--outputdir", default=os.getcwd() + "/data", type=str, help="directory for output data")
parser.add_argument("--n_steps", default=50000, type=int,
                    help="number of steps to simulate, overwritten by t_target, if t_target is >0")
parser.add_argument("--t_target", default=None, type=float, help="time in PU to simulate")
parser.add_argument("--n_stream", default=None, type=float, help="time in PU to simulate")
parser.add_argument("--Ma", default=0.1, type=float, help="Mach number")
parser.add_argument("--Re", default=2000, type=float, help="Reynolds number")
parser.add_argument("--no-cuda", default=True, type=float, help="Only use CPU. Set False to use Cuda")
parser.add_argument("--collision", default="bgk", help="collision operator (bgk, kbc, reg)")

args = vars(parser.parse_args())

outputdir = args["outputdir"]
Ma = args["Ma"]  ## The speed of streaming
no_cuda = args["no-cuda"]

### APPLICATION ###
# turbine_diameter =
wing_length = 1  ## 'depth' of airfoil profile
tempC = 10  ## degrees celcius
# p = 14.5                   ## air pressure
rho = 1.293  ## kg/m³ air density
vchar = 5  ## usually medium streaming velocity (may also be maximum velocity, around 1.5-times)
## large wind turbines produce maximum power at 15 m/s. This can be assumed to be streaming velocity around the centre
dt_pu = 1e-5  ## this should allow up to 25,000 Hz

### DOMAIN ###
nx = 600  ## number of lattice nodes in x-direction
ny = 150  ## number of lattice nodes in y-direction
# shape = (nx, ny)            ## domain shape
x_wing_nose = 1  ## physical space before wing
x_wing_tail = 3  ## physical space behind wing
chord_length = wing_length  ## physical length of wing
domain_length_x = x_wing_nose + wing_length + x_wing_tail
dx = domain_length_x / nx  ## i.e. resolution
n_wing_nose = int(x_wing_nose // dx)  ## first grid point with wing
n_wing_tail = int(x_wing_tail // dx)  ## first grid point with wing

### FLOW CHARACTERISTICS ###
# Re = 5e6
lchar = wing_length  ## characteristic length in pu is obstacle length
temp = tempC + 273.15  ## temperature in Kelvin
visc_dyn = 2.791e-7 * temp ** 0.7355  ## dynamic viscosity of air
visc_kin = visc_dyn / rho  ## kinematic viscosity of air
Re = vchar * lchar / visc_kin  ## The type of streaming around the foil. Small (1.5m) 8e3,medium (3-5m) 2e5, large (>5m-150m) up to 5e6

### load setup to args
args["domain_length_x"] = domain_length_x
args["x_wing_nose"] = x_wing_nose
args["x_wing_tail"] = x_wing_tail
args["chord_length"] = chord_length
args["vchar"] = vchar
args["wing_length"] = wing_length

# number of steps depends on Mach number and resolution (?)
# nmax = 100000
if args["t_target"] is None:
    if args["n_stream"] is None:
        nmax = args["n_steps"]
    else:
        args["t_target"] = wing_length / vchar * args["n_stream"]  ## simulate tmax-seconds

### SIMULATION PARAMETERS ##
# how often to report (every n simulation steps)
nreport = 500
# how often to print (every n simulation steps)
nconsole = 2000
# how often to plot (every n console steps)
nplot = 5
# test for convergence and crash
test_iterations = True
test_convergence = False
epsilon = 1e-7
# run pre-simulation with low Re to get rid of initialization pulses
Re_pre = 1000
n_pre = 2000  # wing profiles with camber may crash at low Re

wing_dict = {
    'NACA-63215-highAOA',
    'NACA-63215-lowAOA',
    'NACA-63215-noAOA',
    'NACA-0012-lowAOA',
    'NACA-0012-noAOA'
}
re_dict = {
    1e3,
    1e4,
    1e5,
    1e6,
    1e7
}
res_dict = {
    100,
    200,
    300,
    400,
    500,
    600
}
# Re -> vchar -> nmax

### LETTUCE PARAMETERS ###
if no_cuda:
    lattice = lt.Lattice(lt.D2Q9, torch.device("cpu"), use_native=False)
else:
    lattice = lt.Lattice(lt.D2Q9, torch.device("cuda"), use_native=False)

def setup_simulation(wing_name, file_name=None, re_number=Re, n_x=nx, n_y=ny):
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
    simulation = lt.Simulation(flow, lattice, collision, lt.StandardStreaming(lattice))
    if args["t_target"] is not None:
        nmax = flow.units.convert_time_to_lu(args["t_target"])
    else:
        nmax = args["n_steps"]
        args["t_target"] = flow.units.convert_velocity_to_pu(args["n_steps"])
    print("Doing up to ", "{:.2e}".format(nmax), " steps.")
    print("Key paramters: run name:", file_name, ", chord length", chord_length, "[m], Re", "{:.2e}".format(re_number),
          "[1]")
    print("I will record every", nreport, "-th step, print every", nconsole, "-th step, and plot every",
          nconsole * nplot, "-th step.\n",
          "1000 steps correspond to", args["t_target"] / nmax * 1e3, "seconds.\n")

    # set up reporters
    Energy = lt.IncompressibleKineticEnergy(lattice, flow)
    # energy_reporter_internal = lt.ObservableReporter(Energy, interval=nreport, out=None)
    # simulation.reporters.append(energy_reporter_internal)
    simulation.reporters.append(lt.ObservableReporter(Energy, interval=nconsole))  # print energy
    simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=nreport, filename_base=filename_base))
    return simulation, flow, Energy, nmax


def run_n_plot(simulation, flow, Energy, nmax):
    # initialize simulation
    simulation.initialize_f_neq()
    if test_iterations:
        energy_new = 0
        mlups = 0
        iterations = int(nmax // nconsole)
        for i in range(iterations):
            mlups += simulation.step(nconsole)
            energy_old = energy_new
            energy_new = Energy(simulation.f).mean().item()
            rchange = abs((energy_new - energy_old) / energy_new)
            print("avg MLUPS: ", mlups / (i + 1))
            if test_convergence and rchange < epsilon:
                print("CONVERGENCE! Less than ", epsilon * 100, " % relative change")
                break
            elif test_convergence:
                print("no convergence, still ", round(rchange * 100, 5), " % relative change")
            elif not energy_new == energy_new:
                print("CRASHED!")
                break
            if i % nplot == 0:
                u = flow.units.convert_velocity_to_pu(lattice.u(simulation.f).detach().cpu().numpy())
                plt.imshow(u[0, ...].T, origin="lower")
                plt.show()
        u = flow.units.convert_velocity_to_pu(lattice.u(simulation.f).detach().cpu().numpy())
        plt.imshow(u[0, ...].T, origin="lower")
        plt.show()
    else:
        mlups = simulation.step(nmax)
        print("MLUPS: ", mlups)
        u = flow.units.convert_velocity_to_pu(lattice.u(simulation.f).detach().cpu().numpy())
        plt.imshow(u[0, ...].T, origin="lower")
        plt.show()
    return


setup = {}
# do comparison of resolution
for ny in res_dict:
    name = 'NACA-0012-lowAOA'
    run_name = name + '_ny' + str(ny)
    setup[run_name] = setup_simulation(name, run_name, re_number=Re, n_x=4 * ny, n_y=ny)
# do comparison of wings and reynolds numbers
# for name in wing_dict:
#     for Re in re_dict:
#         vchar = Re*visc_kin/lchar
#         run_name = name+'_Re'+"{:.2e}".format(Re)
#         setup[run_name] = setup_simulation(name, run_name, tmax=wing_length/vchar*n_stream, re_number=Re)

for run_name in setup:
    t = time()
    [sim, flo, Ener, n_max] = setup[run_name]
    run_n_plot(sim, flo, Ener, n_max)
    print(run_name, " took ", time() - t, " s\n")
