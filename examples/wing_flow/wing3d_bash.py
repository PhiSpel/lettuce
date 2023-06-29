import os
import lettuce as lt
import torch
from time import time
from naca_obstacle import Naca
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gc
from collections import Counter

#########################
# ARGUMENT PARSING
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--outputdir", default=os.getcwd() + "/wing_data/3D", type=str, help="directory for output data")
parser.add_argument("--name", default='NACA-0012-lowAOA', type=str, help="name of wing profile file")
parser.add_argument("--outputname", default=None, type=str, help="name base of output files")
parser.add_argument("--n_steps", default=20000, type=int, help="number of steps to simulate, overwritten by t_target")
parser.add_argument("--nmax", default=100000, type=int, help="maximum number of steps to simulate, not overwritten")
parser.add_argument("--t_target", default=None, type=float, help="time in PU to simulate")
parser.add_argument("--n_stream", default=None, type=float, help="stream past the profile n_stream times")
parser.add_argument("--n_stream_pre", default=5, type=float, help="stream past the profile n_stream times with low Re")
parser.add_argument("--nreport", default=500, type=int, help="vtk output every nreport steps")
parser.add_argument("--nreport_pre", default=500, type=int, help="vtk output every nreport steps during pre-run")
parser.add_argument("--ntest", default=1000, type=int, help="test for nans every ntest steps")
parser.add_argument("--Ma", default=0.1, type=float, help="Mach number")
parser.add_argument("--Re", default=10000, type=float, help="Reynolds number, set 0 to calculate")
parser.add_argument("--Re_pre", default=1000, type=float, help="Reynolds number for pre-run")
parser.add_argument("--no_cuda", default=0, type=bool, help="Set False to use CPU instead of Cuda")
parser.add_argument("--collision", default="kbc", help="collision operator (bgk, kbc, reg)")
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

# APPLICATION #
# turbine_diameter =
tempC = 10  # degrees celcius
# p = 14.5                   # air pressure
rho = 1.293  # kg/m³ air density
dt_pu = 1e-5  # this should allow up to 25,000 Hz

# DOMAIN #
args["domain_length_x"] = args["x_before"] + args["wing_length"] + args["x_behind"]
if args["nx"] is None:
    args["nx"] = int(args["domain_length_x"] * args["ny"] // 2)
if args["nz"] is None:
    args["nz"] = int(args["ny"] // 2)
args["shape"] = (args["nx"], args["ny"], args["nz"])
args["dx"] = args["domain_length_x"] / args["nx"]  # i.e. resolution
args["n_wing_nose"] = int(args["x_before"] // args["dx"])  # first grid point with wing
args["n_wing_tail"] = int((args["x_before"] + args["wing_length"]) // args["dx"])  # last grid point with wing

# FLOW CHARACTERISTICS #
# Re = 5e6
Ma = args["Ma"]
if args["Re"] == 0:
    lchar = args["wing_length"]  # characteristic length in pu is obstacle length
    temp = tempC + 273.15  # temperature in Kelvin
    visc_dyn = 2.791e-7 * temp ** 0.7355  # dynamic viscosity of air
    visc_kin = visc_dyn / rho  # kinematic viscosity of air
    args["Re"] = args["vchar"] * lchar / visc_kin  # The type of streaming around the foil.
    # Small (1.5m) 8e3,medium (3-5m) 2e5, large (>5m-150m) up to 5e6

# number of steps depends on Mach number and resolution (?)
# nmax = 100000
if args["t_target"] is None and not args["n_stream"] is None:
    args["t_target"] = args["wing_length"] / args["vchar"] * args["n_stream"]  # simulate tmax-seconds
# test for convergence and crash
test_iterations = True

if not "outputname" in args:
    args["outputname"] = args["name"] + '_ny' + str(args["ny"]) + "_Re{:.1e}".format(args["Re"]) + '_Ma' + str(Ma)
args["filename_base"] = args["outputdir"] + args["outputname"]

run_name = args["name"] + '_ny' + str(args["ny"]) + "_Re{:.1e}".format(args["Re"]) + "_Ma" + str(Ma)
t = time()

# unpacking args
# t_target = args["t_target"]
# n_steps = args["n_steps"]
# Re = args["Re"]
# collision_type = args["collision"]
filename_base = args["filename_base"]
n_test = args["ntest"]
wing_length = args["wing_length"]
nreport = args["nreport"]
ntest = args["ntest"]

flow = Naca(**args)
tau = flow.units.relaxation_parameter_lu
# collision operator
collision_type = args["collision"]
if collision_type == "kbc":
    collision = lt.KBCCollision3D(flow.lattice, tau)
elif collision_type == "reg":
    collision = lt.RegularizedCollision(flow.lattice, tau)
elif collision_type == "bgk":
    collision = lt.BGKCollision(flow.lattice, tau)
else:
    assert ValueError("collision must be set to kbc, reg, or bgk")
simulation = lt.Simulation(flow, flow.lattice, collision, lt.StandardStreaming(flow.lattice))
if args["t_target"] is not None:
    args["n_steps"] = flow.units.convert_time_to_lu(args["t_target"])
t_target = flow.units.convert_velocity_to_pu(args["n_steps"])
print("Key paramters of ", args["outputname"], ": {:.0e}".format(args["n_steps"]), "steps, chord length", wing_length,
      "[m], Re {:.2e}".format(args["Re"]), "[1], Ma {:.2f}".format(Ma))
print("Doing up to {:.0e}".format(args["n_steps"]), " steps.")
print("I will record every", nreport, "-th step, print every", n_test, "-th step. ",
      "1 step corresponds to {:.4f}".format(t_target / args["n_steps"]), "seconds.\nReports are in ", filename_base)

# set up reporters
energy = lt.IncompressibleKineticEnergy(flow.lattice, flow)
# energy_reporter_internal = lt.ObservableReporter(energy, interval=nreport, out=None)
# simulation.reporters.append(energy_reporter_internal)
simulation.reporters.append(lt.ObservableReporter(energy, interval=ntest))  # print energy
simulation.reporters.append(lt.VTKReporter(flow.lattice, flow, interval=nreport, filename_base=filename_base))

# initialize simulation
simulation.initialize_f_neq()
if test_iterations:
    mlups = 0
    it = 0
    i = 0
    while it <= args["n_steps"]:
        i += 1
        it += ntest
        mlups += simulation.step(ntest)
        energy_test = energy(simulation.f).cpu().mean().item()
        # print("avg MLUPS: ", mlups / (i + 1))
        if not energy_test == energy_test:
            print("CRASHED!")
            break
else:
    mlups = simulation.step(args["n_steps"])
    print("MLUPS: ", mlups)

print(run_name, " took ", time() - t, " s")

# Tidying up: Reading allocated memories

print(torch.cuda.memory_summary(device="cuda:0"))

# list present torch tensors:
txt_all_tensors = args["filename_base"] + "_GPU_list_of_tensors.txt"
output_file = open(txt_all_tensors, "a")
total_bytes = 0
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            output_file.write("\n" + str(obj.size()) + ", " + str(obj.nelement() * obj.element_size()))
            total_bytes = total_bytes + obj.nelement() * obj.element_size()
    except:
        pass
output_file.write("\n\ntotal bytes for tensors:" + str(total_bytes))
output_file.close()

# count occurence of tensors in list of tensors:
my_file = open(txt_all_tensors, "r")
data = my_file.read()
my_file.close()
data_into_list = data.split("\n")
c = Counter(data_into_list)
txt_counted_tensors = args["filename_base"] + "_GPU_counted_tensors.txt"
output_file = open(txt_counted_tensors, "a")
for k, v in c.items():
    output_file.write("type,size,bytes: {}, number: {}\n".format(k, v))
output_file.write("\ntotal bytes for tensors:" + str(total_bytes))
output_file.close()

# TODO: Append output to a csv file
