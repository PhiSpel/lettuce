import os
import lettuce as lt
import torch
from time import time
import gc
from collections import Counter
from flatplate import FlatPlate
import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--outputdir", default=os.getcwd() + "/data/", type=str, help="directory for output data")
parser.add_argument("--logfile", default=None, type=str, help="logfile")
parser.add_argument("--outputname", default="flatplate", type=str, help="name base of output files")

args = vars(parser.parse_args())

nx = 5000
ny = 100
nz = 25
Re = 2.5e5
shape = (nx, ny, nz)
# [1] C. Doolan and D. Moreau, Flow Noise: Theory. Singapore: Springer Nature Singapore, 2022. doi: 10.1007/978-981-19-2484-2.
# see here for calculation of domain length
domain_length_x = 20  # [m]
Ma = 0.1
nmax = 100000
nreport = 500
ntest = 5000
filename_base = args["outputdir"] + args["outputname"]
printing = sys.stdout

# LOG FILE #
if args["logfile"] is not None:
    class Logger(object):
        def __init__(self):
            self.terminal = sys.stdout
            self.log = open(args["logfile"], "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            # this flush method is needed for python 3 compatibility.
            # this handles the flush command by doing nothing.
            # you might want to specify some extra behavior here.
            pass
    sys.stdout = Logger()

t = time()

lattice = lt.Lattice(lt.D3Q27, torch.device("cuda"), use_native=False)
flow = FlatPlate(shape, Re, Ma, lattice, domain_length_x)
t_target = flow.units.convert_time_to_pu(nmax)
tau = flow.units.relaxation_parameter_lu
collision = lt.KBCCollision3D(lattice, tau)
simulation = lt.Simulation(flow, lattice, collision, lt.StandardStreaming(lattice))
print("Key paramters of ", args["outputname"], ": {:.0e}".format(nmax), "steps, chord length", domain_length_x,
      "[m], Re {:.2e}".format(Re), "[1], Ma {:.2f}".format(Ma))
print("I will record every", nreport, "-th step, print every", ntest, "-th step. ",
      "1 step corresponds to {:.4f}".format(t_target / nmax), "seconds.\nReports are in ", filename_base)

# set up reporters
energy = lt.IncompressibleKineticEnergy(lattice, flow)
# energy_reporter_internal = lt.ObservableReporter(energy, interval=nreport, out=None)
# simulation.reporters.append(energy_reporter_internal)
simulation.reporters.append(lt.ObservableReporter(energy, interval=ntest))  # print energy
simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=nreport, filename_base=filename_base))

# initialize simulation
simulation.initialize_f_neq()
mlups = 0
it = 0
i = 0
while it <= nmax:
    i += 1
    it += ntest
    mlups += simulation.step(ntest)
    energy_test = energy(simulation.f).cpu().mean().item()
    # print("avg MLUPS: ", mlups / (i + 1))
    if not energy_test == energy_test:
        print("CRASHED!")
        break
    print("avg MLUPS: ", mlups / i)

print(args["outputname"], " took ", time() - t, " s")

# Tidying up: Reading allocated memories

print(torch.cuda.memory_summary(device="cuda:0"))

# list present torch tensors:
txt_all_tensors = filename_base + "_GPU_list_of_tensors.txt"
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
txt_counted_tensors = filename_base + "_GPU_counted_tensors.txt"
output_file = open(txt_counted_tensors, "a")
for k, v in c.items():
    output_file.write("type,size,bytes: {}, number: {}\n".format(k, v))
output_file.write("\ntotal bytes for tensors:" + str(total_bytes))
output_file.close()

# redirect stdout to printing
printing = sys.stdout

# TODO: Append output to a csv file
