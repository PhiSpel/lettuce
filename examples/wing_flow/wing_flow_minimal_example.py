import lettuce as lt
import torch
import numpy as np
from matplotlib import pyplot as plt
from time import time
from maskfromcsv import mask_from_csv

t = time()

Ma = 0.05  ## The speed of streaming
wing_name = 'NACA-63215-lowAOA'
filename = wing_name

### APPLICATION ###
# turbine_diameter =
wing_length_x = 2  ## 'depth' of airfoil profile
tempC = 10  ## degrees celcius
# p = 14.5                   ## air pressure
rho = 1.293  ## kg/m³ air density
vchar = 5  ## usually medium streaming velocity (may also be maximum velocity, around 1.5-times)
## large wind turbines produce maximum power at 15 m/s. This can be assumed to be streaming velocity around the centre
dt_pu = 2e-5  ## this should allow up to 25,000 Hz

### DOMAIN ###
nx = 2000  ## number of lattice nodes in x-direction
ny = 500  ## number of lattice nodes in y-direction
shape = (nx, ny)  ## domain shape
x_wing_nose = 1  ## physical space before wing
x_wing_tail = 3  ## physical space behind wing
chord_length = wing_length_x  ## physical length of wing
domain_length_x = x_wing_nose + wing_length_x + x_wing_tail
dx = domain_length_x / nx  ## i.e. resolution
n_wing_nose = int(x_wing_nose // dx)  ## first grid point with wing
n_wing_tail = int(x_wing_tail // dx)  ## first grid point with wing

### FLOW CHARACTERISTICS ###
# Re = 5e6
lchar = wing_length_x  ## characteristic length in pu is obstacle length
temp = tempC + 273.15  ## temperature in Kelvin
visc_dyn = 2.791e-7 * temp ** 0.7355  ## dynamic viscosity of air
visc_kin = visc_dyn / rho  ## kinematic viscosity of air
Re = vchar * lchar / visc_kin  # # The type of streaming around the foil. Small (1.5m) 8e3,medium (3-5m) 2e5,
# large (>5m-150m) up to 5e6

### SIMULATION PARAMETERS ##
# nmax = 100000
n_stream = 5  ## air should have passed the wing length n_stream-times
tmax = wing_length_x / vchar * n_stream  ## simulate tmax-seconds
# how often to report (every n simulation steps)
nreport = 500
# how often to print (every n simulation steps)
nconsole = 2000
# how often to plot (every n console steps)
nplot = 1
# test for convergence and crash
test_iterations = True
test_convergence = False
epsilon = 1e-7
# run pre-simulation with low Re to get rid of initialization pulses
Re_pre = 1000
n_pre = 2000  # wing profiles with camber may crash at low Re

char_length_lu = shape[0] / domain_length_x * lchar
def convert_length_to_pu(nx):
    return nx * lchar / char_length_lu
xyz = tuple(convert_length_to_pu(np.arange(n)) for n in shape)
[x, y] = np.meshgrid(*xyz, indexing='ij')
bool_mask = mask_from_csv(x, y, wing_name, n_wing_nose, n_wing_tail, wing_length_x)

### LETTUCE PARAMETERS ###
lattice = lt.Lattice(lt.D2Q9, torch.device("cuda:0"), use_native=False)
flow = lt.Obstacle(shape,
                   reynolds_number=Re,
                   mach_number=Ma,
                   lattice=lattice,
                   domain_length_x=domain_length_x,
                   char_length=chord_length,
                   char_velocity=vchar)
flow.mask = bool_mask
filename_base = r"/media/philipp/Storage/data/" + filename
tau = flow.units.relaxation_parameter_lu
simulation = lt.Simulation(flow,
                           lattice,
                           lt.KBCCollision2D(lattice, tau),
                           lt.StandardStreaming(lattice))
nmax = flow.units.convert_time_to_lu(tmax)
print("Doing up to ", "{:.2e}".format(nmax), " steps.")
print("Key paramters: Chord length", chord_length, "[m], Re", "{:.2e}".format(Re), "[]")
print("I will record every", nreport, "-th step, print every", nconsole, "-th step, and plot every", nconsole * nplot,
      "-th step.\n",
      "1000 steps correspond to", tmax / nmax * 1e3, "seconds.\n")

# set up reporters
Energy = lt.IncompressibleKineticEnergy(lattice, flow)
simulation.reporters.append(lt.ObservableReporter(Energy, interval=nconsole))  # print energy
simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=nreport, filename_base=filename_base))

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

print("finished after ", time()-t, "seconds")
