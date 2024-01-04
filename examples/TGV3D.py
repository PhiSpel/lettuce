import lettuce as lt
import matplotlib.pyplot as plt
import numpy as np
import datetime

nmax = 15000
ncheck = 500
nreport = ncheck


def plot_u(u, title='Velocity'):
    midpoint = int(len(u)//2)
    u_norm = np.linalg.norm(u, axis=0)
    u_xy = u_norm[:, :, midpoint]
    plt.imshow(u_xy)
    plt.title(title)
    plt.show()
    return u


def run(lattice, collision_operator, resolution, reynolds_number, mach_number):
    flow = lt.TaylorGreenVortex3D(resolution=resolution, reynolds_number=reynolds_number, mach_number=mach_number, lattice=lattice)
    collision = collision_operator(lattice, tau=flow.units.relaxation_parameter_lu)
    simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=lt.StandardStreaming(lattice))
    Energy = lt.IncompressibleKineticEnergy(lattice, flow)
    # simulation.reporters.append(lt.ObservableReporter(Energy, interval=nreport, out=None))
    # simulation.reporters.append(lt.ObservableReporter(lt.EnergySpectrum(lattice, flow), interval=500, out=None))
    if lattice.stencil == lt.D3Q19:
        prefix = 'D3Q19_'
    elif lattice.stencil == lt.D3Q27:
        prefix = 'D3Q27_'
    else:
        return
    if collision_operator == lt.BGKCollision:
        prefix += 'BGK_res_'
    elif collision_operator == lt.KBCCollision3D:
        prefix += 'KBC_res_'
    else:
        return
    prefix += str(resolution) + '_Re' + str(reynolds_number) + '_Ma' + str(mach_number)
    f1 = open('data/TGV/'+prefix+'_enstrophy', 'w')
    simulation.reporters.append(lt.ObservableReporter(lt.Enstrophy(lattice, flow), interval=500, out=f1))
    f2 = open('data/TGV/' + prefix + '_energy_spectrum', 'w')
    simulation.reporters.append(lt.ObservableReporter(lt.EnergySpectrum(lattice, flow), interval=500, out=f2))
    f3 = open('data/TGV/'+prefix+'_energy_incompressible', 'w')
    simulation.reporters.append(lt.ObservableReporter(lt.IncompressibleKineticEnergy(lattice, flow), interval=500, out=f3))
    simulation.reporters.append(lt.VTKReporter(lattice, flow, interval=500,
                                               filename_base='/media/philipp/Storage/TGV_data/'+prefix+'_VTK'))

    u = flow.units.convert_velocity_to_pu(lattice.u(simulation.f)).cpu().numpy()
    plot_u(u, 'Initialized velocity' + prefix)

    simulation.initialize_pressure()
    simulation.initialize_f_neq()
    for _ in range(int(nmax//ncheck)):
        simulation.step(ncheck)
        mean_energy = Energy(simulation.f).mean().item()
        u = flow.units.convert_velocity_to_pu(lattice.u(simulation.f)).cpu().numpy()
        plot_u(u, 'Initialized velocity' + prefix)
        if not mean_energy == mean_energy:
            print("CRASHED!")
            return

    u = flow.units.convert_velocity_to_pu(lattice.u(simulation.f)).cpu().numpy()
    plot_u(u, 'Velocity after simulation' + prefix)


for stencil in [lt.D3Q27]:
    lat = lt.Lattice(stencil, device="cuda:0", use_native=False)
    for coll in [lt.KBCCollision3D]:
        for res in [64, 75, 128, 256]:
            for Re in [100, 1e4, 1e5]:
                for Ma in [0.1, 0.05, 0.01]:
                    print('Current simulation running since ', datetime.datetime.now())
                    run(lat, coll, res, Re, Ma)

# for stencil in [lt.D3Q27, lt.D3Q19]:
#     lat = lt.Lattice(stencil, device="cuda:0", use_native=False)
#     for coll in [lt.BGKCollision, lt.KBCCollision3D]:
#         for res in [64, 75, 128, 256]:
#             for Re in [10, 100, 1e4, 1e5]:
#                 for Ma in [0.1, 0.05, 0.01]:
#                     print('Current simulation running since ', datetime.datetime.now())
#                     run(lat, coll, res, Re, Ma)
