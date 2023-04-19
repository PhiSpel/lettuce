import lettuce as lt
import torch
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

nx = 1000
n_wing_nose = int(nx//4)    # wing starts after 1/4 of domain length
n_wing_tail = int(nx*3//4)  # wing goes until 3/4 of domain length
ny = 500
n_wing_height = int(ny//2)  # wing sits at middle of domain length
Re = 50.0
Ma = 0.05
lattice = lt.Lattice(lt.D2Q9, device=torch.device("cuda:0"))
length_x = 10.1
flow = lt.Obstacle(
    (nx, ny),
    reynolds_number=Re,
    mach_number=Ma,
    lattice=lattice,
    domain_length_x=length_x,
)
x, y = flow.grid

# read wing data
top_surface_data = np.fromfile('top_surface.csv', sep=' ', dtype=float)
top_surface_data = np.reshape(top_surface_data, (int(len(top_surface_data)/2),2)).transpose()
bottom_surface_data = np.fromfile('bottom_surface.csv', sep=' ', dtype=float)
bottom_surface_data = np.reshape(bottom_surface_data, (int(len(bottom_surface_data)/2),2)).transpose()

x_data_top = top_surface_data[0]
y_data_top = top_surface_data[1]
x_data_bottom = bottom_surface_data[0]
y_data_bottom = bottom_surface_data[1]

# calculate scaling factor
x_wing_nose = x[n_wing_nose,0] # 2.525
x_wing_tail = x[n_wing_tail,0] # 7.575
available_length_x = x_wing_tail - x_wing_nose # 5.05
available_length_n = n_wing_tail-n_wing_nose
actual_wing_length_x = max(max(x_data_top), max(x_data_bottom)) # 100
scaling_factor = available_length_x / actual_wing_length_x # 0.0505
# scale wing data to fit domain restrictions
x_data_top *= scaling_factor    # [5.05 ... 0]
x_data_bottom *= scaling_factor # [0 ... 5.05]
y_data_top *= scaling_factor    # [0 ... 0.39 ... 0]
y_data_bottom *= scaling_factor # [0 ... -0.20 ... 0]

# mapping data to the grid
x_data_interp        = np.linspace(0, available_length_x, available_length_n)            # [0 ... 5.05]
y_data_top_interp    = interpolate.interp1d(x_data_top, y_data_top)(x_data_interp)       # .interp1d object
y_data_bottom_interp = interpolate.interp1d(x_data_bottom, y_data_bottom)(x_data_interp) # .interp1d object

# shifting the wing up by half the grid
y_wing_height = y[0, n_wing_height]
y_data_top_interp += y_wing_height
y_data_bottom_interp += y_wing_height

# setting y data in a 2D grid to compare with flow.grid[1]
y_data_top_mapped = np.zeros(np.shape(y))
y_data_top_mapped[n_wing_nose:n_wing_tail, :] = np.array([y_data_top_interp]).transpose()  # np.tile(np.array([y_data_top_interp]).transpose(), (1, available_length_n))
y_data_bottom_mapped = np.zeros(np.shape(y))
y_data_bottom_mapped[n_wing_nose:n_wing_tail, :] = np.array([y_data_bottom_interp]).transpose()  # np.tile(np.array([y_data_top_interp]).transpose(), (1, available_length_n))

# creating and plotting mask
flow.mask = ((y < y_data_top_mapped) & (y > y_data_bottom_mapped))
plt.imshow(flow.mask.transpose(), origin="lower", cmap='gray_r')
plt.show()
