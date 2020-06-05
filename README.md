# Molecular Dynamics Simulator

A basic molecular dynamics simulator built in Python.
The core computations are reliant on Numpy broadcasting to eliminate the need for nested loops and keep the overall runtime fast.

![](images/equilibration.gif)

## Basic Usage
```python
# import the simulation class
import numpy as np
from MDSim import MDSim


# set up constants and config
kg_conv = 1.66053906660e-27  # kg / u
m_conv = 1e-10  # m / Å
s_conv = 1e-12  # s / ps
J_conv = kg_conv * m_conv ** 2 * (1 / s_conv ** 2)  # J / (u Å^2 / ps^2)

k_B = 1.380649e-23  # [J / K = kg m^2 / (s^2 K)]
k_B = k_B / J_conv  # [u Å^2 / (ps^2 K)]

ε = 1.65e-21  # [J]
ε = ε / J_conv  # [u Å^2 / ps^2]

σ = 3.4e-10  # [m]
σ = σ / m_conv  # [Å]

N = 100
ρ = 0.1  # [1 / Å^2]
L = np.sqrt(N / ρ)  # [Å]

m = 39.9  # [u]

config = {
    "N": 100, # number of particles
    "d": 2, # dimensionality of the lattice, must be 2 or 3
    "m": 39.9, # particle mass
    "T": 150, # temperature at which to keep the heat bath
    "τ": 0.01, # velocity Verlet time step
    "L": L, # lattice side length
    "k_B": k_B, # Boltzmann's constant
    "ε": ε, # Lennard-Jones ε parameter
    "σ": σ, # Lennard-Jones σ parameter
    "seed": 8, # seed for the rng (for reproducibility)
    "n_iter": 1000, # number of velocity Verlet steps to take
    "ensemble_type": "NVT", # NVT (canonical) or NVE (microcanonical)
    "rescale_velocity_interval": 10, # heath bath parameter (only applies to NVT)
    "save_path": "/tmp/mdsim/data", # where to save the data and images
}


# run a simulation and save a plot of the system after the last integration step
mdsim = MDSim(config)
mdsim.velocity_verlet()
mdsim.plot_system(save_as="particles.png")

# run the simulation again, but this time make an animation
config["n_iter"] = 500
mdsim = MDSim(config)
mdsim.velocity_verlet(animate=True, save_as="equilibration.gif")
```
