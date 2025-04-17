"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Chao Wang (chaow4@illinois.edu)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. Defense Advanced Research Projects Agency (DARPA) Young Faculty Award
  (N660012314013)
- NSF CAREER Award CMMI-2047692
- NSF Award CMMI-2245251

Reference:
- Jia, Y., Wang, C. & Zhang, X.S. FEniTop: a simple FEniCSx implementation
  for 2D and 3D topology optimization supporting parallel computing.
  Struct Multidisc Optim 67, 140 (2024).
  https://doi.org/10.1007/s00158-024-03818-7
"""

import os  # Enables output to results folder
import numpy as np
from mpi4py import MPI
from dolfinx.mesh import create_rectangle, CellType

from fenitop.topopt import topopt

# Create mesh
mesh = create_rectangle(MPI.COMM_WORLD, [[0, 0], [60, 20]],
                        [200, 60], CellType.quadrilateral)
if MPI.COMM_WORLD.rank == 0:
    mesh_serial = create_rectangle(MPI.COMM_SELF, [[0, 0], [60, 20]],
                                   [200, 60], CellType.quadrilateral)
else:
    mesh_serial = None

# FEM settings
fem = {
    "name": "beam_2d_fulledge",  # ← Added: defines output folder name
    "mesh": mesh,
    "mesh_serial": mesh_serial,
    "young's modulus": 100,
    "poisson's ratio": 0.25,
    "disp_bc": lambda x: np.isclose(x[0], 0),
    "traction_bcs": [[(0, -0.2),
                      lambda x: (np.isclose(x[0], 60) & np.greater(x[1], 0) & np.less(x[1], 20))]],
    "body_force": (0, 0),
    "quadrature_degree": 2,
    "petsc_options": {
        "ksp_type": "cg",
        "pc_type": "gamg",
    },
}

# Optimization parameters
opt = {
    "max_iter": 400,
    "opt_tol": 1e-5,
    "vol_frac": 0.5,
    "solid_zone": lambda x: (x[1] < 1.0) | (x[1] > 19.0),   #enforces solid on top and bottom edge
    "void_zone": lambda x: np.full(x.shape[1], False),
    "penalty": 3.0,
    "epsilon": 1e-6,
    "filter_radius": 1.2,
    "beta_interval": 50,
    "beta_max": 128,
    "use_oc": True,
    "move": 0.02,
    "opt_compliance": True,
}

if __name__ == "__main__":
    topopt(fem, opt)

# To execute in parallel:
# mpirun -n 8 python3 beam_2d.py
