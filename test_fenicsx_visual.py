import numpy as np
import pyvista as pv
from dolfinx import mesh, fem, plot
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.cpp.la import as_petsc
from ufl import TrialFunction, TestFunction, dx, grad, inner

# Create mesh and function space
domain = mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
V = fem.FunctionSpace(domain, ("CG", 1))

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = fem.Constant(domain, 1.0)
a = inner(grad(u), grad(v)) * dx
L = f * v * dx

# Dirichlet boundary condition
def boundary(x):
    return np.full(x.shape[1], True)

bc = fem.dirichletbc(fem.Constant(domain, 0.0),
                     fem.locate_dofs_geometrical(V, boundary), V)

# Assemble system
a_form = fem.form(a)
L_form = fem.form(L)
A = fem.assemble_matrix(a_form, bcs=[bc])
b = fem.assemble_vector(L_form)
fem.set_bc(b.array, [bc])
b.scatter_forward()

# ✅ Convert to PETSc matrix using C++ backend
A_petsc = as_petsc(A)

# Solve
uh = fem.Function(V)
solver = PETSc.KSP().create(domain.comm)
solver.setType("cg")
solver.getPC().setType("hypre")
solver.setOperators(A_petsc)
solver.solve(b, uh.vector)
uh.x.scatter_forward()

# Save plot with PyVista
if MPI.COMM_WORLD.rank == 0:
    grid = plot.create_vtk_topology(domain, domain.topology.dim)
    grid.point_data["u"] = uh.x.array.real

    p = pv.Plotter(off_screen=True)
    p.add_mesh(grid, scalars="u", cmap="viridis")
    p.view_xy()
    p.screenshot("fenics_poisson_solution.jpg")
    print("✅ Saved solution as 'fenics_poisson_solution.jpg'")
