import numpy as np
from scipy.spatial import cKDTree
from petsc4py import PETSc
import dolfinx.io
from dolfinx.fem import form, Function
from dolfinx import la
from dolfinx.fem.petsc import (create_vector, create_matrix,
                               assemble_vector, assemble_matrix, set_bc)
import pyvista


def create_mechanism_vectors(func_space, in_spring, out_spring):
    index_map = func_space.dofmap.index_map
    block_size = func_space.dofmap.index_map_bs
    spring_vec = la.create_petsc_vector(index_map, block_size)
    l_vec = spring_vec.copy()

    local_range = index_map.local_range
    local_indices = np.arange(local_range[0], local_range[1]).astype(np.int32)
    local_size = np.ptp(local_range)
    local_nodes = func_space.tabulate_dof_coordinates()[:local_size]

    for n, (locator, direction, value) in enumerate([in_spring, out_spring]):
        ctrl_nodes = local_indices[locator(local_nodes.T)]
        offset = ["x", "y", "z"].index(direction)
        ctrl_dofs = ctrl_nodes * block_size + offset
        spring_vec.setValues(ctrl_dofs, [value] * ctrl_dofs.size)
        if n == 1:
            l_vec.setValues(ctrl_dofs, [1.0] * ctrl_dofs.size)
    spring_vec.assemble()
    l_vec.assemble()
    return spring_vec, l_vec


class LinearProblem:
    def __init__(self, u, lam, lhs, rhs, l_vec, spring_vec, bcs=[], petsc_options={}):
        self.u, self.lam = u, lam
        self.u_wrap = la.create_petsc_vector_wrap(self.u.x)
        self.lam_wrap = la.create_petsc_vector_wrap(self.lam.x)
        self.lhs_form, self.rhs_form = form(lhs), form(rhs)
        self.lhs_mat = create_matrix(self.lhs_form)
        self.rhs_vec = create_vector(self.rhs_form)
        self.bcs, self.l_vec, self.spring_vec = bcs, l_vec, spring_vec

        self.solver = PETSc.KSP().create(self.u.function_space.mesh.comm)
        self.solver.setOperators(self.lhs_mat)
        prefix = f"linear_solver_{id(self)}"
        self.solver.setOptionsPrefix(prefix)

        opts = PETSc.Options()
        opts.prefixPush(prefix)
        for key, value in petsc_options.items():
            opts[key] = value
        opts.prefixPop()
        self.solver.setFromOptions()
        for var in [self.lhs_mat, self.rhs_vec, self.l_vec]:
            if var is not None:
                var.setOptionsPrefix(prefix)
                var.setFromOptions()

        assemble_vector(self.rhs_vec, self.rhs_form)
        self.rhs_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.rhs_vec, self.bcs)

    def solve_fem(self):
        self.lhs_mat.zeroEntries()
        assemble_matrix(self.lhs_mat, self.lhs_form, bcs=self.bcs)
        self.lhs_mat.assemble()
        if self.spring_vec is not None:
            self.lhs_mat.setDiagonal(self.lhs_mat.getDiagonal() + self.spring_vec)
        self.solver.solve(self.rhs_vec, self.u_wrap)
        self.u.x.scatter_forward()

    def solve_adjoint(self):
        self.solver.solve(-self.l_vec, self.lam_wrap)
        self.lam.x.scatter_forward()

    def __del__(self):
        self.solver.destroy()
        self.lhs_mat.destroy()
        self.rhs_vec.destroy()
        self.u_wrap.destroy()
        self.lam_wrap.destroy()
        if self.spring_vec is not None:
            self.spring_vec.destroy()
            self.l_vec.destroy()


class Communicator:
    def __init__(self, func_space, mesh_serial, size=1):
        self.size = size
        self.comm = func_space.mesh.comm
        idx_map = func_space.dofmap.index_map

        num_local_nodes = idx_map.size_local
        num_global_nodes = idx_map.size_global
        num_nodal_dofs = func_space.dofmap.index_map_bs
        self.num_global_dofs = num_global_nodes * num_nodal_dofs

        local_nodal_range = np.asarray(idx_map.local_range, dtype=np.int32)
        local_dof_range = local_nodal_range * num_nodal_dofs
        local_nodes = func_space.tabulate_dof_coordinates()[:num_local_nodes]

        local_nodal_range_gather = self.comm.gather(local_nodal_range, root=0)
        self.local_dof_range_gather = self.comm.gather(local_dof_range, root=0)
        local_nodes_gather = self.comm.gather(local_nodes, root=0)

        element = func_space.ufl_element()
        if self.comm.rank == 0:
            func_space_serial = dolfinx.fem.FunctionSpace(mesh_serial, element)
            nodes_serial = func_space_serial.tabulate_dof_coordinates()

            nodes_collect = np.zeros((num_global_nodes, 3))
            for r, nodes in zip(local_nodal_range_gather, local_nodes_gather):
                nodes_collect[r[0]:r[1]] = nodes
            global_to_local_nodes = compare_matrices(nodes_serial, nodes_collect)
            local_to_global_nodes = compare_matrices(nodes_collect, nodes_serial)

            def node2dof(nodes, ndof):
                return (np.tile(nodes, (ndof, 1)) * ndof + np.arange(ndof).reshape(-1, 1)).ravel("F")

            global_to_local_dofs = node2dof(global_to_local_nodes, num_nodal_dofs)
            self.local_to_global_dofs = node2dof(local_to_global_nodes, num_nodal_dofs)
            self.local_to_global_dofs = (
                np.tile(self.local_to_global_dofs.reshape(-1, 1), (1, size)) * size + np.arange(size)).ravel()
        else:
            global_to_local_dofs = None
        global_to_local_dofs = self.comm.bcast(global_to_local_dofs, root=0)
        self.idx = global_to_local_dofs[local_dof_range[0]:local_dof_range[1]]

    def bcast(self, func, global_values):
        if func.vector.size != global_values.size:
            raise ValueError("Mismatched sizes.")
        func.vector.array = global_values[self.idx]

    def gather(self, func):
        if isinstance(func, Function):
            values_gather = self.comm.gather(func.vector.array, root=0)
        elif isinstance(func, PETSc.Vec):
            values_gather = self.comm.gather(func.array, root=0)
        elif isinstance(func, np.ndarray):
            values_gather = self.comm.gather(func, root=0)
        else:
            raise TypeError("Unsupported func.")

        if self.comm.rank == 0:
            values_collect = np.zeros(self.num_global_dofs * self.size)
            for r, local_values in zip(self.local_dof_range_gather, values_gather):
                values_collect[r[0] * self.size:r[1] * self.size] = local_values
            global_values = values_collect[self.local_to_global_dofs]
        else:
            global_values = None
        return global_values


def compare_matrices(array1, array2, precision=12, k=1):
    kd_tree = cKDTree(array1.round(precision))
    return kd_tree.query(array2.round(precision), k=k)[1]


class Plotter:
    def __init__(self, mesh):
        pyvista.OFF_SCREEN = True
        pyvista.start_xvfb()
        self.dim = mesh.topology.dim
        elements, cell_types, nodes = dolfinx.plot.vtk_mesh(mesh, self.dim)
        self.grid = pyvista.UnstructuredGrid(elements, cell_types, nodes)

    def plot(self, density, threshold=0.49, smooth_iter=100, path=""):
        self.grid.point_data["density"] = np.hstack(density)
        if self.dim == 2:
            grid = self.grid
        else:
            grid = self.grid.threshold(threshold).extract_surface()
        empty_mesh = (self.dim == 3 and grid.n_faces_strict == 0)

        if not empty_mesh:
            if self.dim == 3:
                grid = grid.smooth(n_iter=smooth_iter)
                grid.point_data["density"] = 0.4
            plotter = pyvista.Plotter()
            plotter.background_color = "white"
            lighting = self.dim == 3
            plotter.add_mesh(grid, clim=[0, 1], cmap="Greys", lighting=lighting,
                             show_scalar_bar=False)
            if self.dim == 2:
                plotter.view_xy()
            plotter.screenshot(path + "optimized_design.jpg", window_size=(1000, 1000))
            plotter.close()

    def save(self, density, path):
        self.plot(density, path=path)


def save_xdmf(mesh, rho, path=""):
    xdmf = dolfinx.io.XDMFFile(mesh.comm, path + "optimized_design.xdmf", "w")
    xdmf.write_mesh(mesh)
    rho.name = "density"
    xdmf.write_function(rho)
