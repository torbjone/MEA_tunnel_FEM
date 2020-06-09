import os
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import dolfin as df

eps = 1e-9


sigma = 0.3  # Extracellular conductivity (S/m)

df.parameters['allow_extrapolation'] = False

root_folder = '..'
mesh_folder = join(root_folder, "mesh_nmi_tunnel")
mesh_name = "nmi_mea"
out_folder = join(root_folder, 'results_control')
sim_name = "tunnel_test"
fem_fig_folder = join(root_folder, "fem_figs_control")
[os.makedirs(f, exist_ok=True) for f in [out_folder, fem_fig_folder]]

# example values for validation
source_pos = np.array([[-150, 0, 2.5],
                       [-160, 0, 2.5]])
imem = np.array([[-1.0], [1.0]])
tvec = np.array([0.])
num_tsteps = imem.shape[1]
num_sources = source_pos.shape[0]

nx = 200
ny = 200
nz = 200

dx_tunnel = 5
dz_tunnel = 5
cylinder_radius = 500
x0 = -cylinder_radius / 2 + eps
x1 = cylinder_radius / 2 - eps
y0 = -cylinder_radius / 2 + eps
y1 = cylinder_radius / 2 - eps
z0 = 0 + eps
z1 = cylinder_radius / 4



def analytic_mea(x, y, z):
    phi = 0
    for idx in range(len(imem)):
        r = np.sqrt((x - source_pos[idx, 0])**2 +
                    (y - source_pos[idx, 1])**2 +
                    (z - source_pos[idx, 2])**2)
        phi += imem[idx] / (2 * sigma * np.pi * r)
    return phi


def plot_FEM_results(phi, t_idx):
    """ Plot the set-up, transmembrane currents and electric potential
    """


    x = np.linspace(x0, x1, nx)
    z = np.linspace(z0, z1, nz)
    y = np.linspace(y0, y1, nz)

    mea_x_values = np.zeros(len(x))
    analytic = np.zeros(len(x))
    for idx in range(len(x)):
        mea_x_values[idx] = phi(x[idx], 0, eps)
        analytic[idx] = analytic_mea(x[idx], 0, 1e-9)

    phi_plane_xz = np.zeros((len(x), len(z)))
    phi_plane_xy = np.zeros((len(x), len(z)))
    for x_idx in range(len(x)):
        for z_idx in range(len(z)):
            try:
                phi_plane_xz[x_idx, z_idx] = phi(x[x_idx], 0.0, z[z_idx])
            except RuntimeError:
                phi_plane_xz[x_idx, z_idx] = np.NaN
        for y_idx in range(len(y)):
            try:
                phi_plane_xy[x_idx, y_idx] = phi(x[x_idx], y[y_idx], 0.0 + eps)
            except RuntimeError:
                phi_plane_xy[x_idx, y_idx] = np.NaN
    plt.close("all")
    fig = plt.figure(figsize=[18, 9])
    fig.subplots_adjust(hspace=0.9, bottom=0.07, top=0.97, left=0.2)

    ax_setup = fig.add_subplot(511, aspect=1, xlabel='x [$\mu$m]', ylabel='z [$\mu$m]',
                          title='Axon (green) and tunnel (gray)', xlim=[x0 - 5, x1 + 5], ylim=[z0 - 5, z1 + 5])

    axon_center_idx = np.argmin(np.abs(source_pos[:, 0] - 0))

    imem_max = np.max(np.abs(imem))
    ax_imem_temporal = fig.add_axes([0.05, 0.8, 0.08, 0.1], xlabel='Time [ms]', ylabel='nA',
                                    xlim=[0, tvec[-1]], ylim=[-imem_max, imem_max],
                          title='Transmembrane currents\n(x=0)')

    ax_imem_spatial = fig.add_subplot(512, xlabel=r'x [$\mu$m]', ylabel='nA',
                                      ylim=[-imem_max - 1, imem_max + 1],
                          title='Transmembrane currents across axon', xlim=[x0 - 5, x1 + 5])

    ax1 = fig.add_subplot(513, aspect=1, xlabel=r'x [$\mu$m]', ylabel=r'y [$\mu$m]',
                          title='Potential cross section (z=0)')

    ax2 = fig.add_subplot(514, aspect=1, xlabel=r'x [$\mu$m]', ylabel=r'z [$\mu$m]',
                          title='Potential cross section (y=0)')

    ax3 = fig.add_subplot(515, xlabel=r'x [$\mu$m]', ylabel='MEA potential (mV)',
                          xlim=[x0 - 5, x1 + 5])

    #  Draw set up with tunnel and axon
    rect = mpatches.Rectangle([x0, z0], dx_tunnel, dz_tunnel, ec="k", fc='0.8')
    ax_setup.add_patch(rect)

    ax_setup.plot(source_pos[:, 0], source_pos[:, 2], c='g', lw=2)
    ax_imem_temporal.plot(tvec, imem[axon_center_idx, :])
    ax_imem_temporal.axvline(tvec[t_idx], c='gray', ls="--")

    ax_imem_spatial.plot(source_pos[:, 0], imem[:, t_idx])

    img1 = ax1.imshow(phi_plane_xy.T, interpolation='nearest', origin='lower', cmap='bwr',
                      extent=(x[0], x[-1], y[0], y[-1]))
    img2 = ax2.imshow(phi_plane_xz.T, interpolation='nearest', origin='lower', cmap='bwr',
                      extent=(x[0], x[-1], z[0], z[-1]))

    cax = fig.add_axes([0.95, 0.5, 0.01, 0.1])

    plt.colorbar(img1, cax=cax, label="mV")
    l, = ax3.plot(x, mea_x_values,  lw=2, c='k')
    la, = ax3.plot(x, analytic,  lw=1, c='r', ls="--")
    fig.legend([l, la], ["FEM", "Analytic semi-infinite"], frameon=False)
    plt.savefig(join(fem_fig_folder, 'results_{}_t_idx_{}.png'.format(sim_name, t_idx)))




# Define mesh
mesh = df.Mesh(join(mesh_folder, "{}.xml".format(mesh_name)))
subdomains = df.MeshFunction("size_t", mesh, join(mesh_folder, "{}_physical_region.xml".format(mesh_name)))
boundaries = df.MeshFunction("size_t", mesh, join(mesh_folder, "{}_facet_region.xml".format(mesh_name)))


print("Number of cells in mesh: ", mesh.num_cells())
# mesh = refine_mesh(mesh)

np.save(join(out_folder, "mesh_coordinates.npy"), mesh.coordinates())

sigma = df.Constant(sigma)

V = df.FunctionSpace(mesh, "CG", 2)
v = df.TestFunction(V)
u = df.TrialFunction(V)


ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
dx = df.Measure("dx", domain=mesh, subdomain_data=subdomains)


a = df.inner(sigma * df.grad(u), df.grad(v)) * dx(1)
# Define function space and basis functions

# This corresponds to Neumann boundary conditions zero, i.e. all outer boundaries are insulating.
L = df.Constant(0) * v * dx

# Define Dirichlet boundary conditions at left and right boundaries
bcs = [df.DirichletBC(V, 0.0, boundaries, 1)]


for t_idx in range(num_tsteps):

    print("Time step {} of {}".format(t_idx, num_tsteps))
    phi = df.Function(V)
    A = df.assemble(a)
    b = df.assemble(L)

    [bc.apply(A, b) for bc in bcs]

    # Adding point sources from neural simulation
    for s_idx, s_pos in enumerate(source_pos):

        point = df.Point(s_pos[0], s_pos[1], s_pos[2])
        delta = df.PointSource(V, point, imem[s_idx, t_idx])
        delta.apply(b)

    df.solve(A, phi.vector(), b, 'cg', "ilu")

    # df.File(join(out_folder, "phi_t_vec_{}.xml".format(t_idx))) << phi
    np.save(join(out_folder, "phi_t_vec_{}_out_of_tunnel.npy".format(t_idx)), phi.vector())

    plot_FEM_results(phi, t_idx)


