import os
import sys
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import dolfin as df
from plotting_convention import mark_subplots, simplify_axes


eps = 1e-9
sigma = 0.3  # Extracellular conductivity (S/m)

df.parameters['allow_extrapolation'] = False

root_folder = '..'
mesh_folder = join(root_folder, "mesh_nmi_tunnel")
neural_sim_folder = join(root_folder, "neural_sim_results")
mesh_name = "nmi_mea"
out_folder = join(root_folder, 'results_highres')
sim_name = "tunnel_test"
fem_fig_folder = join(root_folder, "fem_figs_control")
[os.makedirs(f, exist_ok=True) for f in [out_folder, fem_fig_folder]]

# example values for validation
# source_pos = np.array([[-150, 0, 2.5],
#                        [-160, 0, 2.5]])
# imem = np.array([[-1.0], [1.0]])
# tvec = np.array([0.])
source_pos = np.load(join(neural_sim_folder, "source_pos.npy"))
# print(source_pos)

imem = np.load(join(neural_sim_folder, "axon_imem.npy"))
vmem = np.load(join(neural_sim_folder, "axon_vmem.npy"))
tvec = np.load(join(neural_sim_folder, "axon_tvec.npy"))
num_tsteps = imem.shape[1]
num_sources = source_pos.shape[0]

cell_plot_positions = [[-200, 0, 100],
                       [-150, 0, 2.5],
                       [0, 0, 2.5],
                       #[150, 0, 2.5]
                       ]

cell_plot_idxs = []
for p_idx in range(len(cell_plot_positions)):
    cell_plot_idxs.append(np.argmin(np.sum((cell_plot_positions[p_idx] - source_pos)**2, axis=1)))

plot_idx_clrs = lambda idx: plt.cm.viridis(idx / (len(cell_plot_idxs) - 1))

nx = 200
ny = 200
nz = 200

dx_tunnel = 5
dz_tunnel = 5
cylinder_radius = 500
tunnel_radius = 5
structure_radius = 100
structure_height = 55

x0 = -cylinder_radius / 2
x1 = cylinder_radius / 2
y0 = -cylinder_radius / 2
y1 = cylinder_radius / 2
z0 = 0 + eps
z1 = cylinder_radius / 4


def analytic_mea(x, y, z, t_idx):
    phi = 0
    for idx in range(len(imem)):
        r = np.sqrt((x - source_pos[idx, 0])**2 +
                    (y - source_pos[idx, 1])**2 +
                    (z - source_pos[idx, 2])**2)
        phi += imem[idx, t_idx] / (2 * sigma * np.pi * r)
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
        analytic[idx] = analytic_mea(x[idx], 0, 1e-9, t_idx)

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

    np.save(join(out_folder, "phi_xz_t_vec_{}.npy".format(t_idx)), phi_plane_xz)
    np.save(join(out_folder, "phi_xy_t_vec_{}.npy".format(t_idx)), phi_plane_xy)
    np.save(join(out_folder, "phi_mea_t_vec_{}.npy".format(t_idx)), mea_x_values)
    np.save(join(out_folder, "phi_mea_analytic_t_vec_{}.npy".format(t_idx)), analytic)


    plt.close("all")
    fig = plt.figure(figsize=[6, 9])
    fig.subplots_adjust(hspace=0.9, bottom=0.12, top=0.97, left=0.12, wspace=0.4)


    imem_max = np.max(np.abs(imem))
    ax_vmem = fig.add_subplot(523, xlabel='time [ms]', ylabel='mV',
                                    xlim=[0, tvec[-1]], ylim=[-110, 30],
                          title='membrane potentials')

    ax_imem = fig.add_subplot(524, xlabel='time [ms]', ylabel='nA',
                                    xlim=[0, tvec[-1]], ylim=[-imem_max, imem_max],
                          title='transmembrane currents')

    ax_setup = fig.add_subplot(511, aspect=1, xlabel='x [$\mu$m]', ylabel='z [$\mu$m]',
                          title='set-up',
                               xlim=[x0 - 5, x1 + 5], ylim=[z0 - 5, z1 + 5])

    ax_xz = fig.add_subplot(513, aspect=1, xlabel=r'x [$\mu$m]', ylabel=r'z [$\mu$m]',
                          title='Potential cross section (y=0)')

    ax_xy = fig.add_subplot(514, aspect=1, xlabel=r'x [$\mu$m]', ylabel=r'y [$\mu$m]',
                          title='Potential cross section (z=0)')

    ax_mea = fig.add_subplot(515, xlabel=r'x [$\mu$m]', ylabel='mV',
                           xlim=[x0 - 5, x1 + 5], ylim=[-1, 1])

    #  Draw set up with tunnel and axon
    rect = mpatches.Rectangle([-structure_radius, tunnel_radius],
                              2 * structure_radius, structure_height, ec="k", fc='0.8')
    ax_setup.add_patch(rect)


    ax_setup.plot(source_pos[:, 0], source_pos[:, 2], c='g', lw=2)
    for counter, p_idx in enumerate(cell_plot_idxs):
        ax_setup.plot(source_pos[p_idx, 0], source_pos[p_idx, 2],
                      c=plot_idx_clrs(counter), marker='o')

        ax_imem.plot(tvec, imem[p_idx, :], c=plot_idx_clrs(counter))
        ax_vmem.plot(tvec, vmem[p_idx, :], c=plot_idx_clrs(counter))

    ax_imem.axvline(tvec[t_idx], c='gray', ls="--")
    ax_vmem.axvline(tvec[t_idx], c='gray', ls="--")
    # ax_imem_spatial.plot(source_pos[:, 0], imem[:, t_idx])

    xy_masked = np.ma.masked_where(np.isnan(phi_plane_xy.T), phi_plane_xy.T)
    xz_masked = np.ma.masked_where(np.isnan(phi_plane_xz.T), phi_plane_xz.T)

    vmax = 1
    from matplotlib.colors import SymLogNorm
    img1 = ax_xy.imshow(xy_masked, interpolation='nearest', origin='lower', cmap='bwr',
                      extent=(x[0], x[-1], y[0], y[-1]), norm=SymLogNorm(0.01, vmax=1, vmin=-1))
                        #vmin=-vmax, vmax=vmax)
    img2 = ax_xz.imshow(xz_masked, interpolation='nearest', origin='lower', cmap='bwr',
                      extent=(x[0], x[-1], z[0], z[-1]), norm=SymLogNorm(0.01, vmax=1, vmin=-1))
                        #vmin=-vmax, vmax=vmax)

    cax = fig.add_axes([0.85, 0.25, 0.01, 0.15])

    plt.colorbar(img1, cax=cax, label="mV")
    l, = ax_mea.plot(x, mea_x_values,  lw=2, c='k')
    la, = ax_mea.plot(x, analytic,  lw=1, c='r', ls="--")
    fig.legend([l, la], ["FEM", "Analytic semi-infinite"], frameon=False, loc="lower right")
    plt.savefig(join(fem_fig_folder, 'results_{}_t_idx_{:04d}.png'.format(sim_name, t_idx)))


def simulate_FEM():

    # Define mesh
    mesh = df.Mesh(join(mesh_folder, "{}.xml".format(mesh_name)))
    subdomains = df.MeshFunction("size_t", mesh, join(mesh_folder, "{}_physical_region.xml".format(mesh_name)))
    boundaries = df.MeshFunction("size_t", mesh, join(mesh_folder, "{}_facet_region.xml".format(mesh_name)))


    print("Number of cells in mesh: ", mesh.num_cells())
    # mesh = refine_mesh(mesh)

    np.save(join(out_folder, "mesh_coordinates.npy"), mesh.coordinates())

    sigma_vec = df.Constant(sigma)

    V = df.FunctionSpace(mesh, "CG", 2)
    v = df.TestFunction(V)
    u = df.TrialFunction(V)

    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    dx = df.Measure("dx", domain=mesh, subdomain_data=subdomains)


    a = df.inner(sigma_vec * df.grad(u), df.grad(v)) * dx(1)
    # Define function space and basis functions

    # This corresponds to Neumann boundary conditions zero, i.e. all outer boundaries are insulating.
    L = df.Constant(0) * v * dx

    # Define Dirichlet boundary conditions outer cylinder boundaries
    bcs = [df.DirichletBC(V, 0.0, boundaries, 1)]


    for t_idx in range(num_tsteps):

        f_name = join(out_folder, "phi_xz_t_vec_{}.npy".format(t_idx))
        if os.path.isfile(f_name):
            print("skipping ", f_name)
            continue

        # if not divmod(t_idx, SIZE)[1] == RANK:
        #     continue
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
        # np.save(join(out_folder, "phi_t_vec_{}.npy".format(t_idx)), phi.vector())

        plot_FEM_results(phi, t_idx)


def reconstruct_MEA_times_from_FEM():
    x = np.linspace(x0, x1, nx)
    z = np.linspace(z0, z1, nz)
    y = np.linspace(y0, y1, nz)

    mea_x_plot_pos = np.array([np.argmin(np.abs(x - x_)) for x_ in [-150, 0]])

    # print(mea_x_plot_pos, x[mea_x_plot_pos])

    mea_analytic = np.zeros((len(mea_x_plot_pos), num_tsteps))
    mea_fem = np.zeros((len(mea_x_plot_pos), num_tsteps))
    phi_plane_xz = np.zeros((len(x), len(z), len(tvec)))
    for t_idx in range(num_tsteps):
        phi_plane_xz[:, :, t_idx] = np.load(join(out_folder, "phi_xz_t_vec_{}.npy".format(t_idx)))
        # phi_plane_xy_ = np.load(join(out_folder, "phi_xy_t_vec_{}.npy".format(t_idx)))
        mea_fem[:, t_idx] = np.load(join(out_folder, "phi_mea_t_vec_{}.npy".format(t_idx)))[mea_x_plot_pos]
        mea_analytic[:, t_idx] = np.load(join(out_folder, "phi_mea_analytic_t_vec_{}.npy".format(t_idx)))[mea_x_plot_pos]

    plt.close("all")
    fig = plt.figure(figsize=[9, 6])
    fig.subplots_adjust(hspace=0.4, bottom=0.12, top=0.92, left=0.12, wspace=0.6, right=0.96)

    ax_setup = fig.add_axes([0.12, 0.4, 0.86, 0.6], aspect=1, xlabel='x [$\mu$m]', ylabel='z [$\mu$m]',
                            xlim=[x0 - 5, x1 + 5], ylim=[z0 - 45, z1 + 5])

    ax_vmem = fig.add_subplot(337, xlabel='time [ms]', ylabel='membrane\npotential (mV)',
                                    xlim=[0, tvec[-1]], ylim=[-110, 40],
                          # title='membrane potentials'
                              )

    ax_mea_free = fig.add_subplot(338, xlabel=r'time [ms]', ylabel='$\phi$ (mV)',
                                  xlim=[0, tvec[-1]], ylim=[-0.015, 0.015],
                            )
    ax_mea_tunnel = fig.add_subplot(339, xlabel=r'time [ms]', ylabel='$\phi$ (mV)',
                                    xlim=[0, tvec[-1]], ylim=[-1.5, 1.5],
                          )

    ax_mea_free.set_title("I", color="orange")
    ax_mea_tunnel.set_title("II", color="orange")

    ax_setup.text(-195, 105, "soma", color="gray", va='bottom')
    ax_setup.text(-200, 75, "axon", color="gray", va='top', rotation=-90)
    ax_setup.text(95, 55, "tunnel lid", ha='right', va='top')
    ax_setup.text(247, -3, "electrode plate", ha='right', va='top')
    #  Draw set up with tunnel and axon
    rect = mpatches.Rectangle([-structure_radius, tunnel_radius],
                              2 * structure_radius, structure_height, ec="k", fc='0.8')
    ax_setup.add_patch(rect)

    rect_bottom = mpatches.Rectangle([-1000, 0],
                              2000, -1000, ec="k", fc='0.7', linewidth=0.5)
    ax_setup.add_patch(rect_bottom)

    ax_setup.plot(source_pos[:, 0], source_pos[:, 2], c='gray', lw=1)
    ax_setup.plot(source_pos[0, 0], source_pos[0, 2], c='gray', marker='^', ms=17)
    for counter, p_idx in enumerate(cell_plot_idxs):
        ax_setup.plot(source_pos[p_idx, 0], source_pos[p_idx, 2],
                      c=plot_idx_clrs(counter), marker='o')
        if p_idx > 0:

        # for mea_idx in mea_x_plot_pos:
            rect_elec = mpatches.Rectangle([source_pos[p_idx, 0] - 5, 0],
                                              10, -5, ec="k", fc='orange', linewidth=0.5)
            ax_setup.add_patch(rect_elec)
            # ax_setup.arrow(source_pos[p_idx, 0], -5, 100, -70, lw=2,
            #                clip_on=False, color='orange', head_width=5)
            ax_setup.text(source_pos[p_idx, 0], - 6,
                          ["I", "II"][counter - 1], color='orange', va='top', ha="center")

    for counter, p_idx in enumerate(cell_plot_idxs):
        ax_vmem.plot(tvec, vmem[p_idx, :], c=plot_idx_clrs(counter), lw=2)

    # xy_masked = np.ma.masked_where(np.isnan(phi_plane_xy.T), phi_plane_xy.T)
    xz_masked = np.ma.masked_where(np.isnan(phi_plane_xz), phi_plane_xz)
    from matplotlib.colors import SymLogNorm, LogNorm
    img1 = ax_setup.imshow(xz_masked[:, :, 0].T, interpolation='nearest', origin='lower', cmap='bwr',
                      extent=(x[0], x[-1], z[0], z[-1]), norm=SymLogNorm(0.01, vmax=1, vmin=-1))
    vmax = 1

    cax = fig.add_axes([0.83, 0.67, 0.01, 0.2])

    cbar = plt.colorbar(img1, cax=cax, label="$\phi$ (mV)")
    l, = ax_mea_free.plot(tvec, mea_fem[0],  lw=2, c='k')
    la, = ax_mea_tunnel.plot(tvec, mea_fem[1],  lw=2, c='k', ls="-")
    # fig.legend([l, la], ["FEM", "Analytic semi-infinite"], frameon=False, loc="lower right")

    t1 = ax_vmem.axvline(tvec[0], c='gray', ls="--")
    t2 = ax_mea_free.axvline(tvec[0], c='gray', ls="--")
    t3 = ax_mea_tunnel.axvline(tvec[0], c='gray', ls="--")


    mark_subplots(ax_setup, "A", ypos=1.05, xpos=0.0)
    simplify_axes([ax_setup, ax_mea_free, ax_mea_tunnel, ax_vmem])
    mark_subplots([ax_vmem, ax_mea_free, ax_mea_tunnel], "BCD", xpos=0.0)

    for t_idx in range(num_tsteps):
        img1.set_data(xz_masked[:, :, t_idx].T)
                        #vmin=-vmax, vmax=vmax)
        t1.set_xdata(tvec[t_idx])
        t2.set_xdata(tvec[t_idx])
        t3.set_xdata(tvec[t_idx])

        plt.savefig(join(fem_fig_folder, 'anim_results_{}_t_idx_{:04d}.png'.format(sim_name, t_idx)))

    img1.set_data([[]])
    cax.clear()

    img1 = ax_setup.imshow(np.max(np.abs(xz_masked), axis=-1).T, interpolation='nearest', origin='lower', cmap='Reds',
                      extent=(x[0], x[-1], z[0], z[-1]), norm=LogNorm(0.002, vmax=1))

    cbar = plt.colorbar(img1, cax=cax, label="$\phi$ (mV)")

    plt.savefig(join(fem_fig_folder, 'max_results_{}.png'.format(sim_name)))

if __name__ == '__main__':
    reconstruct_MEA_times_from_FEM()