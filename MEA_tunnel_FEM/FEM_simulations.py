import os
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
matplotlib.rc('pdf', fonttype=42)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from plotting_convention import mark_subplots, simplify_axes


plt.rcParams.update({
    'xtick.labelsize': 6,
    'xtick.major.size': 2,
    'ytick.labelsize': 6,
    'ytick.major.size': 2,
    'font.size': 7,
    'font.family': "Arial",
    'axes.labelsize': 6,
    'axes.titlesize': 7,
    'legend.fontsize': 6,
    'figure.subplot.wspace': 0.4,
    'figure.subplot.hspace': 0.4,
    'figure.subplot.left': 0.1,
})


eps = 1e-9
sigma = 1.46  # Extracellular conductivity (S/m)

root_folder = '..'
mesh_folder = join(root_folder, "mesh_nmi_tunnel")
neural_sim_folder = join(root_folder, "neural_sim_results")
mesh_name = "nmi_mea"
out_folder = join(root_folder, 'results')
sim_name = "nmi_tunnel"
fem_fig_folder = join(root_folder, "fem_figs_control")
[os.makedirs(f, exist_ok=True) for f in [out_folder, fem_fig_folder]]

# example values for validation
# source_pos = np.array([[-150, 0, 2.5],
#                        [-160, 0, 2.5]])
# imem = np.array([[-1.0], [1.0]])
# tvec = np.array([0.])
source_pos = np.load(join(neural_sim_folder, "source_pos.npy"))
source_line_pos = np.load(join(neural_sim_folder, "source_line_pos.npy"))

imem = np.load(join(neural_sim_folder, "axon_imem.npy"))
vmem = np.load(join(neural_sim_folder, "axon_vmem.npy"))
tvec = np.load(join(neural_sim_folder, "axon_tvec.npy"))
num_tsteps = imem.shape[1]
num_sources = source_pos.shape[0]

mea_x_positions = [-180, 0]

cell_plot_positions = [[-200, 0, 65],
                       [mea_x_positions[0], 0, 2.5],
                       [mea_x_positions[1], 0, 2.5],
                       #[150, 0, 2.5]
                       ]
cell_plot_idxs = []
for p_idx in range(len(cell_plot_positions)):
    cell_plot_idxs.append(np.argmin(np.sum((cell_plot_positions[p_idx] -
                                            source_pos)**2, axis=1)))

plot_idx_clrs = ["#6d319c", "#1f8f8b", "#cb089c"]

nx = 200
ny = 200
nz = 200

dx_tunnel = 5
dz_tunnel = 5
cylinder_radius = 1000
tunnel_radius = 5
structure_radius = 125
structure_height = 25

x0 = -cylinder_radius / 4
x1 = cylinder_radius / 4
y0 = -cylinder_radius / 4
y1 = cylinder_radius / 4
z0 = 0 + eps
z1 = cylinder_radius / 8


def analytic_mea(x, y, z, t_idx):
    """
    Calculate extracellular potential analytically, through method of images
    """
    phi = 0
    for idx in range(len(imem)):
        r = np.sqrt((x - source_pos[idx, 0])**2 +
                    (y - source_pos[idx, 1])**2 +
                    (z - source_pos[idx, 2])**2)
        phi += imem[idx, t_idx] / (2 * sigma * np.pi * r)
    return phi


def plot_and_save_simulation_results(phi, t_idx):
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
    moi_plane_xy = np.zeros((len(x), len(z)))
    for x_idx in range(len(x)):
        for z_idx in range(len(z)):
            try:
                phi_plane_xz[x_idx, z_idx] = phi(x[x_idx], 0.0, z[z_idx])
            except RuntimeError:
                phi_plane_xz[x_idx, z_idx] = np.NaN
        for y_idx in range(len(y)):
            try:
                phi_plane_xy[x_idx, y_idx] = phi(x[x_idx], y[y_idx], 0.0 + eps)
                moi_plane_xy[x_idx, y_idx] =  analytic_mea(x[x_idx], y[y_idx],
                                                           0.0 + eps, t_idx)
            except RuntimeError:
                phi_plane_xy[x_idx, y_idx] = np.NaN
                moi_plane_xy[x_idx, y_idx] = np.NaN

    np.save(join(out_folder, "phi_xz_t_vec_{}.npy".format(t_idx)), phi_plane_xz)
    np.save(join(out_folder, "phi_xy_t_vec_{}.npy".format(t_idx)), phi_plane_xy)
    np.save(join(out_folder, "moi_xy_t_vec_{}.npy".format(t_idx)), moi_plane_xy)
    np.save(join(out_folder, "phi_mea_t_vec_{}.npy".format(t_idx)), mea_x_values)
    np.save(join(out_folder, "phi_mea_analytic_t_vec_{}.npy".format(t_idx)), analytic)


    plt.close("all")
    fig = plt.figure(figsize=[6, 9])
    fig.subplots_adjust(hspace=0.9, bottom=0.12, top=0.97, left=0.12, wspace=0.4)


    imem_max = np.max(np.abs(imem[cell_plot_idxs, :]))
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
                      c=plot_idx_clrs[counter], marker='o')

        ax_imem.plot(tvec, imem[p_idx, :], c=plot_idx_clrs[counter])
        ax_vmem.plot(tvec, vmem[p_idx, :], c=plot_idx_clrs[counter])

    ax_imem.axvline(tvec[t_idx], c='gray', ls="--")
    ax_vmem.axvline(tvec[t_idx], c='gray', ls="--")
    # ax_imem_spatial.plot(source_pos[:, 0], imem[:, t_idx])

    xy_masked = np.ma.masked_where(np.isnan(phi_plane_xy.T), phi_plane_xy.T)
    xz_masked = np.ma.masked_where(np.isnan(phi_plane_xz.T), phi_plane_xz.T)

    vmax = 1
    from matplotlib.colors import SymLogNorm
    img1 = ax_xy.imshow(xy_masked, interpolation='nearest',
                        origin='lower', cmap='bwr',
                      extent=(x[0], x[-1], y[0], y[-1]),
                        norm=SymLogNorm(0.01, vmax=1, vmin=-1))
                        #vmin=-vmax, vmax=vmax)
    img2 = ax_xz.imshow(xz_masked, interpolation='nearest', origin='lower',
                        cmap='bwr', extent=(x[0], x[-1], z[0], z[-1]),
                        norm=SymLogNorm(0.01, vmax=1, vmin=-1))
                        #vmin=-vmax, vmax=vmax)

    cax = fig.add_axes([0.85, 0.25, 0.01, 0.15])

    plt.colorbar(img1, cax=cax, label="mV")
    l, = ax_mea.plot(x, mea_x_values,  lw=2, c='k')
    la, = ax_mea.plot(x, analytic,  lw=1, c='r', ls="--")
    fig.legend([l, la], ["FEM", "Analytic semi-infinite"], frameon=False,
               loc="lower right")
    plt.savefig(join(fem_fig_folder, 'results_{}_t_idx_{:04d}.png'.format(
        sim_name, t_idx)))


def simulate_FEM():

    import dolfin as df
    df.parameters['allow_extrapolation'] = False

    # Define mesh
    mesh = df.Mesh(join(mesh_folder, "{}.xml".format(mesh_name)))
    subdomains = df.MeshFunction("size_t", mesh, join(mesh_folder,
                          "{}_physical_region.xml".format(mesh_name)))
    boundaries = df.MeshFunction("size_t", mesh, join(mesh_folder,
                          "{}_facet_region.xml".format(mesh_name)))

    print("Number of cells in mesh: ", mesh.num_cells())

    np.save(join(out_folder, "mesh_coordinates.npy"), mesh.coordinates())

    sigma_vec = df.Constant(sigma)

    V = df.FunctionSpace(mesh, "CG", 2)
    v = df.TestFunction(V)
    u = df.TrialFunction(V)

    ds = df.Measure("ds", domain=mesh, subdomain_data=boundaries)
    dx = df.Measure("dx", domain=mesh, subdomain_data=subdomains)

    a = df.inner(sigma_vec * df.grad(u), df.grad(v)) * dx(1)

    # This corresponds to Neumann boundary conditions zero, i.e.
    # all outer boundaries are insulating.
    L = df.Constant(0) * v * dx

    # Define Dirichlet boundary conditions outer cylinder boundaries (ground)
    bcs = [df.DirichletBC(V, 0.0, boundaries, 1)]

    for t_idx in range(num_tsteps):

        f_name = join(out_folder, "phi_xz_t_vec_{}.npy".format(t_idx))
        # if os.path.isfile(f_name):
        #     print("skipping ", f_name)
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

        plot_and_save_simulation_results(phi, t_idx)


def plot_soma_EAP_amp_with_distance():
    x = np.linspace(x0, x1, nx)
    z = np.linspace(z0, z1, nz)
    phi_plane_xz = np.zeros((len(x), len(z), len(tvec)))
    for t_idx in range(num_tsteps):
        phi_plane_xz[:, :, t_idx] = np.load(join(out_folder,
                                         "phi_xz_t_vec_{}.npy".format(t_idx)))

    soma_height = 65
    soma_diam = 10
    soma_xpos = -200
    z_idx = np.argmin(np.abs(z - soma_height))
    x_idxs = np.where(x > soma_xpos + soma_diam)

    peak_to_peaks = np.zeros(len(x))
    for x_idx in range(len(x)):
        peak_to_peaks[x_idx] = np.max(phi_plane_xz[x_idx, z_idx]) - \
                               np.min(phi_plane_xz[x_idx, z_idx])
        plt.plot(phi_plane_xz[x_idx, z_idx])
    plt.savefig(join(root_folder, "all_spikes_x_dir.png"))

    plt.close("all")
    fig = plt.figure(figsize=[5, 5])
    fig.subplots_adjust(left=0.15)
    ax1 = fig.add_subplot(111, xlabel="distance from soma ($\mu$m)",
                          ylabel="$\mu$V", xlim=[0, 70])
    plt.plot(x[x_idxs] - soma_xpos, peak_to_peaks[x_idxs] * 1000)
    plt.savefig(join(root_folder, "peak_to_peak_amp_with_distance.pdf"))


def make_results_figure():
    x = np.linspace(x0, x1, nx)
    z = np.linspace(z0, z1, nz)

    mea_x_plot_pos = np.array([np.argmin(np.abs(x - x_))
                               for x_ in mea_x_positions])

    # print(mea_x_plot_pos, x[mea_x_plot_pos])

    mea_analytic = np.zeros((len(mea_x_plot_pos), num_tsteps))
    mea_fem = np.zeros((len(mea_x_plot_pos), num_tsteps))
    phi_plane_xz = np.zeros((len(x), len(z), len(tvec)))
    for t_idx in range(num_tsteps):
        phi_plane_xz[:, :, t_idx] = 1000 * np.load(join(out_folder,
                                        "phi_xz_t_vec_{}.npy".format(t_idx)))
        # phi_plane_xy_ = np.load(join(out_folder, "phi_xy_t_vec_{}.npy".format(t_idx)))
        mea_fem[:, t_idx] = 1000 * np.load(join(out_folder,
                    "phi_mea_t_vec_{}.npy".format(t_idx)))[mea_x_plot_pos]
        mea_analytic[:, t_idx] = 1000 * np.load(join(out_folder,
             "phi_mea_analytic_t_vec_{}.npy".format(t_idx)))[mea_x_plot_pos]

    noise_level = 10  # uV
    soma_height = 65
    soma_diam = 8
    soma_xpos = -200
    z_idx = np.argmin(np.abs(z - soma_height))
    x_idxs = (-150 + soma_diam > x) & (x > soma_xpos + soma_diam / 2)

    eap_amp = np.zeros(len(x))
    for x_idx in range(len(x)):
        eap_amp[x_idx] = np.max(np.abs(phi_plane_xz[x_idx, z_idx]))

    plt.close("all")
    fig = plt.figure(figsize=[117 * 0.03937, 48 * 0.03937])
    # fig = plt.figure(figsize=[117 * 0.03937 * 5, 48 * 0.03937 * 5])
    fig.subplots_adjust(hspace=0.45, bottom=0.17, top=0.99,
                        left=0.16, wspace=1.0, right=0.96)

    ax_h = 0.25
    ax_w = 0.11
    ax_left = 0.10

    ax_setup = fig.add_axes([0.28, 0.13, 0.69, 0.85], #aspect=1,
                            xlim=[-240, 145], ylim=[-15, 117])

    ax_vmem = fig.add_axes([ax_left, 0.13 + 2 * (ax_h + 0.04), ax_w, ax_h],
                           xlim=[0, tvec[-1]], ylim=[-110, 40])

    ax_mea_free = fig.add_axes([ax_left, 0.13 + ax_h + 0.04, ax_w, ax_h],
                               xlim=[0, tvec[-1]], ylim=[-8, 4])
    ax_mea_tunnel = fig.add_axes([ax_left, 0.13, ax_w, ax_h], xlim=[0, tvec[-1]],
                                    ylim=[-600, 400])
    ax_EAP_decay = fig.add_axes([0.57, 0.70, 0.08, 0.17],
                                xlim=[0, 50], facecolor='none')


    ax_setup.set_xlabel('X (µm)', labelpad=0)
    ax_setup.set_ylabel('Z (µm)', labelpad=-1)

    ax_vmem.set_ylabel('Membrane\npotential (mV)', labelpad=0)
    ax_vmem.set_xticklabels(["", ""])
    # ax_vmem.set_xlabel('Time (ms)', labelpad=-0.1)

    ax_mea_free.set_ylabel('OME (µV)', labelpad=9)
    ax_mea_free.set_xticklabels(["", ""])

    # ax_mea_free.set_xlabel('Time (ms)', labelpad=-0.1)

    ax_mea_tunnel.set_ylabel('CME (µV)', labelpad=2)
    ax_mea_tunnel.set_xlabel('Time (ms)', labelpad=-0.1)

    ax_EAP_decay.set_ylabel('Peak\namplitude (µV)', labelpad=1)
    ax_EAP_decay.set_xlabel('Distance from\nsoma (µm)', labelpad=-0.0)

    # ax_mea_free.set_title("OµE", pad=-5)
    # ax_mea_tunnel.set_title("CµE", pad=-5)
    ax_setup.text(139, -7, "Substrate", ha='right', va='top')

    dist_from_soma = x[x_idxs] - soma_xpos

    ax_EAP_decay.plot(dist_from_soma, eap_amp[x_idxs], lw=1, c='b')

    ax_setup.plot(x[x_idxs], np.ones(len(x[x_idxs])) * soma_height,
                  ls=':', c='blue', lw=1)

    noise_level_dist_idx = np.argmin(np.abs(eap_amp[x_idxs] - noise_level))
    noise_level_dist = dist_from_soma[noise_level_dist_idx]
    ax_EAP_decay.axhline(noise_level, ls='--', c='gray', lw=0.5)
    ax_EAP_decay.axvline(noise_level_dist, ls='--', c='gray', lw=0.5)
    # ax_EAP_decay.text(noise_level_dist + 5, noise_level + 5,
    #                   "{:1.1f}\n µm".format(noise_level_dist), fontsize=6)

    #  Draw set up with tunnel and axon
    rect = mpatches.Rectangle([-structure_radius - 2, tunnel_radius],
                              2 * structure_radius + 4, structure_height,
                              ec="k", fc='0.8', linewidth=0.3)
    ax_setup.add_patch(rect)

    rect_bottom = mpatches.Rectangle([-1000, 0],
                              2000, -1000, ec="k", fc='0.7', linewidth=0.3)
    ax_setup.add_patch(rect_bottom)

    for source_idx in range(len(source_line_pos)):
        xstart, xend = source_line_pos[source_idx, :, 0]
        zstart, zend = source_line_pos[source_idx, :, 2]

        ax_setup.plot([xstart, xend], [zstart, zend], c='#18e10c', lw=1,
                      solid_capstyle="round", solid_joinstyle="round")
    ax_setup.plot(source_pos[0, 0], source_pos[0, 2], c='#18e10c', marker='o',
                  ms=14)

    for counter, p_idx in enumerate(cell_plot_idxs):
        ax_setup.plot(source_pos[p_idx, 0], source_pos[p_idx, 2],
                      c=plot_idx_clrs[counter], marker='o', ms=5)
        if p_idx > 0:

            rect_elec = mpatches.Rectangle([source_pos[p_idx, 0] - 5, 0],
                                              10, -5, ec="k", fc='k',
                                           linewidth=0.5)
            ax_setup.add_patch(rect_elec)
            ax_setup.text(source_pos[p_idx, 0], - 6,
                          ["OME", "CME"][counter - 1], va='top', ha="center")

    for counter, p_idx in enumerate(cell_plot_idxs):
        ax_vmem.plot(tvec, vmem[p_idx, :], c=plot_idx_clrs[counter], lw=1)

    num = 11
    levels = np.logspace(-2., 0, num=num)
    scale_max = 500

    levels_norm = scale_max * np.concatenate((-levels[::-1], levels))
    bwr_cmap = plt.cm.get_cmap('bwr')  # rainbow, spectral, RdYlBu

    colors_from_map = [bwr_cmap(i * np.int(255 / (len(levels_norm) - 2)))
                       for i in range(len(levels_norm) - 1)]
    colors_from_map[num - 1] = (1.0, 1.0, 1.0, 1.0)

    xz_masked = np.ma.masked_where(np.isnan(phi_plane_xz), phi_plane_xz)

    ep_intervals = ax_setup.contourf(x, z, xz_masked[:, :, 0],
                                zorder=-2, colors=colors_from_map,
                                levels=levels_norm, extend='both')

    ep_intervals_ = ax_setup.contour(x, z, xz_masked[:, :, 0].T, colors='k',
                                     linewidths=(0.3), zorder=-2,
                                     levels=levels_norm)

    cax = fig.add_axes([0.85, 0.5, 0.01, 0.35])

    cbar = plt.colorbar(ep_intervals, cax=cax, label="$\phi$ (µV)")
    cbar.set_ticks(np.array([-1, -0.1, -0.02, 0.02, 0.1, 1]) * scale_max)

    #ax_mea_free.plot(tvec, mea_analytic[0], lw=1, c='gray')
    l, = ax_mea_free.plot(tvec, mea_fem[0],  lw=1, c='k')
    la, = ax_mea_tunnel.plot(tvec, mea_fem[1],  lw=1, c='k', ls="-")

    rel_error = np.max(np.abs((mea_analytic[0] - mea_fem[0])) / np.max(
        np.abs(mea_fem[0])))
    print("Relative error between FEM and MoI (free elec): {:1.4f}".format(
        rel_error))

    t1 = ax_vmem.axvline(tvec[0], c='gray', ls="--")
    t2 = ax_mea_free.axvline(tvec[0], c='gray', ls="--")
    t3 = ax_mea_tunnel.axvline(tvec[0], c='gray', ls="--")

    simplify_axes([ax_setup, ax_mea_free, ax_mea_tunnel, ax_vmem, ax_EAP_decay])
    #mark_subplots([ax_vmem, ax_mea_free, ax_mea_tunnel, ax_EAP_decay], "BCDE", xpos=-0.05, ypos=1.07)

    ## This is to make animation. Can be commented out to save time
    for t_idx in range(num_tsteps):

        for tp in ep_intervals.collections:
            tp.remove()
        ep_intervals = ax_setup.contourf(x, z, xz_masked[:, :, t_idx].T,
                                         zorder=-2, colors=colors_from_map,
                                         levels=levels_norm, extend='both')
        for tp in ep_intervals_.collections:
            tp.remove()

        ep_intervals_ = ax_setup.contour(x, z, xz_masked[:, :, t_idx].T,
                                         colors='k', linewidths=(1), zorder=-2,
                         levels=levels_norm)

        t1.set_xdata(tvec[t_idx])
        t2.set_xdata(tvec[t_idx])
        t3.set_xdata(tvec[t_idx])

        plt.savefig(join(fem_fig_folder, 'anim_results_{}_t_idx_{:04d}.png'.format(
            sim_name, t_idx)), dpi=300)

    cax.clear()
    t1.set_xdata(100)
    t2.set_xdata(100)
    t3.set_xdata(100)

    # num = 11
    # levels = np.logspace(-2.0, 0, num=num)
    # scale_max = 500
    # levels_norm = scale_max * levels
    # bwr_cmap = plt.cm.get_cmap('Reds')  # rainbow, spectral, RdYlBu
    # colors_from_map = [bwr_cmap(i * np.int(255 / (len(levels_norm) - 2)))
    #                    for i in range(len(levels_norm) - 1)]
    #
    #

    levels_norm = [0, 10.0, 1e9]#scale_max * levels
    # bwr_cmap = plt.cm.get_cmap('Reds')  # rainbow, spectral, RdYlBu
    colors_from_map = ['0.95', '#ffbbbb', (0.5, 0.5, 0.5, 1)]

    #colors_from_map[0] = (1.0, 1.0, 1.0, 1.0)

    xz_crossmax = np.array(np.max(np.abs(xz_masked[:, :, :]), axis=-1).T)

    for tp in ep_intervals.collections:
        tp.remove()
    ep_intervals = ax_setup.contourf(x, z, xz_crossmax,
                                     zorder=-2, colors=colors_from_map,
                                     levels=levels_norm, extend='both')

    for tp in ep_intervals_.collections:
        tp.remove()

    # ep_intervals_ = ax_setup.contour(x, z, xz_crossmax, colors='k',
    #                                  linewidths=(0.3), zorder=-2,
    #                                  levels=levels_norm)

    #img1 = ax_setup.imshow(np.max(np.abs(xz_masked), axis=-1).T,
    #           interpolation='nearest', origin='lower', cmap='Reds',
    #           extent=(x[0], x[-1], z[0], z[-1]), norm=LogNorm(0.002, vmax=1))

    cbar = plt.colorbar(ep_intervals, cax=cax)
    # cbar.set_ticks(np.array([0.01, 0.1, 1]) * scale_max)
    # print(cbar.get_ticks())
    cbar.set_ticks(np.array([5, 1e9/2]))
    cbar.set_ticklabels(np.array(["<10 µV", ">10 µV"]))
    #cax.set_xticklabels(np.array(np.array([-1, -0.1, -0.01, 0, 0.01, 0.1, 1])
                    # * scale_max, dtype=int), fontsize=7, rotation=0)
    plt.savefig(join(root_folder, 'Fig_{}_4.png'.format(sim_name)), dpi=300)
    plt.savefig(join(root_folder, 'Fig_{}_4.pdf'.format(sim_name)), dpi=300)


    plot_data_folder = join(root_folder, 'figure_data')
    os.makedirs(plot_data_folder, exist_ok=True)
    np.save(join(plot_data_folder, "xz_max_amp.npy"), xz_crossmax)
    np.save(join(plot_data_folder, "xz_x_values.npy"), x)
    np.save(join(plot_data_folder, "xz_z_values.npy"), z)
    np.save(join(plot_data_folder, "mea_phi_values.npy"), mea_fem)
    np.save(join(plot_data_folder, "source_line_pos.npy"), source_line_pos)
    np.save(join(plot_data_folder, "t_vec.npy"), tvec)
    np.save(join(plot_data_folder, "memb_pot.npy"), vmem[cell_plot_idxs, :])
    np.save(join(plot_data_folder, "eap_amp.npy"), eap_amp[x_idxs])
    np.save(join(plot_data_folder, "eap_amp_dist.npy"), dist_from_soma)


if __name__ == '__main__':
    #simulate_FEM()
    # plot_soma_EAP_amp_with_distance()
    make_results_figure()
