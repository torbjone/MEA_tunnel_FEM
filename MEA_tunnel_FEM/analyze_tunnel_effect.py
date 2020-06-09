import os
from os.path import join
import numpy as np
import matplotlib
#matplotlib.use("AGG")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from plotting_convention import simplify_axes, mark_subplots
eps = 1e-9

sigma = 0.3  # Extracellular conductivity (S/m)

dx_tunnel = 300.0  # um
nx = 300
x0 = -dx_tunnel / 2
x1 = x0 + dx_tunnel
x = np.linspace(x0, x1, nx)

def analytic_mea(x, y, z):
    phi = np.zeros(imem.shape[1])
    for idx in range(imem.shape[0]):
        print(xmid[idx], ymid[idx], zmid[idx])
        r = np.sqrt((x - xmid[idx])**2 + (y - ymid[idx])**2 + (z - zmid[idx])**2)
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


if __name__ == '__main__':
    data_folder = 'results_remade'
    sim_name = "tunnel_test_remade"
    fig_folder = "tunnel_effect_figs"

    dx_tunnel = 300.0  # um
    nx = 300
    x0 = -dx_tunnel / 2
    x1 = x0 + dx_tunnel
    x = np.linspace(x0, x1, nx)
    os.makedirs(fig_folder, exist_ok=True)

    imem = np.load(join(data_folder, "axon_imem.npy"))
    vmem = np.load(join(data_folder, "axon_vmem.npy"))
    tvec = np.load(join(data_folder, "axon_tvec.npy"))
    xmid, ymid, zmid = np.load(join(data_folder, "source_pos.npy")).T
    # print(source_pos.shape)
    # xmid = np.load(join(data_folder, "axon_xmid.npy"))
    # ymid = np.load(join(data_folder, "axon_ymid.npy"))
    # zmid = np.load(join(data_folder, "axon_zmid.npy"))

    num_tsteps = len(tvec)
    num_comps = imem.shape[0]

    # print(len(xmid), imem.shape)

    phi_mid = np.zeros(num_tsteps)
    mid_idx_elec = np.argmin(np.abs(x))
    mid_idx_comp = np.argmin(np.abs(xmid))

    phi_mid_moi = analytic_mea(x[mid_idx_elec], 0.0, 0.0)
    for t_idx in range(num_tsteps):
        phi = np.load(join(data_folder, "mea_x_values_{}_t_idx_{:04d}.npy".format(sim_name, t_idx)))
        phi_mid[t_idx] = phi[mid_idx_elec]
        #print(phi.shape)
        #plot_FEM_results(phi, t_idx)

    fig = plt.figure(figsize=[8, 8])
    fig.subplots_adjust(wspace=0.5, top=0.7, hspace=0.6, left=0.15)
    ax0 = fig.add_axes([0.15, 0.8, 0.7, 0.15], xlim=[-120, 120], ylim=[-3, 6], xlabel="x ($\mu$m)", ylabel="y ($\mu$m)", title="geometry")
    ax1 = fig.add_subplot(221, xlabel="time (ms)", title="membrane potential", ylabel="mV")
    ax2 = fig.add_subplot(222, xlabel="time (ms)", title="transmembrane currents", ylabel="nA")
    ax3 = fig.add_subplot(223, xlabel="time (ms)", title="$\phi$, semi-infinite", ylabel="$\mu$V")
    ax4 = fig.add_subplot(224, xlabel="time (ms)", title="$\phi$, with tunnel", ylabel="$\mu$V")

    ax0.text(xmid[0], zmid[0] + 0, "axon", color='green', va='bottom', ha="left")
    ax0.text(x[mid_idx_elec], -0.5, "electrode", color='k', va='top', ha="left")
    ax0.plot(xmid, zmid, c='g', lw=3)
    ax0.plot([x[mid_idx_elec] - 5, x[mid_idx_elec] + 5], [0, 0], lw=4, c='k')
    ax0.plot(xmid[mid_idx_comp], zmid[mid_idx_comp], '*',  ms=15, c='r')

    ax0.axhline(0, ls=':', lw=0.5, c='gray')
    ax0.axhline(5, ls=':', lw=0.5, c='gray')

    ax1.plot(tvec, vmem[mid_idx_comp], c='r', lw=2)
    ax2.plot(tvec, imem[mid_idx_comp], c='r', lw=2)
    ax3.plot(tvec, 1000 * phi_mid_moi, c='k', lw=2)
    ax4.plot(tvec, 1000 * phi_mid, c='k', lw=2)

    simplify_axes(fig.axes)

    plt.savefig(join(fig_folder, "tunnel_effect.png"))


