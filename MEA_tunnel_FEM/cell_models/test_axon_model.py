import os
import sys
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import neuron
import LFPy


morph_file = '950923b.CNG.swc'
neuron.load_mechanisms(join("HallermannEtAl2012"))
cell_parameters = {
        'morphology': morph_file,
        'v_init': -85,
        'nsegs_method': "lambda_f",
        "lambda_f": 500,
        "tstart": -50,
        "tstop": 3,
        "dt": 2**-6,
        "pt3d": True,
        "custom_code": ["add_axon.hoc"]
}

cell = LFPy.Cell(**cell_parameters)
cell.set_rotation(x=np.pi/2, )
cell.set_pos(x=-200, z=65)

stim_params = {
             'idx': 0,
             'record_current': True,
             'syntype': 'Exp2Syn',
             'tau1': 0.1,
             'tau2': 0.1,
             'weight': 0.1,
            }

syn = LFPy.Synapse(cell, **stim_params)
syn.set_spike_times(np.array([0.1]))


# for sec in neuron.h.allsec():
#     print(sec.name())

# print cell.totnsegs

print(np.min(cell.zmid))
cell.simulate(rec_vmem=True, rec_imem=True)

elec_x_mea = np.array([-150, 0])
elec_y_mea = np.array([0, 0])
elec_z_mea = np.array([0, 0])

sigma = 1  # S/m
noise_level = 10  # uV

# Define electrode parameters
mea_parameters = {
    'sigma_T': sigma,      # Saline bath conductivity
    'sigma_S': sigma,      # Saline bath conductivity
    'sigma_G': 0,      # Saline bath conductivity
    'x': elec_x_mea,  # electrode requires 1d vector of positions
    'y': elec_y_mea,
    'z': elec_z_mea,
    "method": "soma_as_point",
}
mea_electrode = LFPy.RecMEAElectrode(cell, **mea_parameters)

mea_electrode.calc_lfp()
# mea_electrode.LFP *= 2  # Because of non-conducting electrode plane

soma_diam = 10
elec_x = np.linspace(-200 + soma_diam/2, -100 + soma_diam/2, 50)
elec_y = np.zeros(len(elec_x))
elec_z = np.ones(len(elec_x)) * 65

# Define electrode parameters
elec_parameters = {
    'sigma': sigma,      # Saline bath conductivity
    'x': elec_x,  # electrode requires 1d vector of positions
    'y': elec_y,
    'z': elec_z,
    "method": "soma_as_point",
}
electrode = LFPy.RecExtElectrode(cell, **elec_parameters)

electrode.calc_lfp()

fig = plt.figure(figsize=[9, 6])
fig.subplots_adjust(wspace=0.5, top=0.97, bottom=0.1)
ax1 = fig.add_subplot(211, aspect=1, frameon=False, xlim=[-250, 250],
                      ylim=[-30, 150])
ax2 = fig.add_subplot(234, xlabel="time (ms)", ylabel="membrane\npotential (mV)")
ax4 = fig.add_subplot(235, xlabel="time (ms)", ylabel="$\phi$ ($\mu$V)")
ax5 = fig.add_subplot(236, xlabel="distance from soma ($\mu$m)",
                      ylabel="amplitude ($\mu$V)")

ax1.plot(elec_x, elec_z, ls=':', c='blue', lw=2)
# from matplotlib.collections import PolyCollection
# zips = []
# for x, z in cell.get_idx_polygons(projection=('x', 'z')):
#     zips.append(list(zip(x, z)))
# polycol = PolyCollection(zips, edgecolors='none', facecolors='k')
# ax1.add_collection(polycol)
[ax1.plot([cell.xstart[idx], cell.xend[idx]], [cell.zstart[idx], cell.zend[idx]],
          c='gray', lw=1) for idx in range(cell.totnsegs)]
ax1.plot(cell.xmid[0], cell.zmid[0], 'o', c='gray', ms=18)


ax1.axhline(0)
plot_idxs = np.array([0,
                      cell.get_closest_idx(x=-150, z=-2.5, y=0, section="axon"),
                      cell.get_closest_idx(x=0, z=-2.5, y=0, section="axon")])
plot_idx_clrs = {idx: plt.cm.viridis(num / (len(plot_idxs) - 1))
                 for num, idx in enumerate(plot_idxs)}

for idx in plot_idxs:
    ax1.plot(cell.xmid[idx], cell.zmid[idx], 'o', c=plot_idx_clrs[idx])
    ax2.plot(cell.tvec, cell.vmem[idx, :], c=plot_idx_clrs[idx], lw=2)

for elec_idx in range(len(elec_x_mea)):
    ax1.plot(elec_x_mea[elec_idx], elec_z_mea[elec_idx], 's')
    ax4.plot(cell.tvec, mea_electrode.LFP[elec_idx] * 1000)

eap_amps = np.zeros(len(elec_x))
for elec_idx in range(len(elec_x)):
    eap_amps[elec_idx] = 1000 * np.max(np.abs(electrode.LFP[elec_idx]))

noise_level_dist_idx = np.argmin(np.abs(eap_amps - noise_level))
noise_level_dist = elec_x[noise_level_dist_idx] - cell.xmid[0]
ax5.plot(elec_x - cell.xmid[0], eap_amps, 'b', lw=2)
ax5.axhline(noise_level, ls='--', c='gray')
ax5.axvline(noise_level_dist, ls='--', c='gray')
ax5.text(noise_level_dist + 2, noise_level + 5, "{:1.1f} $\mu$m".format(
    noise_level_dist))


plt.savefig(join("morph_test.png"))