#!/usr/bin/env python
'''
Toy example to get transmembrane currents from axon
'''
import os
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
import neuron
import LFPy


root_folder = '..'
outfolder = join(root_folder, "neural_sim_results")
os.makedirs(outfolder, exist_ok=True)

cell_model_folder = "cell_models"

def return_cell():

    neuron.load_mechanisms(join(cell_model_folder, "HallermannEtAl2012"))
    # Define cell parameters
    cell_parameters = {          # various cell parameters,
        'morphology' : join(cell_model_folder, 'unmyelinated_axon_Hallermann.hoc'), # Mainen&Sejnowski, 1996
        'v_init' : -80.,    # initial crossmembrane potential
        'passive' : False,   # switch off passive mechs
        'nsegs_method' : 'lambda_f',
        'lambda_f' : 1000.,
        'dt' : 2.**-6,   # [ms] dt's should be in powers of 2
        'tstart' : -200.,    # start time of simulation, recorders start at t=0
        'tstop' : 3.,   # stop simulation at 2 ms.
    }

    cell = LFPy.Cell(**cell_parameters)
    cell.set_pos(x=-200, z=100)

    #  To induce a spike:
    synapseParameters = {
        'idx' : 0,               # insert synapse on index "0", the soma
        'e' : 0.,                # reversal potential of synapse
        'syntype' : 'Exp2Syn',   # conductance based double-exponential synapse
        'tau1' : 0.1,            # Time constant, decay
        'tau2' : 0.1,            # Time constant, decay
        'weight' : 0.0075,         # Synaptic weight
        'record_current' : False, # Will enable synapse current recording
    }

    # attach synapse with parameters and input time
    synapse = LFPy.Synapse(cell, **synapseParameters)
    synapse.set_spike_times(np.array([0.1]))

    cell.simulate(rec_vmem=True, rec_imem=True)
    return cell

cell = return_cell()

# print(cell.xmid, cell.ymid, cell.zmid)

source_pos = np.array([cell.xmid, cell.ymid, cell.zmid]).T
np.save(join(outfolder, "source_pos.npy"), source_pos)
np.save(join(outfolder, "axon_imem.npy"), cell.imem)
np.save(join(outfolder, "axon_tvec.npy"), cell.tvec)
np.save(join(outfolder, "axon_vmem.npy"), cell.vmem)

print(np.sum(cell.imem, axis=0))

print(cell.imem.shape)

max_vmem_t_idx = np.argmax(np.abs(cell.vmem[-1] - cell.vmem[0, 0]))
max_imem_t_idx = np.argmax(np.abs(cell.imem[-1] - cell.imem[0, 0]))

fig = plt.figure(figsize=[9, 9])
fig.subplots_adjust(wspace=0.5, hspace=0.5)


ax0 = fig.add_subplot(221, xlabel="x ($\mu$m)", ylabel="z ($\mu$m)", title="morphology")
ax1 = fig.add_subplot(222, xlabel="time (ms)", ylabel="membrane potential (mV)")
ax2 = fig.add_subplot(223, xlabel="time (ms)", ylabel="transmembrane current (nA)")
ax3 = fig.add_subplot(224, xlabel="x ($\mu$m)",
                      ylabel="Membrane current at t={:1.2f}".format(cell.tvec[max_vmem_t_idx]))
ax1.axvline(cell.tvec[max_imem_t_idx], ls=":", c='gray')
ax2.axvline(cell.tvec[max_imem_t_idx], ls=":", c='gray')
[ax1.plot(cell.tvec, cell.vmem[idx, :]) for idx in range(cell.totnsegs)]
[ax2.plot(cell.tvec, cell.imem[idx, :]) for idx in range(cell.totnsegs)]

for comp in range(cell.totnsegs):
    if comp == 0:
        ax0.plot(cell.xmid[comp], cell.zmid[comp], 'o', ms=12, c='k')
    else:
        ax0.plot([cell.xstart[comp], cell.xend[comp]], [cell.zstart[comp], cell.zend[comp]], c='k')

clrs = lambda t_idx: plt.cm.Reds(t_idx / len(cell.tvec))
for t_idx in np.arange(len(cell.tvec)):
    ax3.plot(cell.xmid, cell.imem[:, t_idx], c=clrs(t_idx))

ax3.plot(cell.xmid, cell.imem[:, max_imem_t_idx], lw=2, c='k')
ax3.axhline(0, c='gray', ls='--')

plt.savefig(join(outfolder, "axon_simulation.png"))
