#!/usr/bin/env bash
python3 neural_simulation.py
cd ../mesh_nmi_tunnel/
sh mesh_it.sh
cd ../MEA_tunnel_FEM
python3 FEM_simulations.py
