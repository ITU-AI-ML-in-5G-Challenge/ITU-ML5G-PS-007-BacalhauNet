#!/bin/bash

python3 bacalhaunet_quant_reg.py run_config.ods outputs /workspace/dataset/GOLD_XYZ_OSC.0001_1024.hdf5
python3 bacalhaunet_quant_reg_prune1.py run_config.ods outputs outputs /workspace/dataset/GOLD_XYZ_OSC.0001_1024.hdf5
python3 bacalhaunet_quant_reg_prune2.py run_config.ods outputs outputs /workspace/dataset/GOLD_XYZ_OSC.0001_1024.hdf5
python3 bacalhaunet_quant_reg_prune3.py run_config.ods outputs outputs /workspace/dataset/GOLD_XYZ_OSC.0001_1024.hdf5