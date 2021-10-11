# Instructions for reproduction:
1. Required python packages must be installed:
   - `pip install odfpy`
2. Environment variables must be set according to `https://github.com/Xilinx/brevitas-radioml-challenge-21` instructions.
3. Execute script `run-docker.sh`
4. Within docker, go to notebooks directory
5. Open a terminal instance and execute `script.sh`
6. When train is complete, the final weights are saved on `models/finalweights.pth`
7. Jupyter notebook `evaluation.ipynb` may be used to confirm accuracy and inference cost score

# Further comments:
In our experience the 56% accuracy requirement may not be met depending on the initial weight initialisation, several executions of the described reproduction procedure may be required.
To overcome this drawback you may load the weights produced after the training procedure and before the 1st pruning:
1. Open `./notebooks/bacalhaunet_quant_reg_prune1.py`
2. Uncomment line 300.
3. Run the commands below:
``` bash
python3 bacalhaunet_quant_reg_prune1.py run_config.ods outputs outputs /workspace/dataset/GOLD_XYZ_OSC.0001_1024.hdf5
python3 bacalhaunet_quant_reg_prune2.py run_config.ods outputs outputs /workspace/dataset/GOLD_XYZ_OSC.0001_1024.hdf5
python3 bacalhaunet_quant_reg_prune3.py run_config.ods outputs outputs /workspace/dataset/GOLD_XYZ_OSC.0001_1024.hdf5
```

