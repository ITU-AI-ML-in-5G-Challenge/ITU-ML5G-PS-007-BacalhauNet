1) Check if all python depencies are installed (odfpy package is needed)
2) Dataset must be loaded on /workspace/dataset
3) Execute script.sh (inside source folder)
4) Final weights are saved on models/finalweights.pth
5) evaluation.ipynb may be used to confirm accuracy and inference cost score

# Further comments:
-> In our experience the 56% accuracy requirement may not be met depending on the initial weight initialization. Several executions of the described reproduction procedure may be required.