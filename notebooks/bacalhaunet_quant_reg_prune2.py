from bacalhaunetv1_quant import BacalhauNetV1, BacalhauNetConfig, BacalhauNetLayerConfig
from brevitas.export.onnx.generic.manager import BrevitasONNXManager
from finn.util.inference_cost import inference_cost
from dataset_wrapper import RadioML18Dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from sys import argv
from torch import nn
import torch.nn.utils.prune as prune
from brevitas import nn as qnn
import pandas as pd
import numpy as np
import datetime
import torch
import time
import json
import os
import copy

help_msg = f"\nBased on an excel with the columns 'kernel_length', 'stride_length' and 'output_channels' computes " \
           f"the accuracy and inference cost of a model and stores it on the 'test_accuracy', 'inference_cost', " \
           f"'inference_cost_bops' and 'inference_cost_wbits' columns. Kernel, stride and output channels " \
           f"must have the same number of elements separeted by a comma. Script arguments:\n" \
           f"- excel_path: the path pointing to the excel to read and write the values.\n" \
           f"- export_path: the path pointing to a directory where the output files (ONNX models, Inference Cost " \
           f"Dict, Model as string) will be saved.\n" \
           f"- dataset_path: the path pointing to radioml2018.01a dataset ('GOLD_XYZ_OSC.0001_1024.hdf5').\n" \
           f"example: python evaluate_bacalhaunet.py ./Results_BacalhauNet_5.ods ./models/ " \
           f"./datasets/deepsig-radioml.2018.01a/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5"

min_snr = -6
batch_size = 1024
num_epochs = 20
testmode = False
evaluate_acc = True

try:
    excel_path = argv[1]
    export_path = argv[2]
    import_path = argv[3]
    dataset_path = argv[4]
except IndexError:
    raise ValueError(help_msg)

if not os.path.isfile(excel_path):
    raise FileNotFoundError(f"File {excel_path} not found.")
if not os.path.isdir(export_path):
    raise FileNotFoundError(f"Directory {export_path} not found.")
if not os.path.isdir(import_path):
    raise FileNotFoundError(f"Directory {import_path} not found.")
if not os.path.isfile(dataset_path):
    raise FileNotFoundError(f"Dataset {dataset_path} not found!")

print(f"Running {excel_path}.\n"
      f"Exporting to {export_path}.\n"
      f"Minimum Train SNR set to {min_snr}")

bops_baseline = 807699904
w_bits_baseline = 1244936

# Select which GPU to use (if available)
gpu = 0
if torch.cuda.is_available():
    torch.cuda.device(gpu)
    print(f"Using GPU {gpu}")
else:
    gpu = None
    print("Using CPU only")

file: pd.DataFrame
file = pd.read_excel(excel_path)

def train(model, train_loader, optimizer, criterion):
    losses = []
    # ensure model is in training mode
    model.train()

    for (inputs, target, snr) in tqdm(train_loader, desc="Batches", leave=False):
        if gpu is not None:
            inputs = inputs.cuda()
            target = target.cuda()

        # forward pass
        output = model(inputs)
        loss = criterion(output, target)

        # backward pass + run optimizer to update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # keep track of loss value
        losses.append(loss.cpu().detach().numpy())

    return losses


def test(model, test_loader):
    # ensure model is in eval mode
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for (inputs, target, snr) in test_loader:
            if gpu is not None:
                inputs = inputs.cuda()
                target = target.cuda()
            output = model(inputs)
            pred = output.argmax(dim=1, keepdim=True)
            y_true.extend(target.tolist())
            y_pred.extend(pred.reshape(-1).tolist())

    return accuracy_score(y_true, y_pred)


def save_model(model, path):
    for m in model.modules():
        if isinstance(m, qnn.QuantConv1d):
            prune.remove(m, "weight")
        if isinstance(m, qnn.QuantLinear):
            prune.remove(m, "weight")

    torch.save(model.state_dict(), path)

    for m in model.modules():
        if isinstance(m, qnn.QuantConv1d):
            w = m.weight.data.clone().cpu().numpy().flatten().tolist()
            w = np.abs(w)
            count = sum(map(lambda x : x <= 0, w))
            perc = count / len(w)
            prune.l1_unstructured(m, name="weight", amount=perc)
        if isinstance(m, qnn.QuantLinear):
            w = m.weight.data.clone().cpu().numpy().flatten().tolist()
            w = np.abs(w)
            count = sum(map(lambda x : x <= 0, w))
            perc = count / len(w)
            prune.l1_unstructured(m, name="weight", amount=perc)


for line in range(len(file)):
    # ignore lines without parameters defined
    if pd.isnull(file.loc[line, "hardtanh_bit_width"]) or pd.isnull(file.loc[line, "kernel_length"]) or \
            pd.isnull(file.loc[line, "stride_length"]) or pd.isnull(file.loc[line, "output_channels"]) or \
            pd.isnull(file.loc[line, "weight_bit_width"]) or pd.isnull(file.loc[line, "activation_bit_width"]) or \
            pd.isnull(file.loc[line, "maxpool_bit_width"]) or pd.isnull(file.loc[line, "dropout_prob"]) or \
            pd.isnull(file.loc[line, "fullyconnected_bit_width"]):
        print(f"Missing parameters on line {line}. Ignoring and resuming...")
        continue

    # When a space is missing between some values this is considered a float. In this case we reformat the value.
    if isinstance(file.loc[line, "kernel_length"], float):
        kernels = str(file.loc[line, "kernel_length"]).strip().split(".")
        if kernels[-1] == "0":
            kernels = [int(kernels[0])]
        else:
            kernels = list(np.asarray(kernels).astype(int))
    elif isinstance(file.loc[line, "kernel_length"], str):
        kernels = list(np.asarray(str(file.loc[line, "kernel_length"]).strip().split(",")).astype(int))
    elif isinstance(file.loc[line, "kernel_length"], int):
        kernels = [int(file.loc[line, "kernel_length"])]
    else:
        print(f"Unable to unpack kernel_length at line {line}. Ignoring and resuming...")
        continue

    if isinstance(file.loc[line, "stride_length"], float):
        strides = str(file.loc[line, "stride_length"]).strip().split(".")
        if strides[-1] == "0":
            strides = [int(strides[0])]
        else:
            strides = list(np.asarray(strides).astype(int))
    elif isinstance(file.loc[line, "stride_length"], str):
        strides = list(np.asarray(str(file.loc[line, "stride_length"]).strip().split(",")).astype(int))
    elif isinstance(file.loc[line, "stride_length"], int):
        strides = [int(file.loc[line, "stride_length"])]
    else:
        print(f"Unable to unpack stride_length at line {line}. Ignoring and resuming...")
        continue

    if isinstance(file.loc[line, "output_channels"], float):
        out_channels = str(file.loc[line, "output_channels"]).strip().split(".")
        if out_channels[-1] == "0":
            out_channels = [int(out_channels[0])]
        else:
            out_channels = list(np.asarray(out_channels).astype(int))
    elif isinstance(file.loc[line, "output_channels"], str):
        out_channels = list(np.asarray(str(file.loc[line, "output_channels"]).strip().split(",")).astype(int))
    elif isinstance(file.loc[line, "output_channels"], int):
        out_channels = [int(file.loc[line, "output_channels"])]
    else:
        print(f"Unable to unpack output_channels at line {line}. Ignoring and resuming...")
        continue

    if isinstance(file.loc[line, "weight_bit_width"], float):
        layers_w_bits = str(file.loc[line, "weight_bit_width"]).strip().split(".")
        if layers_w_bits[-1] == "0":
            layers_w_bits = [int(layers_w_bits[0])]
        else:
            layers_w_bits = list(np.asarray(layers_w_bits).astype(int))
    elif isinstance(file.loc[line, "weight_bit_width"], str):
        layers_w_bits = list(np.asarray(str(file.loc[line, "weight_bit_width"]).strip().split(",")).astype(int))
    elif isinstance(file.loc[line, "weight_bit_width"], int):
        layers_w_bits = [int(file.loc[line, "weight_bit_width"])]
    else:
        print(f"Unable to unpack weight_bit_width at line {line}. Ignoring and resuming...")
        continue

    if isinstance(file.loc[line, "activation_bit_width"], float):
        layers_a_bits = str(file.loc[line, "activation_bit_width"]).strip().split(".")
        if layers_a_bits[-1] == "0":
            layers_a_bits = [int(layers_a_bits[0])]
        else:
            layers_a_bits = list(np.asarray(layers_a_bits).astype(int))
    elif isinstance(file.loc[line, "activation_bit_width"], str):
        layers_a_bits = list(np.asarray(str(file.loc[line, "activation_bit_width"]).strip().split(",")).astype(int))
    elif isinstance(file.loc[line, "activation_bit_width"], int):
        layers_a_bits = [int(file.loc[line, "activation_bit_width"])]
    else:
        print(f"Unable to unpack activation_bit_width at line {line}. Ignoring and resuming...")
        continue

    if isinstance(file.loc[line, "weight_decay"], float):
        weight_decay = file.loc[line, "weight_decay"]
    else:
        print(f"Unable to unpack weight_decay at line {line}.")
        exit()

    if isinstance(file.loc[line, "weight_decay_tune"], float):
        weight_decay_tune = file.loc[line, "weight_decay_tune"]
    else:
        print(f"Unable to unpack weight_decay_tune at line {line}.")
        exit()

    if isinstance(file.loc[line, "min_weight_value"], float):
        min_weight_value = file.loc[line, "min_weight_value"]
    else:
        print(f"Unable to unpack min_weight_value at line {line}.")
        exit()

    if isinstance(file.loc[line, "weight_decay_tune2"], float):
        weight_decay_tune2 = file.loc[line, "weight_decay_tune2"]
    else:
        print(f"Unable to unpack weight_decay_tune2 at line {line}.")
        exit()

    if isinstance(file.loc[line, "min_weight_value2"], float):
        min_weight_value2 = file.loc[line, "min_weight_value2"]
    else:
        print(f"Unable to unpack min_weight_value2 at line {line}.")
        exit()

    if not (len(kernels) == len(strides) == len(out_channels) == len(layers_w_bits) == len(layers_a_bits)):
        print(f"Length of parameters on line {line} not equal. Ignoring and resuming...")
        continue

    # Create the layers configurations
    layers = []
    for sub_idx in range(len(kernels)):
        # recast to python int is needed on wbits and abits since brevitas required them to be python int data types
        layers.append(BacalhauNetLayerConfig(kernel=int(kernels[sub_idx]), stride=int(strides[sub_idx]),
                                             out_channels=int(out_channels[sub_idx]),
                                             w_bits=int(layers_w_bits[sub_idx]),
                                             a_bits=int(layers_a_bits[sub_idx])))

    # Defines the model with the defined layers configurations
    bacalhaunetv1 = BacalhauNetV1(
        BacalhauNetConfig(
            in_samples=1024,
            in_channels=2,
            num_classes=24,
            hardtanh_bit_width=int(file.loc[line, "hardtanh_bit_width"]),
            layers=layers,
            pool_bit_width=int(file.loc[line, "maxpool_bit_width"]),
            dropout_prob=float(file.loc[line, "dropout_prob"]),
            fc_bit_width=int(file.loc[line, "fullyconnected_bit_width"])
        )
    )

    export_base_name = f"bacalhaunetv1_tuned2" \
                       f"K{str(kernels).replace(' ', '')}" \
                       f"S{str(strides).replace(' ', '')}" \
                       f"O{str(out_channels).replace(' ', '')}" \
                       f"W{str(layers_w_bits).replace(' ', '')}" \
                       f"A{str(layers_a_bits).replace(' ', '')}_" \
                       f"H[{int(file.loc[line, 'hardtanh_bit_width'])}]" \
                       f"P[{int(file.loc[line, 'maxpool_bit_width'])}]" \
                       f"D[0.0]" \
                       f"F[{int(file.loc[line, 'fullyconnected_bit_width'])}]" \
                       f"WD[{str(weight_decay)}]" \
                       f"WDT[{str(weight_decay_tune)}]" \
                       f"minW[{str(min_weight_value)}]" \
                       f"WDT2[{str(weight_decay_tune2)}]" \
                       f"minW2[{str(min_weight_value2)}]" \
                       f"_SNR[{min_snr},30]" 


    import_base_name = f"bacalhaunetv1_tuned" \
                       f"K{str(kernels).replace(' ', '')}" \
                       f"S{str(strides).replace(' ', '')}" \
                       f"O{str(out_channels).replace(' ', '')}" \
                       f"W{str(layers_w_bits).replace(' ', '')}" \
                       f"A{str(layers_a_bits).replace(' ', '')}_" \
                       f"H[{int(file.loc[line, 'hardtanh_bit_width'])}]" \
                       f"P[{int(file.loc[line, 'maxpool_bit_width'])}]" \
                       f"D[0.0]" \
                       f"F[{int(file.loc[line, 'fullyconnected_bit_width'])}]" \
                       f"WD[{str(weight_decay)}]" \
                       f"WDT[{str(weight_decay_tune)}]" \
                       f"minW[{str(min_weight_value)}]" \
                       f"_SNR[{min_snr},30]" 


    # Load trained model
    savefile = os.path.join(import_path, import_base_name + "_finalweights.pth")
    saved_state = torch.load(savefile, map_location=torch.device("cpu"))
    bacalhaunetv1.load_state_dict(saved_state)
    if gpu is not None:
        bacalhaunetv1 = bacalhaunetv1.cuda()


    # Prune model
    print("Pruning weights with abs under " + str(min_weight_value2))
    print("")
    for m in bacalhaunetv1.modules():
        if isinstance(m, qnn.QuantConv1d):
            w = m.weight.data.clone().cpu().numpy().flatten().tolist()
            w = np.abs(w)
            count = sum(map(lambda x : x < min_weight_value2, w))
            perc = count / len(w)
            print("*** CONV LAYER ***")
            print("Number of params: " + str(len(w)) + " - Fraction under minimum: " + str(perc))
            prune.l1_unstructured(m, name="weight", amount=perc)
            print("Layer pruned!")
            print("")
        if isinstance(m, qnn.QuantLinear):
            w = m.weight.data.clone().cpu().numpy().flatten().tolist()
            w = np.abs(w)
            count = sum(map(lambda x : x < min_weight_value, w))
            perc = count / len(w)
            print("*** LINEAR LAYER ***")
            print("Number of params: " + str(len(w)) + " - Fraction under minimum: " + str(perc))
            prune.l1_unstructured(m, name="weight", amount=perc)
            print("Layer pruned!")
            print("")

    # Get accuracy after prune
    dataset = RadioML18Dataset(dataset_path=dataset_path, min_snr=min_snr)
    data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)
    test_acc = test(bacalhaunetv1, data_loader_test)
    file.loc[line, "test_accuracy_after_prune2"] = test_acc

    if evaluate_acc:
        with open(os.path.join(export_path, export_base_name + "_model.txt"), "w") as _file2write:
            _file2write.write(str(bacalhaunetv1))

        # Train the model
        if testmode:
            dataset = RadioML18Dataset(dataset_path=dataset_path,
                                       min_snr=-28, mod_classes=[22, 23], filter_testset=True)
        else:
            dataset = RadioML18Dataset(dataset_path=dataset_path,
                                       min_snr=min_snr)

        data_loader_train = DataLoader(dataset, batch_size=batch_size, sampler=dataset.train_sampler)
        data_loader_test = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)

        if gpu is not None:
            bacalhaunetv1 = bacalhaunetv1.cuda()

        # loss criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        if gpu is not None:
            criterion = criterion.cuda()
        optimizer = torch.optim.Adam(bacalhaunetv1.parameters(), lr=0.01, weight_decay=weight_decay_tune2)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)

        running_loss = []
        running_test_acc = []

        start = time.time()
        max_acc = 0
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            loss_epoch = train(bacalhaunetv1, data_loader_train, optimizer, criterion)
            test_acc = test(bacalhaunetv1, data_loader_test)
            if test_acc > max_acc:
                max_acc = test_acc
                max_acc_save_path = os.path.join(export_path, export_base_name + "_Acc" + str(max_acc) + ".pth")
                print(f"Highter accuracy found, saving model weights at {max_acc_save_path}")
                save_model(bacalhaunetv1, max_acc_save_path)

            print("Epoch %d: Training loss = %f, test accuracy = %f" % (epoch, np.mean(loss_epoch), test_acc))
            running_loss.append(loss_epoch)
            running_test_acc.append(test_acc)
            lr_scheduler.step()
        done = time.time()

        diff = done - start
        print(f"Trainning time: {str(datetime.timedelta(seconds=diff))} (h:m:s)")

        file.loc[line, "test_accuracy_final2"] = max_acc
        file.loc[line, "notes"] = os.path.join(export_path, export_base_name + "_Acc" + str(max_acc) + ".pth")

        save_model(bacalhaunetv1, os.path.join(export_path, export_base_name + "_finalweights.pth"))

    # Computes the inference cost
    export_onnx_path = os.path.join(export_path, export_base_name + "_export.onnx")
    final_onnx_path = os.path.join(export_path, export_base_name + "_final.onnx")
    cost_dict_path = os.path.join(export_path, export_base_name + "_cost.json")

    BrevitasONNXManager.export(bacalhaunetv1.cpu(), input_t=torch.randn(1, 2, 1024), export_path=export_onnx_path)
    inference_cost(export_onnx_path, output_json=cost_dict_path, output_onnx=final_onnx_path, preprocess=True,
                   discount_sparsity=True)

    with open(cost_dict_path, 'r') as f:
        inference_cost_dict = json.load(f)

    bops = int(inference_cost_dict["total_bops"])
    w_bits = int(inference_cost_dict["total_mem_w_bits"])

    file.loc[line, "inference_cost_final2"] = 0.5 * (bops / bops_baseline) + 0.5 * (w_bits / w_bits_baseline)
    file.loc[line, "inference_cost_bops_final2"] = bops / bops_baseline
    file.loc[line, "inference_cost_wbits_final2"] = w_bits / w_bits_baseline

# Set file name and save as an excel ods format
save_file_name = os.path.splitext(excel_path)[0] + "_computed" + os.path.splitext(excel_path)[1]
count = 1
while os.path.isfile(save_file_name):
    save_file_name = os.path.splitext(excel_path)[0] + "_computed_" + str(count) + os.path.splitext(excel_path)[1]
    count += 1
file.to_excel(save_file_name, index=False)
