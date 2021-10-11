#!/usr/bin/python3

import os
import netron
from sys import argv


def showInNetron(model_filename):
    localhost_url = os.getenv("LOCALHOST_URL") if os.getenv("NETRON_PORT") else "127.0.0.1"
    netron_port = int(os.getenv("NETRON_PORT")) if os.getenv("NETRON_PORT") else 8081
    netron.start(model_filename, address=(localhost_url, netron_port))


if __name__ == "__main__":
    final_onnx_path = str(argv[1])
    showInNetron(final_onnx_path)
