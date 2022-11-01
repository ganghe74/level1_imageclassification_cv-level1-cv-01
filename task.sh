#!/bin/bash
python train.py -c configs/ENet-b1-weightdecay-0.01.json
python train.py -c configs/ENet-b1-weightdecay-0.1.json
python train.py -c configs/ENet-b1-weightdecay-1.json
python train.py -c configs/ENet-b1-weightdecay-2.json