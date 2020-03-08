#!/bin/bash


python cluster_gcn_amazon.py --gpu 0 --dataset amazon2m --lr 1e-2 --weight-decay 0.0 --psize 15000 --batch-size 10 --psize-val 200\
  --n-epochs 30 --n-hidden 400 --n-layers 3 --log-every 100 --use-pp --self-loop \
  --note self-loop-amazon2m-15000-10-30-400-2-pp-cluster-2-2-wd-0 --dropout 0.2 --use-val --normalize
