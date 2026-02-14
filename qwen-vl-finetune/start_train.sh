#!/bin/bash

nohup ./scripts/sft_qwen3_2b.sh > train.log 2>&1 &
nohup tensorboard --logdir ./output/runs/ --bind_all > tensorboard.log 2>&1 &