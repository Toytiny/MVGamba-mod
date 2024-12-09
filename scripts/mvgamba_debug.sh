#! /bin/bash
accelerate launch --config_file acc_configs/gpu4.yaml main.py mvgamba --workspace /mnt/xuanyuyi/results/workspace_debug_mvgamba
