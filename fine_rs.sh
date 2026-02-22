#!/usr/bin/env bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 train.py --dataset refseg --model_id RMSIN --epochs 60 --output-dir ./job_data --img_size 1024 2>&1 | tee ./output

WORKDIR=$(pwd)
echo ${WORKDIR}
cd ./job_data/ckpt_model_best
python zero_to_fp32.py . ../pytorch_model.bin
cd ${WORKDIR}
python3 merge_lora_weights_and_save_hf_model.py --weight ./job_data/pytorch_model.bin --save_path ./job_data/evf

python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 test.py --dataset refseg --model_id RMSIN --inference --version='./job_data/evf' --precision="fp32" --img_size 1024
cd ./job_data
rm -rf ckpt_model
rm -rf checkpoint.pth
rm -rf ckpt_model_best
rm -rf checkpoint_best.pth
rm -rf pytorch_model.bin