#!/bin/bash -l

#SBATCH --gres=gpu:4
#SBATCH --partition=tesla
#SBATCH -N 1
#SBATCH --mem 64000
#SBATCH --time=100:00:00
#SBATCH -c 8
#SBATCH -o ./slurm/output.%A.out # STDOUT

conda activate /data/ndallase/envs/atlas


WANDB_API_KEY= python3 train_derived.py \
-gen_bs 32 \
-dis_bs 16 \
--accumulated_times 4 \
--g_accumulated_times 4 \
--dist-url 'tcp://localhost:10641' \
--dist-backend 'nccl' \
--multiprocessing-distributed \
--world-size 1 \
--rank 0 \
--dataset celeba \
--data_path ../celebahq \
--bottom_width 8 \
--img_size 256 \
--max_iter 500000 \
--gen_model Celeba256_gen \
--dis_model Celeba256_dis \
--g_window_size 16 \
--d_window_size 4 \
--g_norm pn \
--df_dim 384 \
--d_depth 3 \
--g_depth 5,4,4,4,4,4 \
--latent_dim 512 \
--gf_dim 1024 \
--num_workers 32 \
--g_lr 0.0001 \
--d_lr 0.0001 \
--optimizer adam \
--loss wgangp-eps \
--wd 1e-3 \
--beta1 0 \
--beta2 0.99 \
--phi 1 \
--eval_batch_size 10 \
--num_eval_imgs 50000 \
--init_type xavier_uniform \
--n_critic 4 \
--val_freq 10 \
--print_freq 50 \
--grow_steps 0 0 \
--fade_in 0 \
--patch_size 2 \
--diff_aug filter,translation,erase_ratio,color,hue \
--fid_stat fid_stat/fid_stats_celeba_hq_256.npz \
--ema 0.995 \
--exp_name celeba_hq_256