cd "$(dirname "$0")/../.."

seed=0
n_gpu=1
lr=1e-4
epoch=36000
warmup=0.06
dropout=0.1
batch_size=4
MASTER_PORT=6666
local_batch_size=4
save_dir="./ckpts/design_ckpt"
data_path="./datasets/design_dataset"
weight_path='./ckpts/pretrain_ckpt/pretrain_phase2_ckpt/checkpoint_best.pt'

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
update_freq=`expr $batch_size / $local_batch_size`
torchrun --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path \
       --user-dir ./polyconfm --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task polyconfm_design --loss polyconfm_design --arch polyconfm_design \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
       --batch-size $local_batch_size --required-batch-size-multiple 1 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --pooler-dropout $dropout\
       --update-freq $update_freq --seed $seed \
       --log-interval 100 --log-format simple \
       --finetune-from-model $weight_path \
       --validate-interval 1 --keep-last-epochs 10 \
       --save-dir $save_dir --tmp-save-dir $save_dir/tmp --tensorboard-logdir $save_dir/tsb \
       --find-unused-parameters 
