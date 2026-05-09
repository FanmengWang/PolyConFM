cd "$(dirname "$0")/../.."

data_list=("Egc" "Egb" "Eea" "Ei" "Xc" "EPS" "Nc" "Eat")
for fold in {1..5}
       do
              for data in "${data_list[@]}"
                     do
                            seed=0
                            n_gpu=1
                            lr=3e-5
                            epoch=600
                            warmup=0.06
                            dropout=0.1
                            batch_size=2
                            MASTER_PORT=9999
                            local_batch_size=2
                            metric="valid_agg_r2"
                            task_name="dataset_${data}_fold_${fold}" 
                            data_path="./datasets/property_dataset/dataset_${data}"
                            save_dir="./ckpts/property_ckpt/ckpt_${data}/ckpt_${data}_fold_${fold}"
                            weight_path='./ckpts/pretrain_ckpt/pretrain_phase2_ckpt/checkpoint_best.pt'

                            export NCCL_ASYNC_ERROR_HANDLING=1
                            export OMP_NUM_THREADS=1
                            update_freq=`expr $batch_size / $local_batch_size`
                            torchrun --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path \
                                   --task-name $task_name --user-dir ./polyconfm --train-subset train --valid-subset valid \
                                   --num-workers 8 --ddp-backend=c10d \
                                   --task polyconfm_property --loss polyconfm_property --arch polyconfm_property \
                                   --classification-head-name $task_name \
                                   --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
                                   --batch-size $local_batch_size --required-batch-size-multiple 1 \
                                   --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --pooler-dropout $dropout\
                                   --update-freq $update_freq --seed $seed \
                                   --finetune-from-model $weight_path \
                                   --log-interval 100 --log-format simple \
                                   --best-checkpoint-metric $metric --maximize-best-checkpoint-metric \
                                   --validate-interval 1 --patience 30 --keep-last-epochs 10 \
                                   --save-dir $save_dir --tmp-save-dir $save_dir/tmp --tensorboard-logdir $save_dir/tsb \
                                   --find-unused-parameters 
                     done
       done
