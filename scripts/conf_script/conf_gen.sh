cd "$(dirname "$0")/../.."

for gen_idx in $(seq 1 10)  
       do
              batch_size=1
              mar_num_ar_steps=64
              data_path="./datasets/pretrain_dataset"
              results_path="./results/conf_result/conf_result_${gen_idx}"
              weight_path="./ckpts/pretrain_ckpt/checkpoint.pt"

              export CUDA_VISIBLE_DEVICES=0  
              python ./polyconfm/infer.py --user-dir ./polyconfm $data_path --valid-subset test \
                     --task polyconfm_conf_gen --loss polyconfm_conf_gen --arch polyconfm_conf_gen \
                     --path $weight_path --results-path $results_path \
                     --mar-num-ar-steps $mar_num_ar_steps --seed $gen_idx \
                     --num-workers 8 --ddp-backend=c10d \
                     --batch-size $batch_size --pad-to-multiple 1 --required-batch-size-multiple 1 \
                     --log-interval 10 --log-format simple --mode infer 
       done