cd "$(dirname "$0")/../.."

data_list=("Egc" "Egb" "Eea" "Ei" "Xc" "EPS" "Nc" "Eat")
for fold in {1..5}
       do
              for data in "${data_list[@]}"
                     do
                            batch_size=4
                            task_name="dataset_${data}_fold_${fold}" 
                            data_path="./datasets/property_dataset/dataset_${data}"
                            weight_path="./ckpts/property_ckpt/ckpt_${data}/ckpt_${data}_fold_${fold}/checkpoint.pt"
                            results_path="./results/property_result/result_${data}/result_${data}_fold_${fold}"
                            
                            export CUDA_VISIBLE_DEVICES=0  
                            python ./polyconfm/infer.py --user-dir ./polyconfm $data_path --valid-subset test \
                                   --task polyconfm_property --loss polyconfm_property --arch polyconfm_property \
                                   --task-name $task_name --path $weight_path --results-path $results_path \
                                   --num-workers 8 --ddp-backend=c10d \
                                   --classification-head-name $task_name \
                                   --batch-size $batch_size --required-batch-size-multiple 1 \
                                   --log-interval 10 --log-format simple
                     done
       done