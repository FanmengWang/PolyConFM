cd "$(dirname "$0")/../.."

batch_size=4
data_path="./datasets/design_dataset"
results_path="./results/design_result"
weight_path="./ckpts/design_ckpt/checkpoint.pt"

export CUDA_VISIBLE_DEVICES=0  
python ./polyconfm/infer.py --user-dir ./polyconfm $data_path --valid-subset test \
       --task polyconfm_design --loss polyconfm_design_inference --arch polyconfm_design \
       --path $weight_path  --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d \
       --batch-size $batch_size --required-batch-size-multiple 1 \
       --log-interval 1 --log-format simple --mode infer