# The name of experiment
name=scot_vqav2_comp         

output=output/vqacl_comp/$name  

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1   

export PYTHONPATH=$PYTHONPATH:$(pwd)/VL-T5/src

torchrun \
    --nproc_per_node=$1 \
    --master_port 12345 \
    -m vqacl_comp_tv \
        --distributed --multiGPU \
        --train karpathy_train \
        --valid karpathy_val \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 1e-4 \
        --epochs 3 \
        --num_workers 4 \
        --backbone 'models/t5-base' \
        --output $output ${@:2} \
        --num_beams 5 \
        --batch_size 80 \
        --valid_batch_size 100 \
        --from_scratch \
        --comp_cate G1
