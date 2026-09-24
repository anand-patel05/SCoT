#!/bin/bash
name=nextqa_tv_G5       # nextqa_CL_tv_biplab_2ndN_Improve2    seed      master-port     epochs=100 if converge     comp_cate      memory_req

output_base=output/nextqa/$name

export PYTHONPATH=$PYTHONPATH:$(pwd)/VL-T5/nextqa
export NLTK_DATA=/home/vis-comp/24m2119/HypVQACL/VL-T5/nextqa/nltk_data

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1                # nextqa_CL_tv_biplab_2ndN

SEEDS=(10)                                       # std -> 10; comp -> 6666

for seed in "${SEEDS[@]}"; do
    output=${output_base}_seed${seed}
    torchrun \
        --nproc_per_node=$1 \
        --master_port $((12355 + seed % 1000)) \
        -m nextqa_CL_tv \
            --distributed --multiGPU \
            --optim adamw \
            --warmup_ratio 0.1 \
            --clip_grad_norm 5 \
            --epochs 3 \
            --num_workers 4 \
            --backbone 'models/t5-base' \
            --output $output ${@:2} \
            --num_beams 5 \
            --batch_size 80 \
            --valid_batch_size 100 \
            --from_scratch \
            --comp_cate G5 \
            --now_train \
            --memory \
            --m_size 500 \
            --ifseed \
            --seed $seed \
            --proto_beta 0.5 \
            --proto_alpha 0.3
done


# #!/bin/bash
# name=tv                     

# output=VL-T5/snap/nextqa/$name

# export PYTHONPATH=$PYTHONPATH:$(pwd)/VL-T5/nextqa
# export NLTK_DATA=/home/vis-comp/24m2119/nltk_data

# torchrun \
#     --nproc_per_node=$1 \
#     --master_port 12340 \
#     -m nextqa_CL_tv \
#         --distributed --multiGPU \
#         --optim adamw \
#         --warmup_ratio 0.1 \
#         --clip_grad_norm 5 \
#         --epochs 3 \
#         --num_workers 4 \
#         --backbone 't5-base' \
#         --output $output ${@:2} \
#         --num_beams 5 \
#         --batch_size 80 \
#         --valid_batch_size 100 \
#         --from_scratch \
#         --memory \
#         --m_size 500 \
#         --comp_cate G1 \
#         --ifseed --seed 6666 --proto_beta 0.5 --proto_alpha 0.3 \
