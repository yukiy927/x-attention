#!/bin/bash

#7B llama


# GLOBAL_BATCH_SIZE=$((GPU_NUM * MAX_NODE_NUM))

# DATA_CACHE_DIR="$LPAI_INPUT_DATA_1/xiaobing/megatron_pretrain/data-cache"
# CKPT_DIR="$LPAI_INPUT_DATA_1/xiaobing/megatron_pretrain/ckpt-1.3b-3t-mcore_xiaobing"

mkdir -p $DATA_CACHE_DIR
mkdir -p $CKPT_DIR

LATEST_CKPT_FILE="$CKPT_DIR/latest_checkpointed_iteration.txt"

if [[ -f $LATEST_CKPT_FILE ]]; then
    content=$(cat "$LATEST_CKPT_FILE" | tr -d '[:space:]')
    if [[ $content =~ ^[0-9]+$ ]]; then
        INIT_OR_NOT="--no-initialization"
        echo "resume from $LATEST_CKPT_FILE"
    else
        INIT_OR_NOT=""
        echo "Latest ckpt does not exist."
    fi
else
    INIT_OR_NOT=""
    echo "Ckpt does not exist."
fi

SEQ_LEN=8192
DISTRIBUTED_ARGS="--nnodes=$MIN_NODE_NUM:$MAX_NODE_NUM --nproc_per_node=$GPU_NUM --max_restarts=0 --rdzv_id=$LPAI_TASK_NAME --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv_conf timeout=600"


echo "$DISTRIBUTED_ARGS"

#DS_BASE="${LPAI_INPUT_DATA_0}/cc_net/datasets/preprocess_megatron"
DS_BASE="${LPAI_INPUT_DATASET_0}"

DATASET_CC1="${DS_BASE}/merged_2020-50"
DATASET_CC2="${DS_BASE}/merged_2021-21"
DATASET_CC3="${DS_BASE}/merged_2021-31"
DATASET_CC4="${DS_BASE}/merged_2022-21"
DATASET_CC5="${DS_BASE}/merged_2022-33"
DATASET_CC6="${DS_BASE}/merged_2022-49"
DATASET_CC7="${DS_BASE}/merged_2023-14"
DATASET_ARXIV="${DS_BASE}/merged_arxiv"
DATASET_PILE_MED="${DS_BASE}/merged_pile-med"
DATASET_REDDIT="${DS_BASE}/merged_reddit"
DATASET_DOUBAN="${DS_BASE}/merged_douban"
DATASET_STACK_EXCHANGE="${DS_BASE}/merged_stack_exchange"
DATASET_WIKIPEDIA="${DS_BASE}/merged_wikipedia"
DATASET_GITHUB="${DS_BASE}/merged_github-cleaned"
DATASET_BOOKS3="${DS_BASE}/merged_books3-minus-gutenberg"
DATASET_ZHIHU="${DS_BASE}/merged_zhihu"
DATASET_GUTENBERG="${LPAI_INPUT_DATASET_2}/gutenberg_megatron_merged"
DATASET_PILE_OF_LAW="${DS_BASE}/merged_pile-of-law"
DATASET_FALCON="${DS_BASE}/merged_falcon_refinedweb"
DATASET_GITHUB_ISSUES="${DS_BASE}/merged_github-issues"
DATASET_WANJUAN_EXAM="${DS_BASE}/merged_wanjuan_exam_text"
DATASET_WANJUAN_PATENT="${DS_BASE}/merged_wanjuan_patent_news"
DATASET_MATHPILE="${LPAI_INPUT_DATASET_1}/math_textbook_megatron_merged"
DATASET_ANCIENT_CHINESE="${LPAI_INPUT_DATASET_3}/ancient_chinese_merged"

DATASET_CC8="${LPAI_INPUT_DATASET_4}/merged_2021-10"
DATASET_CC9="${LPAI_INPUT_DATASET_5}/merged_2021-25"
DATASET_CC10="${LPAI_INPUT_DATASET_6}/merged_2021-39"
DATASET_CC11="${LPAI_INPUT_DATASET_7}/merged_2022-27"
DATASET_CC12="${LPAI_INPUT_DATASET_8}/merged_2023-06"
DATASET_CC13="${LPAI_INPUT_DATASET_9}/merged_2021-49"
DATASET_CC14="${LPAI_INPUT_DATASET_10}/merged_2021-43"
DATASET_CC15="${LPAI_INPUT_DATASET_11}/merged_2020-40"
DATASET_CC16="${LPAI_INPUT_DATASET_12}/merged_2020-34"
DATASET_CC17="${LPAI_INPUT_DATASET_13}/merged_2020-29"

DATASET="0.086 ${DATASET_CC8} 0.086 ${DATASET_CC9} 0.086 ${DATASET_CC10} 0.086 ${DATASET_CC11} 0.086 ${DATASET_CC12} 0.086 ${DATASET_CC13} 0.086 ${DATASET_CC14} 0.018 ${DATASET_ARXIV} 0.011 ${DATASET_PILE_MED} 0.001 ${DATASET_REDDIT} 0.0002 ${DATASET_DOUBAN} 0.01 ${DATASET_STACK_EXCHANGE} 0.042 ${DATASET_WIKIPEDIA} 0.045 ${DATASET_GITHUB} 0.032 ${DATASET_BOOKS3} 0.0001 ${DATASET_ZHIHU} 0.011 ${DATASET_GUTENBERG} 0.0215 ${DATASET_PILE_OF_LAW} 0.161 ${DATASET_FALCON} 0.01 ${DATASET_GITHUB_ISSUES} 0.003 ${DATASET_WANJUAN_EXAM} 0.004 ${DATASET_WANJUAN_PATENT} 0.027 ${DATASET_MATHPILE} 0.0012 ${DATASET_ANCIENT_CHINESE}"


if [ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]; then
    load_or_resume="--load $CKPT_DIR"
else
    load_or_resume="--load $CKPT_DIR"
fi


# see https://github.com/NVIDIA/Megatron-LM/blob/main/docs/llama_mistral.md#launch-model-1
MODEL_ARGS="--seq-length 8192 \
--max-position-embeddings 131072 \
--exit-on-missing-checkpoint \
--no-load-optim \
--no-load-rng \
--use-checkpoint-args \
--finetune \
--untie-embeddings-and-output-weights \
--normalization RMSNorm \
--position-embedding-type rope \
--no-masked-softmax-fusion \
--attention-softmax-in-fp32 \
--transformer-impl transformer_engine \
--group-query-attention \
--num-query-groups 8 \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--rotary-base 500000 \
--rotary-percent 1.0 \
--use-rope-scaling \
--ffn-hidden-size 14336 \
--num-attention-heads 32 \
--swiglu \
--bf16 \
--disable-bias-linear \
--no-bias-dropout-fusion \
--accumulate-allreduce-grads-in-fp32 \
--attention-softmax-in-fp32 \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model /lpai/models/meta-llama__meta-llama-3_1-8b/24-07-24-0705 \
"

exec tini -- torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
        --seed 5165 \
        --use-distributed-optimizer \
        --log-straggler \
        --use-mcore-models \
        --override-opt_param-scheduler \
        --distributed-timeout-minutes 30 \
        --save $CKPT_DIR \
	    $load_or_resume \
        --data-cache-path $DATA_CACHE_DIR \
        --data-path $DATASET \
        --distributed-backend nccl \
        $MODEL_ARGS \
        --transformer-impl transformer_engine \
        --use-flash-attn \
        --pipeline-model-parallel-size 1 \
        --tensor-model-parallel-size $TP_SIZE \
        --context-parallel-size $CP_SIZE \
        --sequence-parallel \
        --norm-epsilon 1e-5\
        --weight-decay 0.1 \
        --init-method-std 0.02 \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --train-samples 400000000 \
        --lr-warmup-samples 2048000 \
        --lr 2e-3 \
        --min-lr 2e-6 \
        --lr-decay-style cosine \
        --log-interval 10 \
        --eval-iters 2 \
        --eval-interval 500 \
        --split 998,2,0 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --save-interval 500 \
        --num-workers 4 \
        --log-validation-ppl-to-tensorboard \
        --log-world-size-to-tensorboard \
        --tensorboard-dir $TENSORBOARD_DIR \
        --log-timers-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        $INIT_OR_NOT