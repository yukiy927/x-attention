rm -rf output/*
CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python eval/efficiency/attention_speedup.py