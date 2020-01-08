python src/run_expertise_affinity.py \
    --data_dir data/iclr-expertise/ \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name expertise_affinity \
    --output_dir tmp/iclr-expertise/ \
    --do_train \
    --do_eval \
    --do_lower_case \
    --max_seq_length 256 \
    --sample_size 8 \
    --margin 0.5 \
    --sequence_embedding_size 128 \
    --per_gpu_train_batch=1 \
    --per_gpu_eval_batch=2 \
    --num_train_epochs 1.0 \


