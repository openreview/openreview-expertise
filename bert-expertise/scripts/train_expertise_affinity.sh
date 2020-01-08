
train_domains=("american_football" "doctor_who" "fallout" "final_fantasy"
               "military" "pro_wrestling" "starwars" "world_of_warcraft")
eval_domains=("coronation_street" "elder_scrolls" "ice_hockey" "muppets")
test_domains=("forgotten_realms" "lego" "star_trek" "yugioh")

python src/run_mention_affinity.py \
    --data_dir data/zeshel/ \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name mention_affinity \
    --output_dir tmp/zeshel/mention_affinity \
    --do_train \
    --do_eval \
    --do_lower_case \
    --max_seq_length 128 \
    --sample_size 16 \
    --margin 0.5 \
    --sequence_embedding_size 128 \
    --per_gpu_train_batch=1 \
    --per_gpu_eval_batch=2 \
    --num_train_epochs 1.0 \
    --train_domains ${train_domains[@]} \
    --val_domains ${eval_domains[@]}


