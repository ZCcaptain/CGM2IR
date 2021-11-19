CUDA_VISIBLE_DEVICES=0  python train_bio.py --data_dir ./dataset/cdr \
--transformer_type bert \
--model_name_or_path allenai/scibert_scivocab_cased \
--save_path saved_model/pair_cdr.pkt \
--load_path  "" \
--train_pickle train.pkl \
--dev_pickle dev.pkl \
--test_pickle test.pkl \
--train_batch_size 4 \
--test_batch_size 4 \
--gradient_accumulation_steps 1 \
--num_labels 1 \
--learning_rate 2e-5 \
--max_grad_norm 1.0 \
--warmup_ratio 0.06 \
--num_train_epochs 30.0 \
--seed 111 \
--max_mention 30 \
--num_class 2