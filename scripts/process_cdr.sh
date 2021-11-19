python processing.py --data_dir ./dataset/cdr \
--dataset cdr \
--transformer_type bert \
--model_name_or_path allenai/scibert_scivocab_cased \
--train_file train_filter.data \
--dev_file dev_filter.data \
--test_file test_filter.data \
--train_pickle train.pkl \
--dev_pickle dev.pkl \
--test_pickle test.pkl \

