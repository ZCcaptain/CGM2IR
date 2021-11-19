python processing.py --data_dir ./dataset/gda \
--dataset gda \
--transformer_type bert \
--model_name_or_path allenai/scibert_scivocab_cased \
--train_file train.data \
--dev_file dev.data \
--test_file test.data \
--train_pickle train.pkl \
--dev_pickle dev.pkl \
--test_pickle test.pkl \

