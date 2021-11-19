python processing.py --data_dir ./dataset/docred \
--dataset docred \
--transformer_type bert \
--model_name_or_path bert-base-cased \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_pickle train_annotated.pkl \
--dev_pickle dev.pkl \
--test_pickle test.pkl \

