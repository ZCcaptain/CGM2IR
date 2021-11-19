python processing.py --data_dir ./dataset/docred \
--dataset docred \
--transformer_type roberta \
--model_name_or_path roberta-large \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--train_pickle train_annotated_roberta.pkl \
--dev_pickle dev_roberta.pkl \
--test_pickle test_roberta.pkl \

