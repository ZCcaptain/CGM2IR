import argparse
import os
import pickle

import numpy as np
import ujson as json
from prepro import read_docred, read_cdr, read_gda
from transformers import AutoConfig, AutoModel, AutoTokenizer
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--dataset", default="docred", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_pickle", default="train_annotated.pkl", type=str)
    parser.add_argument("--dev_pickle", default="dev.pkl", type=str)
    parser.add_argument("--test_pickle", default="test.pkl", type=str)
    args = parser.parse_args()
   
    if args.dataset == 'docred':
        read = read_docred
    elif args.dataset == 'cdr':
        read = read_cdr
    elif args.dataset == 'gda':
        read = read_gda
    else: 
        raise RuntimeError("No read func for this dataset.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, mirror='tuna'
    )
    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length, type='train')
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length, type='dev')
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length, type='test')

    pickle.dump(train_features, open(os.path.join(args.data_dir, args.train_pickle), 'wb'))
    pickle.dump(dev_features, open(os.path.join(args.data_dir, args.dev_pickle), 'wb'))
    pickle.dump(test_features, open(os.path.join(args.data_dir, args.test_pickle), 'wb'))


if __name__ == "__main__":
    main()
