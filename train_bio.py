import argparse
import os
import pickle
import numpy as np
import torch
from apex import amp
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_cdr, read_gda
import wandb

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            model.zero_grad()
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          'ht_visible_mask': batch[5]
                          }
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                # with amp.scale_loss(loss, optimizer) as scaled_loss:
                #     scaled_loss.backward()
                loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        # torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                wandb.log({"loss": loss.item()}, step=num_steps)
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    test_score, test_output = evaluate(args, model, test_features, tag="test")
                    if test_score > best_score:
                        best_score = test_score
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)
                    test_output["best_score"] = best_score
                    test_output['epoch'] = epoch
                    # dev_output['description'] = '1'
                    print(dev_output)
                    print(test_output)
                    wandb.log(dev_output, step=num_steps)
                    wandb.log(test_output, step=num_steps)
                    

        return num_steps

    no_decay = ['bias', 'gamma', 'beta', "LayerNorm.weight"]
    new_layer = ["extractor", "bilinear", "pair_graph", "embedding", "predict", 'query', 'pair_pos']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer) and not any(nd in n for nd in no_decay)], "lr":2e-5, 'weight_decay_rate': 0.01},
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer) and any(nd in n for nd in no_decay)], "lr":2e-5, 'weight_decay_rate': 0.0},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)


def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds, golds = [], []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'ht_visible_mask': batch[5]
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            golds.append(np.concatenate([np.array(label, np.float32) for label in batch[2]], axis=0))

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    golds = np.concatenate(golds, axis=0).astype(np.float32)

    tp = ((preds[:, 1] == 1) & (golds[:, 1] == 1)).astype(np.float32).sum()
    tn = ((golds[:, 1] == 1) & (preds[:, 1] != 1)).astype(np.float32).sum()
    fp = ((preds[:, 1] == 1) & (golds[:, 1] != 1)).astype(np.float32).sum()
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + tn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    output = {
        "{}_f1".format(tag): f1 * 100,
    }
    return f1, output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/cdr", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="allenai/scibert_scivocab_cased", type=str)

    parser.add_argument("--train_file", default="train_filter.data", type=str)
    parser.add_argument("--dev_file", default="dev_filter.data", type=str)
    parser.add_argument("--test_file", default="test_filter.data", type=str)

    parser.add_argument("--train_pickle", default="train.pkl", type=str)
    parser.add_argument("--dev_pickle", default="dev.pkl", type=str)
    parser.add_argument("--test_pickle", default="test.pkl", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=1, type=int,
                        help="Max number of labels in the prediction.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=111,
                        help="random seed for initialization.")
    parser.add_argument("--num_class", type=int, default=2,
                        help="Number of relation types in dataset.")
    parser.add_argument("--max_mention", type=int, default=30,
                        help="log.")

    args = parser.parse_args()
    wandb.init(project="CDR")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    read = read_cdr if "cdr" in args.data_dir else read_gda

    # train_file = os.path.join(args.data_dir, args.train_file)
    # dev_file = os.path.join(args.data_dir, args.dev_file)
    # test_file = os.path.join(args.data_dir, args.test_file)
    # train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
    # dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    # test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)
    train_pkl = os.path.join(args.data_dir, args.train_pickle)
    dev_pkl = os.path.join(args.data_dir, args.dev_pickle)
    test_pkl = os.path.join(args.data_dir, args.test_pickle)
    train_features = pickle.load(open(train_pkl, 'rb'))
    dev_features = pickle.load(open(dev_pkl, 'rb'))
    test_features = pickle.load(open(test_pkl, 'rb'))
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model, num_labels=args.num_labels, num_class=args.num_class, max_mention=args.max_mention)
    model.to(0)

    if args.load_path == "":
        train(args, model, train_features, dev_features, test_features)
    else:
        # model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path))
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        print(dev_output)
        print(test_output)


if __name__ == "__main__":
    main()
