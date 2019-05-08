from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.metrics import classification_report
from lstm import LSTM



logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, label_id):
        self.input_ids = input_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class CnewsProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "cnews.train.txt.clean")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "cnews.val.txt.clean")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "cnews.test.txt.clean")), "dev")


    def get_labels(self):
        """See base class."""
        return ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = None
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

def convert_tokens_to_ids(tokens, token_map):
    input_ids = []
    for token in tokens:
        if token not in token_map.keys():
            input_ids.append(token_map['<UNK>'])
        else:
            input_ids.append(token_map[token])
    return input_ids

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 vocab, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}
    token_map = {token : i for i, token in enumerate(vocab)}
    

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = example.text_a.strip().split(' ')
        tokens = tokens[:max_seq_length]
        input_ids = convert_tokens_to_ids(tokens, token_map)


        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        

        assert len(input_ids) == max_seq_length
        

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              label_id=label_id))
    return features



def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "stock":
        return pearson_and_spearman(preds, labels)
    elif task_name == "cnews":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task_name", 
                        default=None, 
                        type=str, 
                        required=True,
                        choices=['cnews'],
                        help="The task name your have named in processors.")
    parser.add_argument("--vocab_path", 
                        default=None, 
                        type=str, 
                        required=True,
                        help="The vocab path.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--test_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for test.")
    parser.add_argument("--is_pretrained",
                        action='store_true',
                        help="Whether to use pretrained model.")
    parser.add_argument("--pretrained_dir",
                        default=None,
                        type=str,
                        help="The pretrained model dir.")
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--input_size",
                        default=128,
                        type=int,
                        help="LSTM input size.")
    parser.add_argument("--hidden_size",
                        default=512,
                        type=int,
                        help="LSTM hidden size.")
    parser.add_argument("--num_layers",
                        default=1,
                        type=int,
                        help="The number of LSTM layers.")
    parser.add_argument("--bias",
                        default=True,
                        type=bool,
                        help="Using LSTM bias or not.")
    parser.add_argument("--batch_first",
                        default=True,
                        type=bool,
                        help="Using batch_first or not.")
    parser.add_argument("--dropout",
                        default=0.2,
                        type=float,
                        help="Using LSTM dropout or not.")
    parser.add_argument("--bidirectional",
                        default=False,
                        type=bool,
                        help="Using BiLSTM or not.")

    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    args = parser.parse_args()

    processors = {
        "cnews": CnewsProcessor,
    }

    output_modes = {
        "cnews": "classification",
    }

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    if not args.no_cuda:
        n_gpu = torch.cuda.device_count()
    else:
        n_gpu = 0

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_test` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = None
    num_labels = None
    if output_mode == 'classification':
        label_list = processor.get_labels()
        num_labels = len(label_list)


    train_examples = None

    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)


    vocab = []
    if os.path.exists(args.vocab_path):
        with open(args.vocab_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                vocab.append(line.strip().split('\t')[0])
    else:
        raise ValueError("vocab_path: {} does noe exist.".format(args.vocab_path))
    # Prepare model
    # Todo 加载预训练好的模型
    if args.is_pretrained:
        if not os.path.exists(args.pretrained_dir):
            raise ValueError("Pretrained dir can't be None.")
        model = torch.load(args.pretrained_dir)
    else:
        if output_mode == 'classification':
            output_size = num_labels
        elif output_mode == 'regression':
            output_size = 1
        model = LSTM(args.input_size, args.hidden_size, len(vocab), output_size, \
            args.num_layers, args.bias, args.batch_first, args.dropout, args.bidirectional)

    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    # Todo  配置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, vocab, output_mode)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_label_ids)
        
        train_sampler = RandomSampler(train_data)
        
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        # model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                # input_ids = input_ids.view(1, input_ids.size(0), input_ids.size(1))
                # print(input_ids.size())
                logits = model(input_ids, no_cuda=args.no_cuda)
                # print(logits.size())

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if args.do_eval:
                eval_examples = processor.get_dev_examples(args.data_dir)
                eval_features = convert_examples_to_features(
                    eval_examples, label_list, args.max_seq_length, vocab, output_mode)
                logger.info("***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)
                all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
               

                if output_mode == "classification":
                    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
                elif output_mode == "regression":
                    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

                eval_data = TensorDataset(all_input_ids, all_label_ids)
                # Run prediction for full data
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

                model.eval()
                eval_loss = 0
                nb_eval_steps = 0
                preds = []

                for input_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = input_ids.to(device)
                    label_ids = label_ids.to(device)

                    with torch.no_grad():
                        logits = model(input_ids, no_cuda=args.no_cuda)

                    # create eval loss and other metric required by the task
                    if output_mode == "classification":
                        loss_fct = CrossEntropyLoss()
                        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                    elif output_mode == "regression":
                        loss_fct = MSELoss()
                        tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
                    
                    eval_loss += tmp_eval_loss.mean().item()
                    nb_eval_steps += 1
                    if len(preds) == 0:
                        preds.append(logits.detach().cpu().numpy())
                    else:
                        preds[0] = np.append(
                            preds[0], logits.detach().cpu().numpy(), axis=0)

                eval_loss = eval_loss / nb_eval_steps
                preds = preds[0]
                if output_mode == "classification":
                    preds = np.argmax(preds, axis=1)
                    print(classification_report(all_label_ids.numpy().tolist(), list(preds.flat)))
                elif output_mode == "regression":
                    preds = np.squeeze(preds)
                result = compute_metrics(task_name, preds, all_label_ids.numpy())
                loss = tr_loss/nb_tr_steps if args.do_train else None

                result['eval_loss'] = eval_loss
                result['global_step'] = global_step
                result['loss'] = loss

                output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
                with open(output_eval_file, "w") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
        # Save a trained model
        torch.save(model, args.output_dir+'/pytorch_model.pt')

    if args.do_test:
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, vocab, output_mode)
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", len(test_examples))
        logger.info("  Batch size = %d", args.test_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
               
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.float)

        test_data = TensorDataset(all_input_ids, all_label_ids)
        # Run prediction for full data
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.test_batch_size)

        model.eval()
        preds = []

        for input_ids, label_ids in tqdm(test_dataloader, desc="Testing"):
            input_ids = input_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, no_cuda=args.no_cuda)

            # create eval loss and other metric required by the task
            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
            elif output_mode == "regression":
                loss_fct = MSELoss()
                    
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
                # 计算某一类概率
                # preds.append(F(logits).detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)
                # preds[0] = np.append(
                    # preds[0], F(logits).detach().cpu().numpy(), axis=0)

        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
            # preds = preds[1:]
        elif output_mode == "regression":
            preds = np.squeeze(preds)


        output_test_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_file, "w") as writer:
            logger.info("***** Writing Test To TEXT *****")
            writer.write('\n'.join([str(i) for i in preds]))   

if __name__ == "__main__":
    main()
