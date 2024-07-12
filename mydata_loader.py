import json
import os
import random
import numpy as np
import prettytable as pt
import torch
import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import defaultdict
import math

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# dis2idx = np.zeros((1000), dtype="int64")
# dis2idx[1] = 1
# dis2idx[2:] = 2
# dis2idx[4:] = 3
# dis2idx[8:] = 4
# dis2idx[16:] = 5
# dis2idx[32:] = 6
# dis2idx[64:] = 7
# dis2idx[128:] = 8
# dis2idx[256:] = 9
# mask_token_id = -1
# neg_num = 3

SAMPLE_NUM = 0


class Vocabulary(object):
    PAD = "<pad>"
    UNK = "<unk>"
    ARG = "<arg>"

    def __init__(self):
        self.tri_label2id = {}  # trigger
        self.tri_id2label = {}
        # defaultdict(int)的作用是创建一个字典，其中如果访问字典中不存在的键，则会返回一个默认值0
        self.tri_id2count = defaultdict(int)
        # 触发词对应的id的概率，10个触发词都是0.1
        self.tri_id2prob = {}

        self.rol_label2id = {}  # role
        self.rol_id2label = {}

        self.eventTypeinputs = ""
        self.eventTypeLength = 0
        self.triAndArgRelationLen = 0

        self.triAndArgRelation = {}

        self.new_tri_labels = {}

    def relation2id(self, type, role, ):
        id = 99
        for key, value in self.triAndArgRelation.items():
            if (type, role) == value:
                id = key
        if id == 99:
            return Exception("Wrong Relation!")
        else:
            return id

    # 触发词或者角色对应的标签转化为id
    def label2id(self, label, type):
        label = label.lower()
        if type == "tri":
            return self.tri_label2id[label]
        elif type == "rol":
            return self.rol_label2id[label]
        else:
            raise Exception("Wrong Label Type!")

    def add_label(self, label, type):
        label = label.lower()

        if type == "tri":
            if label not in self.tri_label2id:
                # 触发词的数量只会访问一次
                self.tri_label2id[label] = len(self.tri_id2label)
                self.tri_id2label[self.tri_label2id[label]] = label
                self.tri_id2count[self.tri_label2id[label]] += 1
        elif type == "rol":
            if label not in self.rol_label2id:
                self.rol_label2id[label] = len(self.rol_id2label)
                self.rol_id2label[self.rol_label2id[label]] = label
        else:
            raise Exception("Wrong Label Type!")

    def get_prob(self):
        total = np.sum(list(self.tri_id2count.values()))
        for k, v in self.tri_id2count.items():
            self.tri_id2prob[k] = v / total

    @property
    def tri_label_num(self):
        return len(self.tri_label2id)

    @property
    def rol_label_num(self):
        return len(self.rol_label2id)

    @property
    def label_num(self):
        return self.tri_label_num


def my_collate_fn(data):
    inputs, att_mask, word_mask1d, tri_labels, arg_labels, role_labels, tuple_labels, event_list, relation_list, training = map(
        list, zip(*data))
    batch_size = len(inputs)
    # 一个batchsize中句子最大的输入长度
    max_tokens = np.max([x.shape[0] for x in word_mask1d])

    # 用0进行填充
    inputs = pad_sequence(inputs, True)
    att_mask = pad_sequence(att_mask, True)
    word_mask1d = pad_sequence(word_mask1d, True)

    def pad_2d(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    def pad_3d(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :, :x.shape[1], :x.shape[2]] = x
        return new_data

    def pad_4d(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :, :x.shape[1], :x.shape[2], :] = x
        return new_data

    # 这里的max_tokens包括事件类型拼成的句子和原来的句子的长度。本来是只需要句子的长度即可。
    tri_mat = torch.zeros((batch_size, tri_labels[0].size(0), max_tokens - 26), dtype=torch.bool)
    # 8*26*223 8 batch_size 26是事件类型拼成的句子 223是这8句话中的最大的token长度
    tri_labels = pad_2d(tri_labels, tri_mat)
    arg_mat = torch.zeros((batch_size, arg_labels[0].size(0), max_tokens - 26), dtype=torch.bool)
    # 8*18*223
    arg_labels = pad_2d(arg_labels, arg_mat)
    role_mat = torch.zeros(
        (batch_size, role_labels[0].size(0), max_tokens - 26, max_tokens - 26, role_labels[0].size(-1)),
        dtype=torch.bool)
    # 8*65*223*223*4
    role_labels = pad_4d(role_labels, role_mat)

    _tuple_labels = {k: set() for k in ["ti", "tc", "ai", "ac"]}
    if not training[0]:
        for i, x in enumerate(tuple_labels):
            for k, v in x.items():
                _tuple_labels[k] = _tuple_labels[k] | set([(i,) + t for t in x[k]])
    # todo event_idx[i]中的所有元素应该排个顺序吧。
    # 其实这布操作只有训练集用得到

    event_idx = []
    final_event_idx = []
    if training[0] == True:
        L = inputs.size(1) - 26 - 2
        for b in range(inputs.size(0)):
            pos_event, neg_events = event_list[b]
            # todo 最好改为句子的长度
            # 正token的长度是53（采样后维度是53）,NSLength的长度是52，(正token不足52的采样后维度是52)，所以会维度报错
            # 如果把NSLength多加5呢？或者维度扩一点呢？
            NSLength = math.ceil(L * 0.4)
            neg_token_num = NSLength - len(pos_event)
            neg_list = random.choices(neg_events, k=neg_token_num)
            event_idx.append(sorted(pos_event + neg_list))
            # event_idx.append(pos_event + neg_list)
        # event_idx最长的那个样本
        max_length = np.max([len(x) for x in event_idx])
        # 按照event_idx最长的那个样本再扩充一遍
        for i, x in enumerate(event_idx):
            pos_event, neg_events = event_list[i]
            if len(x) < max_length:
                neg_token_num = max_length - len(x)
                neg_list = random.choices(neg_events, k=neg_token_num)
                # neg_list = random.sample(neg_events, neg_token_num)
                final_event_idx.append(sorted(x + neg_list))
            else:
                final_event_idx.append(x)
    # 同理，只有训练集用得到--是不是可以模仿event_idx呢?
    relation_idx = []
    if training[0] == True:
        sample_relation_num = 15
        for b in range(inputs.size(0)):
            pos_relation_list, neg_relation_list = relation_list[b]
            pos_rel_num = len(pos_relation_list)
            neg_rel_num = sample_relation_num - pos_rel_num
            # oneEE和ODRTE使用的随机负采样函数不一样
            sam_neg_rel_list = random.sample(neg_relation_list, neg_rel_num)
            sam_rel_index_list = (pos_relation_list + sam_neg_rel_list)
            sam_rel_index_list.sort()
            relation_idx.append(torch.tensor(sam_rel_index_list))

    return inputs, att_mask, word_mask1d, tri_labels, arg_labels, role_labels, final_event_idx, _tuple_labels, relation_idx


class myRelationDataset(Dataset):
    def __init__(self, inputs, att_mask, word_mask1d, tri_labels, arg_labels,
                 role_labels, gold_tuples, event_list, relation_list):
        self.inputs = inputs
        self.att_mask = att_mask
        self.word_mask1d = word_mask1d
        self.tri_labels = tri_labels
        self.arg_labels = arg_labels
        self.role_labels = role_labels
        self.tuple_labels = gold_tuples
        self.event_list = event_list
        self.relation_list = relation_list
        self.training = True

    def eval_data(self):
        self.training = False

    def __getitem__(self, item):
        return torch.LongTensor(self.inputs[item]), \
               torch.LongTensor(self.att_mask[item]), \
               torch.BoolTensor(self.word_mask1d[item]), \
               torch.BoolTensor(self.tri_labels[item]), \
               torch.BoolTensor(self.arg_labels[item]), \
               torch.BoolTensor(self.role_labels[item]), \
               self.tuple_labels[item], \
               self.event_list[item], \
               self.relation_list[item], \
               self.training

    def __len__(self):
        return len(self.inputs)


# 论文中的触发词、论元，角色打标签方式，进行数据预处理的
# 训练时，一个文本有多个事件，模型会分开训练
def my_process_bert(data, tokenizer, vocab):
    inputs = []
    att_mask = []
    word_mask1d = []
    arg_labels = []
    tri_labels = []
    role_labels = []
    gold_tuples = []
    event_list = []
    role_list = []

    relation_list = []

    eventTypeInputs = vocab.eventTypeinputs
    for ins_id, instance in tqdm.tqdm(enumerate(data), total=len(data)):
        # 将输入文本转化为token--cls=101---sep=102
        _inputs = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
            [x for x in eventTypeInputs.lower()]) + tokenizer.convert_tokens_to_ids(
            [x for x in instance["content"].lower()]) + [tokenizer.sep_token_id]
        length = len(_inputs) - 2
        sentenceLength = len(instance["content"])
        _word_mask1d = np.array([1] * length)
        total_event_set = set([i for i in range(sentenceLength)])
        total_relation_set = set([i for i in range(vocab.triAndArgRelationLen)])
        # 触发词类型、间距矩阵
        _tri_labels = np.zeros((vocab.eventTypeLength, sentenceLength), dtype=bool)
        # 论元间距、角色矩阵
        _arg_labels = np.zeros((vocab.rol_label_num, sentenceLength), dtype=bool)
        # _role_labels = np.zeros((vocab.triAndArgRelationLen, sentenceLength, sentenceLength, 4), dtype=bool)
        _role_labels = np.zeros((vocab.triAndArgRelationLen, sentenceLength, sentenceLength, 4), dtype=bool)
        _att_mask = np.array([1] * len(_inputs))
        event_set = set()
        _gold_tuples = {k: set() for k in ["ti", "tc", "ai", "ac"]}
        events = instance["events"]
        _role_list = []
        position_token_set = set()
        position_relation_set = set()
        for event in events:
            trigger = event["trigger"]
            t_s, t_e = trigger["span"]
            t_e = t_e - 1
            for i in range(t_s, t_e + 1):
                position_token_set.add(i)
            event_type = vocab.label2id(event["type"], "tri")
            event_type_s = vocab.eventTypeinputs.index(event["type"])
            event_type_e = vocab.eventTypeinputs.index(event["type"]) + len(event["type"])
            # 填充触发词矩阵
            _tri_labels[event_type_s:event_type_e, t_s:t_e + 1] = 1
            # _tri_labels[event_type_s:event_type_e, t_s] = 1
            # _tri_labels[event_type_s:event_type_e, t_e] = 1
            # _tri_labels[event_type_s, t_s] = 1
            # _tri_labels[event_type_s, t_e] = 1
            # _tri_labels[event_type_e-1, t_s] = 1
            # _tri_labels[event_type_e-1, t_e] = 1

            _gold_tuples["ti"].add((t_s, t_e))
            _gold_tuples["tc"].add((t_s, t_e, event_type))
            event_set.add(event_type)
            args = event["args"]
            for k, v in args.items():
                for arg in v:
                    a_s, a_e = arg["span"]
                    a_e = a_e - 1
                    for i in range(a_s, a_e + 1):
                        position_token_set.add(i)
                    role = vocab.label2id(k, "rol")
                    _role_list.append(role)
                    # 修改
                    # _arg_labels[role, a_s] = 1
                    # _arg_labels[role, a_e] = 1
                    _arg_labels[role, a_s:a_e + 1] = 1
                    relationId = -1
                    for key, value in vocab.triAndArgRelation.items():
                        if (event_type, role) == value:
                            relationId = key
                    if relationId not in position_relation_set:
                        position_relation_set.add(relationId)
                    _role_labels[relationId, t_s, a_s, 0] = 1
                    _role_labels[relationId, t_s, a_e, 1] = 1
                    _role_labels[relationId, t_e, a_e, 2] = 1
                    _role_labels[relationId, t_e, a_s, 3] = 1

                    # _role_labels[role, t_s, a_s, 0] = 1
                    # _role_labels[role, t_s, a_e, 1] = 1
                    # _role_labels[role, t_e, a_s, 2] = 1
                    # _role_labels[role, t_e, a_e, 3] = 1

                    _gold_tuples["ai"].add((a_s, a_e, event_type))
                    _gold_tuples["ac"].add((a_s, a_e, event_type, role))

        negative_token_list = list(total_event_set - position_token_set)
        negative_relation_list = list(total_relation_set - position_relation_set)

        inputs.append(_inputs)
        att_mask.append(_att_mask)
        word_mask1d.append(_word_mask1d)
        arg_labels.append(_arg_labels)
        tri_labels.append(_tri_labels)
        role_labels.append(_role_labels)
        gold_tuples.append(_gold_tuples)
        event_list.append((list(position_token_set), negative_token_list))
        role_list.append(_role_list)
        relation_list.append((list(position_relation_set), negative_relation_list))
    # return inputs, att_mask, word_mask1d, tri_labels, arg_labels, role_labels, gold_tuples, event_list
    return inputs, att_mask, word_mask1d, tri_labels, arg_labels, role_labels, gold_tuples, event_list, relation_list


def fill_vocab(vocab, dataset):
    statistic = {"tri_num": 0, "arg_num": 0}
    for instance in dataset:
        events = instance["events"]
        for eve in events:
            vocab.add_label(eve["type"], "tri")
            args = eve["args"]
            for k, v in args.items():
                vocab.add_label(k, "rol")
            statistic["arg_num"] += len(args)
        statistic["tri_num"] += len(events)
    return statistic


def load_data(config):
    # 指的是事件类型的个数
    global EVENT_NUN
    # 采样个数，还不是很理解---oneEE模型能并行处理的关键
    global SAMPLE_NUM
    with open("./data/{}/new_train.json".format(config.dataset), "r", encoding="utf-8") as f:
        train_data = [json.loads(x) for x in f.readlines()]
    with open("./data/{}/new_dev.json".format(config.dataset), "r", encoding="utf-8") as f:
        dev_data = [json.loads(x) for x in f.readlines()]
    with open("./data/{}/new_test.json".format(config.dataset), "r", encoding="utf-8") as f:
        test_data = [json.loads(x) for x in f.readlines()]

    # train_data = train_data + dev_data
    # dev_data = test_data
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")

    config.tokenizer = tokenizer
    vocab = Vocabulary()
    train_statistic = fill_vocab(vocab, train_data)
    vocab.get_prob()
    dev_statistic = fill_vocab(vocab, dev_data)
    test_statistic = fill_vocab(vocab, test_data)

    with open("./data/{}/ty_args.json".format(config.dataset), "r", encoding="utf-8") as f:
        tri_args = json.load(f)
    # config.tri_args把事件类型跟论元的id联系起来，例如（0，1）代表第一个事件类型的第一个论元
    config.tri_args = set()

    for k, vs in tri_args.items():
        for v in vs:
            k_i, v_i = vocab.label2id(k, "tri"), vocab.label2id(v, "rol")
            config.tri_args.add((k_i, v_i))
    for i, value in enumerate(config.tri_args):
        vocab.triAndArgRelation[i] = value
    vocab.triAndArgRelationLen = len(config.tri_args)

    # with open("./data/fewFC/new_dev.json", 'r', encoding='utf-8') as file:
    #     content = [json.loads(data) for data in file.readlines()]
    # c=[]
    # for i in content:
    #     r=[]
    #     count=0
    #     events=i['events']
    #     for event in events:
    #         type=vocab.label2id(event['type'], "tri")
    #         args=list(event['args'].keys())
    #         for arg in args:
    #             ar=vocab.label2id(arg, "rol")
    #             for key,value in vocab.triAndArgRelation.items():
    #                 if(type,ar)==value:
    #                     if value not in r:
    #                         r.append(value)
    #                         count+=1
    #     c.append(count)

    eventTypeList = list(tri_args.keys())
    eventinputs = ""
    for i in eventTypeList:
        eventinputs += i
    vocab.eventTypeinputs = eventinputs
    vocab.eventTypeLength = len(eventinputs)
    new_tri_labels = {}
    for i in eventTypeList:
        start_index = eventinputs.find(i)
        end_index = start_index + len(i)
        new_tri_labels[i] = []
        for j in range(start_index, end_index):
            new_tri_labels[i].append(j)
    vocab.new_tri_labels = new_tri_labels
    # 这行代码使用了 PrettyTable 库创建了一个表格对象（PrettyTable），其中包含了指定的列标签和空行
    table = pt.PrettyTable([config.dataset, "#sentence", "#event", "#argument"])
    table.add_row(["train", len(train_data)] + [train_statistic[key] for key in ["tri_num", "arg_num"]])
    table.add_row(["dev", len(dev_data)] + [dev_statistic[key] for key in ["tri_num", "arg_num"]])
    table.add_row(["test", len(test_data)] + [test_statistic[key] for key in ["tri_num", "arg_num"]])
    config.logger.info("\n{}".format(table))

    config.tri_label_num = vocab.tri_label_num
    config.rol_label_num = vocab.rol_label_num
    config.label_num = vocab.tri_label_num
    config.vocab = vocab

    EVENT_NUN = config.tri_label_num
    SAMPLE_NUM = config.event_sample

    print("Processing train data...")
    # process_bert(train_data, tokenizer, vocab) 是一个函数调用，它返回了一个元组或其他可迭代对象，*操作
    # 用于将这个元组或可迭代对象解包，将其中的元素作为参数传递给 RelationDataset 构造器
    train_dataset = myRelationDataset(*my_process_bert(train_data, tokenizer, vocab))
    print("Processing dev data...")
    dev_dataset = myRelationDataset(*my_process_bert(dev_data, tokenizer, vocab))
    print("Processing test data...")
    test_dataset = myRelationDataset(*my_process_bert(test_data, tokenizer, vocab))
    dev_dataset.eval_data()
    test_dataset.eval_data()
    return train_dataset, dev_dataset, test_dataset


def load_lexicon(emb_path, vocab, emb_dim=50):
    emb_dict = load_pretrain_emb(emb_path)
    embed_size = emb_dim
    scale = np.sqrt(3.0 / emb_dim)
    embedding = np.random.uniform(-scale, scale, (len(vocab.word2id) + len(emb_dict), embed_size))

    for k, v in emb_dict.items():
        k = k.lower()
        index = len(vocab.word2id)
        vocab.word2id[k] = index
        vocab.id2word[index] = k

        embedding[index, :] = v
    embedding[0] = np.zeros(embed_size)
    embedding = torch.FloatTensor(embedding)
    return embedding


def load_embedding(emb_path, emb_dim, vocab):
    wvmodel = load_pretrain_emb(emb_path)
    embed_size = emb_dim
    scale = np.sqrt(3.0 / emb_dim)
    embedding = np.random.uniform(-scale, scale, (len(vocab), embed_size))
    hit = 0
    for token, i in vocab.items():
        if token in wvmodel:
            hit += 1
            embedding[i, :] = wvmodel[token]
    print("File: {} Total hit: {} rate {:.4f}".format(emb_path, hit, hit / len(vocab)))
    embedding[0] = np.zeros(embed_size)
    embedding = torch.FloatTensor(embedding)
    return embedding


def load_pretrain_emb(embedding_path):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, "r", encoding="utf-8") as file:
        for line in tqdm.tqdm(file):
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split()
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                assert (embedd_dim + 1 == len(tokens))
            embedd = np.empty([1, embedd_dim])
            embedd[:] = tokens[1:]
            embedd_dict[tokens[0]] = embedd
    return embedd_dict
