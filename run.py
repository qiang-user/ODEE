import argparse
import random

import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader
import torch.nn.functional as F
import config
import utils
import mydata_loader
from eemodel import myModel
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.distributed as dist


def compute_kl_loss(p, q):
    p_loss = F.kl_div(p, q, reduction='none')
    q_loss = F.kl_div(q, p, reduction='none')

    # pad_mask is for seq-level tasks
    # if pad_mask is not None:
    #     p_loss.masked_fill_(pad_mask, 0.)
    #     q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    # p_loss = p_loss.sum()
    # q_loss = q_loss.sum()
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


def compute_dis_loss(x):
    x_mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.sum((x - x_mean) ** 2, dim=-1) / x.size(-1)
    loss = torch.sqrt(var)
    loss = loss.mean()
    return loss


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    https://kexue.fm/archives/7359
    """
    y_true = y_true.float().detach()
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


loss_func = nn.BCEWithLogitsLoss()


def function_loss(pred_logits, true_logits):
    bce_loss = nn.BCELoss(reduction='mean')
    # 确保pred_logits在(0, 1)范围内
    # epsilon = 1e-12
    # pred_logits = torch.clamp(pred_logits, epsilon, 1 - epsilon)
    # 计算批次的平均二元交叉熵损失
    final_loss = bce_loss(pred_logits, true_logits.float())

    return final_loss
    # torch.zeros(batch_size,65,event_idx_len,event_idx_len,4)

    # dimensions=pred_logits.size()
    # batch_size=dimensions[0]
    # 避免log(0)的情况，确保预测值在 (0, 1) 范围内
    # epsilon = 1e-12
    # predictions = torch.clamp(pred_logits, epsilon, 1 - epsilon)
    # 计算损失
    # loss = true_logits.float() * torch.log(pred_logits).cuda() + (1 - true_logits.float()) * torch.log(1 - pred_logits).cuda()
    # loss = -loss  # 由于BCE公式是负数，我们需要取负值
    # 对每个样本的损失进行求和并取平均值
    # 将每个样本的损失加起来，然后除以元素总数
    # loss_per_sample = loss.view(batch_size, -1).mean(dim=1)
    # 最终损失是所有样本损失的平均值
    # final_loss = loss_per_sample.mean()
    # return final_loss


def pad_4d(data, new_data):
    for j, x in enumerate(data):
        new_data[j, :x.shape[0], :x.shape[1], :x.shape[2], :x.shape[3]] = x
    return new_data


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer=optimizer
        # self.scheduler=scheduler

        # bert_params = set(self.model.module.bert.parameters())
        bert_params = set(self.model.bert.parameters())
        # other_params = list(set(self.model.module.parameters()) - bert_params)
        other_params = list(set(self.model.parameters()) - bert_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': config.bert_learning_rate,
             'weight_decay': 0.0},
            {'params': other_params,
             'lr': config.learning_rate,
             'weight_decay': config.weight_decay},
        ]

        self.optimizer = transformers.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = transformers.get_linear_schedule_with_warmup(self.optimizer,
                                                                      num_warmup_steps=warmup_steps,
                                                                      num_training_steps=updates_total)

    def train(self, epoch, data_loader):
        self.model.train()
        loss_list = []
        # 准确率，召回率，c未知
        total_tc_r = 0
        total_tc_p = 0
        total_tc_c = 0

        total_ai_r = 0
        total_ai_p = 0
        total_ai_c = 0

        total_ac_r = 0
        total_ac_p = 0
        total_ac_c = 0

        # overlap = []

        alpha = epoch / config.epochs
        # gamma = gamma ** 2
        for i, data_batch in enumerate(data_loader):
            # data_batch总共有11个元素，分别为inputs, att_mask, word_mask1d,  tri_labels, arg_labels, role_labels, event_idx, _, role_labels_num
            # data_batch = [data.cuda() for data in data_batch[:-2]] + [data_batch[-2], data_batch[-1]]
            data_batch = [data.cuda() for data in data_batch[:-3]] + [data_batch[-3], data_batch[-2]] + [data_batch[-1]]
            inputs, att_mask, word_mask1d, tri_labels, arg_labels, role_labels, event_idx, _, relation_idx = data_batch
            # 你的event_idx一开始没有减去26的话，这步骤很容易忽略，后面就会导致错误。
            final_role_labels = []
            for rel_index, role_label in zip(relation_idx, role_labels):
                # 负采样关系
                final_role_labels.append(torch.index_select(role_label, 0, rel_index.cuda()))
            final_role_labels = torch.stack(final_role_labels).cuda()
            tri_logits, arg_logits, role_logits = model(inputs, att_mask, word_mask1d, tri_labels, arg_labels,
                                                        role_labels, event_idx)
            # 一个句子的长度
            batch_size, L = word_mask1d.size()
            # tri_loss = function_loss(tri_logits, tri_labels)
            tri_loss = loss_func(tri_logits.reshape(-1), tri_labels.reshape(-1).float())
            # arg_loss = function_loss(arg_logits, arg_labels)
            arg_loss = loss_func(arg_logits.reshape(-1), arg_labels.float().reshape(-1))
            select_true_role = []
            # 写代码不规范问题  上面循环已经使用i，怎么还能再次使用呢？
            # for i in range(batch_size):
            for j in range(batch_size):
                select_true_role.append(final_role_labels[j][:, event_idx[j]][:, :, event_idx[j]])
                # select_true_role.append(role_labels[j][:, event_idx[j]][:, :, event_idx[j]])
            selected_role_labels = torch.stack(select_true_role).cuda()
            # role_loss = function_loss(role_logits, selected_role_labels)

            final_role_logits = []
            for rel_index, role_logit in zip(relation_idx, role_logits):
                # 负采样关系
                final_role_logits.append(torch.index_select(role_logit, 0, rel_index.cuda()))
            role_logits = torch.stack(final_role_logits)
            role_loss = loss_func(role_logits, selected_role_labels.float())

            loss = 0.1 * tri_loss + 0.01 * arg_loss + role_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            loss_list.append(loss.cpu().item())

            # 这块注意：训练集在计算损失函数时，损失函数自带的sifmoid,这块你应该带上sigmoid
            tri_outputs = torch.sigmoid(tri_logits).reshape(-1) > 0.5
            tri_outputs = tri_outputs.cuda()
            # 这块到底对不对呢，有待考察
            total_tc_r += tri_labels.long().sum().item()
            total_tc_p += tri_outputs.sum().item()
            total_tc_c += (tri_outputs.long() + tri_labels.reshape(-1).long()).eq(2).sum().item()

            arg_outputs = torch.sigmoid(arg_logits).reshape(-1) > 0.5
            arg_outputs = arg_outputs.cuda()
            total_ai_r += arg_labels.long().sum().item()
            total_ai_p += arg_outputs.sum().item()
            total_ai_c += (arg_outputs.long() + arg_labels.reshape(-1).long()).eq(2).sum().item()

            role_outputs = torch.sigmoid(role_logits) > 0.5
            role_outputs = role_outputs.cuda()
            total_ac_r += selected_role_labels.sum().item()
            total_ac_p += role_outputs.sum().item()
            total_ac_c += (role_outputs + selected_role_labels.long()).eq(2).sum().item()
            self.scheduler.step()
            torch.cuda.empty_cache()
            # 在每个 batch 处理完成后调用 torch.cuda.empty_cache()
        tri_f1, tri_r, tri_p = utils.calculate_f1(total_tc_r, total_tc_p, total_tc_c)
        arg_f1, arg_r, arg_p = utils.calculate_f1(total_ai_r, total_ai_p, total_ai_c)
        role_f1, role_r, role_p = utils.calculate_f1(total_ac_r, total_ac_p, total_ac_c)

        table = pt.PrettyTable(["Train {}".format(epoch), "Loss", "Tri F1", "Arg F1", "Role F1"])
        table.add_row(["Label", "{:.4f}".format(np.mean(loss_list))] +
                      ["{:3.4f}".format(x) for x in [tri_f1, arg_f1, role_f1]])
        logger.info("\n{}".format(table))
        # print(np.mean(overlap))
        # print(np.mean(loss2_list))
        # print(np.mean(loss3_list))
        # print(np.mean(loss4_list))
        # print(np.mean(loss5_list))

        return tri_f1 + arg_f1

    def eval(self, epoch, data_loader, is_test=False):
        self.model.eval()

        total_results = {k + "_" + t: 0 for k in ["ti", "tc", "ai", "ac"] for t in ["r", "p", "c"]}
        with torch.no_grad():
            for i, data_batch in enumerate(data_loader):
                # data_batch = [data.cuda() for data in data_batch[:-2]] + [data_batch[-2], data_batch[-1]]
                data_batch = [data.cuda() for data in data_batch[:-3]] + [data_batch[-3], data_batch[-2]] + [
                    data_batch[-1]]
                inputs, att_mask, word_mask1d, tri_labels, arg_labels, role_labels, event_idx, tuple_labels, relation_idx = data_batch
                results = model(inputs, att_mask, word_mask1d, tri_labels, arg_labels, role_labels, 0)
                results = utils.mydecode(results, tuple_labels, config)
                # results = utils.final_decode(results, config)
                # results = utils.decode(results, tuple_labels, config.tri_args)
                for key, value in results.items():
                    total_results[key] += value
        ti_f1, ti_r, ti_p = utils.calculate_f1(total_results["ti_r"], total_results["ti_p"], total_results["ti_c"])
        tc_f1, tc_r, tc_p = utils.calculate_f1(total_results["tc_r"], total_results["tc_p"], total_results["tc_c"])
        ai_f1, ai_r, ai_p = utils.calculate_f1(total_results["ai_r"], total_results["ai_p"], total_results["ai_c"])
        ac_f1, ac_r, ac_p = utils.calculate_f1(total_results["ac_r"], total_results["ac_p"], total_results["ac_c"])
        title = "EVAL" if not is_test else "TEST"

        table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
        table.add_row(["Trigger I"] + ["{:3.4f}".format(x) for x in [ti_f1, ti_p, ti_r]])
        table.add_row(["Trigger C"] + ["{:3.4f}".format(x) for x in [tc_f1, tc_p, tc_r]])
        table.add_row(["Argument I"] + ["{:3.4f}".format(x) for x in [ai_f1, ai_p, ai_r]])
        table.add_row(["Argument C"] + ["{:3.4f}".format(x) for x in [ac_f1, ac_p, ac_r]])

        logger.info("\n{}".format(table))
        return (ti_f1 + ai_f1 + tc_f1 + ac_f1) / 4

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))


if __name__ == '__main__':
    # 参数运行---开始训练--冲--
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/fewfc.json')
    # 这块把default=1改为了0
    parser.add_argument('--device', type=int, default=0)
    # parser.add_argument('--device', nargs='+', type=int, default=[i for i in range(torch.cuda.device_count())],
    #                     help='Indices of CUDA devices to use')
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--tri_hid_size', type=int)
    parser.add_argument('--eve_hid_size', type=int)
    parser.add_argument('--arg_hid_size', type=int)
    parser.add_argument('--node_type_size', type=int)
    parser.add_argument('--event_sample', type=int)
    parser.add_argument('--layers', type=int)

    parser.add_argument('--dropout', type=float)
    parser.add_argument('--graph_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--warm_epochs', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--grad_clip_norm', type=float)
    parser.add_argument('--gamma', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)

    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    # Python的实例对象在运行过程中是可以动态添加属性的
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # 这些代码的作用是设置随机数种子，禁用一些可能导致结果不一致的优化，并确保模型训练的可重现性，
    # 本次实验没有使用
    if config.seed >= 0:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    logger.info("Loading Data")
    # 对于数据集的处理过程
    datasets = mydata_loader.load_data(config)

    # 调试注意num_workers=2问题
    # debug修改num_workers=2为0,运行时可以恢复为2,训练集打乱数据，且丢弃最后一个不足批量大小的数据
    # num_workers控制着在数据加载过程中使用多少个子进程来预先加载数据，以加速数据加载过程

    train_loader, dev_loader, test_loader = (
        DataLoader(dataset=dataset,
                   batch_size=config.batch_size,
                   collate_fn=mydata_loader.my_collate_fn,
                   shuffle=i == 0,
                   num_workers=0,
                   drop_last=i == 0)
        for i, dataset in enumerate(datasets)
    )

    updates_total = len(datasets[0]) // config.batch_size * config.epochs
    warmup_steps = config.warm_epochs * len(datasets[0])

    model = myModel(config)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, len(datasets[0]) * 1, 1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.999)

    model = model.cuda()

    trainer = Trainer(model)
    # trainer = Trainer(model, optimizer, scheduler)
    best_f1 = 0
    best_test_f1 = 0
    for i in range(config.epochs):
        logger.info("Epoch: {}".format(i))
        trainer.train(i, train_loader)

        # 在每个 epoch 结束后调用 torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        if i >= 5:
            f1 = trainer.eval(i, dev_loader)
            test_f1 = trainer.eval(i, test_loader, is_test=True)
            if f1 > best_f1:
                best_f1 = f1
                best_test_f1 = test_f1
                trainer.save("model.pt")
    logger.info("Best DEV F1: {:3.4f}".format(best_f1))
    logger.info("Best TEST F1: {:3.4f}".format(best_test_f1))
    trainer.load("model.pt")
    trainer.eval("Final", test_loader, True)
