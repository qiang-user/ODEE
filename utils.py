import logging
import time
import pickle
from transformers import AutoTokenizer
import config
import json
import numpy as np
from collections import defaultdict
from itertools import groupby
import torch

# 将日志信息记录到控制台和文件中
def get_logger(dataset):
    pathname = "./log/{}_{}.txt".format(dataset, time.strftime("%m-%d_%H-%M-%S"))
    # 这三行代码是日志记录器的基本配置
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')

    # 将日志消息记录到pathname文件中
    file_handler = logging.FileHandler(pathname)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # 将日志信息记录到标准输出流
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def save_file(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_file(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


# outputs可以获取最终的结果---预测结果
def decode(outputs, labels, tri_args):
    results = {}
    arg_dict = {}

    for key in ["ti", "tc", "ai", "ac"]:
        pred = outputs[key]
        if pred is None:
            pred_set = set()
        else:
            pred_set = set([tuple(x.tolist()) for x in pred])
        if key == "ai":
            # b,x,y,e的意思：第几个样本、论元开始、论元结束、事件类型。---总结例如第一个样本的某一个事件类型
            # 的某一个论元，
            for b, x, y, e in pred_set:
                for c in range(x, y + 1):
                    if (b, c) in arg_dict:
                        # 论元中的一个字与这个论元的起始和结束挂钩---
                        arg_dict[(b, c)].append((b, x, y))
                    else:
                        arg_dict[(b, c)] = [(b, x, y)]
        if key in ["ac"]:
            # 预测的有些e,r不合理
            new_pred_set = set()
            # 第几个样本、论元的一个中文字或者一个英文单词 、事件类型、论元在这个事件类型中扮演的角色
            for b, x, e, r in pred_set:
                if (b, x) in arg_dict:
                    for prefix in arg_dict[(b, x)]:
                        new_pred_set.add(prefix + (e, r))
            # 一些事件类型的r角色是固定好的，预测的有些e,r有些可能是错误的，所以pred_set的结果为第几个样本、论元开始、论元结束、事件类型、论元扮演的角色
            pred_set = set([x for x in new_pred_set if (x[-2], x[-1]) in tri_args])
        # ti_r:8个样本中真实的触发词标签的数量  ti_p：8个样本中预测的触发词的数量
        results[key + "_r"] = len(labels[key])
        results[key + "_p"] = len(pred_set)
        results[key + "_c"] = len(pred_set & labels[key])

        # 存储最终的预测结果
        # results[key] = pred_set

    return results


# def mydecode(outputs, labels, config):
#     finaloutputs = {}
#     tri_args = config.tri_args
#     result = {}
#     finaloutputs['ti'] = new_decode_tri(outputs)
#     # 是不是tc解码出来了，ti就出来了呢？
#     finaloutputs['tc'] = docode_event(outputs, config)
#
#     result['ti'] = finaloutputs['ti']
#     result['tc'] = finaloutputs['tc']
#     finaloutputs['ai'] = decode_arg(outputs, config)
#     result['ai'] = finaloutputs['ai']
#     result['ac'] = set()
#     finaloutputs['finalai'], finaloutputs['ac'] = new_decode(outputs,  config)
#     ac = finaloutputs['ac']
#     for i in ac:
#         triAndRole = (i[-2], i[-1])
#         if triAndRole in tri_args:
#             result['ac'].add(i)
#     finalresults = {}
#     for key in ["ti", "tc", "ai", "ac"]:
#         finalresults[key + "_r"] = len(labels[key])
#         finalresults[key + "_p"] = len(result[key])
#         finalresults[key + "_c"] = len(result[key] & labels[key])
#     return finalresults

def mydecode(outputs, labels, config):
    role_logits, word_mask = outputs
    batchsize = role_logits.size(0)
    tiresult = set()
    tcresult = set()
    airesult = set()
    acresult = set()
    for i in range(batchsize):
        text = role_logits[i]
        mask = word_mask[i]
        sample_index = i
        temptiresult, temptcresult, tempairesult, tempacresult = final_decode(sample_index, text, mask, config)
        tiresult = tiresult | temptiresult
        tcresult = tcresult | temptcresult
        airesult = airesult | tempairesult
        acresult = acresult | tempacresult

    # tiresult, tcresult, airesult, acresult = new_decode(outputs, config)
    result = {}
    result['ti'] = tiresult
    result['tc'] = tcresult
    result['ai'] = airesult
    result['ac'] = acresult

    finalresults = {}
    for key in ["ti", "tc", "ai", "ac"]:
        finalresults[key + "_r"] = len(labels[key])
        finalresults[key + "_p"] = len(result[key])
        finalresults[key + "_c"] = len(result[key] & labels[key])
    return finalresults


def final_decode(sample_index, outputs, mask, config):
    tiresult1 = set()
    tcresult1 = set()
    airesult1 = set()
    acresult1 = set()
    # shaking_tag_negvalue = shaking_tag[:,:,:,3,None].expand(-1,-1,-1,3)
    # shaking_tag = (shaking_tag[:,:,:,:-1]>shaking_tag_negvalue).long()
    shaking_tag_temp = outputs
    # 216*102*102*3
    # shaking_tag = ((outputs[:,:,:,:-1]>0.5).long()+mask[:,:,:,:-1].long()).eq(2).long()
    shaking_tag = (outputs[:, :, :, :-1] > 0.5).long()
    # rel_num, seq_lens, seq_lens, tag_num = shaking_tag.shape
    current_seq_length = mask.sum().item() - 26
    for spot_ind, spot in enumerate(shaking_tag.nonzero().tolist()):
        r_index = spot[0]
        eventType = -1
        role = -1
        for key, value in config.vocab.triAndArgRelation.items():
            if key == r_index:
                eventType = value[0]
                role = value[1]
                break
        tri_start_index = spot[1]
        arg_start_index = spot[2]
        tag = spot[3]
        # f57ac092732d7548c3f56682f067ae67触发词长度为1训练集只有1个，验证机和测试集没有
        if tag == 0 and (spot_ind + 1) < len(shaking_tag.nonzero().tolist()):
            if shaking_tag.nonzero().tolist()[spot_ind + 1][3] == 1:
                # 这块的arg_end_index要不要等于arg_start_index呢？毕竟论元长度为1的很少
                arg_end_index = shaking_tag.nonzero().tolist()[spot_ind + 1][2]
                # for tri_end_index in range(tri_start_index, seq_lens):
                for tri_end_index in range(tri_start_index, current_seq_length):
                    # tri_start_index+1是因为触发词长度不是1
                    if shaking_tag[r_index][tri_end_index][arg_end_index][2] == 1:
                        tiresult1.add((sample_index, tri_start_index, tri_end_index))
                        tcresult1.add((sample_index, tri_start_index, tri_end_index, eventType))
                        airesult1.add((sample_index, arg_start_index, arg_end_index, eventType))
                        acresult1.add((sample_index, arg_start_index, arg_end_index, eventType, role))
                        break

    shaking_tag = shaking_tag_temp
    shaking_tag = torch.cat([shaking_tag[:, :, :, 0][:, :, :, None], shaking_tag[:, :, :, 3][:, :, :, None],
                             shaking_tag[:, :, :, 2][:, :, :, None]], -1)  # 0,3,2
    # shaking_tag = ((shaking_tag>0.5).long()+mask[:,:,:,[0,3,2]].long()).eq(2).long()
    shaking_tag = (shaking_tag > 0.5).long()
    shaking_tag = shaking_tag.permute(0, 2, 1, 3)
    tiresult2 = set()
    tcresult2 = set()
    airesult2 = set()
    acresult2 = set()
    for spot_ind, spot in enumerate(shaking_tag.nonzero().tolist()):
        r_index = spot[0]
        eventType = -1
        role = -1
        for key, value in config.vocab.triAndArgRelation.items():
            if key == r_index:
                eventType = value[0]
                role = value[1]
                break
        tri_start_index = spot[2]
        arg_start_index = spot[1]
        tag = spot[3]
        if tag == 0 and (spot_ind + 1) < len(shaking_tag.nonzero().tolist()):
            if shaking_tag.nonzero().tolist()[spot_ind + 1][3] == 1:
                tri_end_index = shaking_tag.nonzero().tolist()[spot_ind + 1][2]
                # 分析验证机没有触发词为1的论元
                # if tri_start_index!=tri_end_index:
                for arg_end_index in range(arg_start_index, current_seq_length):
                    # for arg_end_index in range(arg_start_index, seq_lens):
                    if shaking_tag[r_index][arg_end_index][tri_end_index][2] == 1:
                        tiresult2.add((sample_index, tri_start_index, tri_end_index))
                        tcresult2.add((sample_index, tri_start_index, tri_end_index, eventType))
                        airesult2.add((sample_index, arg_start_index, arg_end_index, eventType))
                        acresult2.add((sample_index, arg_start_index, arg_end_index, eventType, role))
                        break

    tiresult = tiresult1 & tiresult2
    tcresult = tcresult1 & tcresult2
    airesult = airesult1 & airesult2
    acresult = acresult1 & acresult2

    return tiresult, tcresult, airesult, acresult


def new_decode(outputs, config):
    tiresult = set()
    tcresult = set()
    airesult = set()
    acresult = set()
    data = outputs['ac']
    sample_index = set(data[:, 0])
    for i in sample_index:
        shuju = data[data[:, 0] == i]
        role = set(shuju[:, 1])
        # 解码方向1，挑选出ul, 和所有的触发词开始的位置
        ul_sample_list = shuju[(shuju[:, 1] == role) & (shuju[:, 4] == 0)]
        ul_sample_list_tri = set(ul_sample_list[:, 2])
        ur_sample_list = shuju[(shuju[:, 1] == role) & (shuju[:, 4] == 1)]
        ur_sample_list_tri = set(ur_sample_list[:, 2])
        select_tri_arg = []
        select_arg = set()
        decode_one_tri = ul_sample_list_tri & ur_sample_list_tri
        # 找到所有距离ul的ur顶点
        for tri in decode_one_tri:
            ul_sample_list_arg = ul_sample_list[ul_sample_list[:, 2] == tri][:, 3]
            ur_sample_list_arg = ur_sample_list[ur_sample_list[:, 2] == tri][:, 3]
            # 得保证ul_sample_list_arg\ur_sample_list_arg是按照从小到大的顺序排列的
            for arg1 in ul_sample_list_arg:
                filtered_numbers = [num for num in ur_sample_list_arg if num > arg1]
                if filtered_numbers:
                    arg2 = min(filtered_numbers)
                    airesult.add((arg1, arg2, role))
                    # acresult.add((arg1, arg2, eventType, role))
                    select_tri_arg.append((tri, arg2))
                    select_arg.add(arg2)
                else:
                    continue
                # for arg2 in ur_sample_list_arg:
                #     if arg2 > arg1:
                #         airesult.add((arg1, arg2, role))
                #         acresult.add((arg1, arg2, eventType, role))
                #         select_tri_arg.append((tri, arg2))
                #         select_arg.add(arg2)
                #         break
        lr_sample_list = shuju[(shuju[:, 1] == role) & (shuju[:, 4] == 3)]
        lr_sample_list_arg = set(lr_sample_list[:, 3])
        decode_one_arg = select_arg & lr_sample_list_arg
        arg_dict = {}
        for key in decode_one_arg:
            arg_dict[key] = set()
        for yuansu in select_tri_arg:
            yuansu_tri = yuansu[0]
            yuansu_arg = yuansu[1]
            for k in decode_one_arg:
                if yuansu_arg == k:
                    arg_dict[k].add(yuansu_tri)
        for arg in decode_one_arg:
            ur_sample_list_tri = arg_dict[arg]
            lr_sample_list_tri = lr_sample_list[lr_sample_list[:, 3] == arg][:, 2]
            for tri1 in ur_sample_list_tri:
                filtered_numbers = [num for num in lr_sample_list_tri if num > tri1]
                if filtered_numbers:
                    tri2 = min(filtered_numbers)
                    tiresult.add((tri1, tri2))
                    # tcresult.add((tri1, tri2, eventType))
                else:
                    continue

        # 解码方向2，挑选出lr, 和所有的触发词开始的位置
        lr_sample_list = shuju[(shuju[:, 1] == role) & (shuju[:, 4] == 3)]
        lr_sample_list_tri = set(lr_sample_list[:, 2])

        ll_sample_list = shuju[(shuju[:, 1] == role) & (shuju[:, 4] == 2)]
        ll_sample_list_tri = set(ll_sample_list[:, 2])
        second_select_tri_arg = []
        second_select_arg = set()
        decode_two_tri = lr_sample_list_tri & ll_sample_list_tri
        # 找到所有距离lr的ll顶点
        for tri in decode_two_tri:
            lr_sample_list_arg = lr_sample_list[lr_sample_list[:, 2] == tri][:, 3]
            ll_sample_list_arg = ll_sample_list[ll_sample_list[:, 2] == tri][:, 3]
            # 这块处理的不对--
            for arg1 in lr_sample_list_arg:
                filtered_numbers = [num for num in ll_sample_list_arg if num < arg1]
                if filtered_numbers:
                    arg2 = max(filtered_numbers)
                    airesult.add((arg2, arg1, role))
                    # acresult.add((arg2, arg1, eventType, role))
                    second_select_tri_arg.append((tri, arg2))
                    second_select_arg.add(arg2)
                else:
                    continue
        ul_sample_list = shuju[(shuju[:, 1] == role) & (shuju[:, 4] == 0)]
        ul_sample_list_arg = set(ul_sample_list[:, 3])
        decode_two_arg = second_select_arg & ul_sample_list_arg
        arg_dict = {}
        for key in decode_two_arg:
            arg_dict[key] = set()
        for yuansu in second_select_tri_arg:
            yuansu_tri = yuansu[0]
            yuansu_arg = yuansu[1]
            for k in decode_two_arg:
                if yuansu_arg == k:
                    arg_dict[k].add(yuansu_tri)
        for arg in decode_two_arg:
            ll_sample_list_tri = arg_dict[arg]
            ul_sample_list_tri = ul_sample_list[ul_sample_list[:, 3] == arg][:, 2]
            # 这块也需要改
            for tri1 in ll_sample_list_tri:
                filtered_numbers = [num for num in ul_sample_list_tri if num < tri1]
                if filtered_numbers:
                    tri2 = max(filtered_numbers)
                    tiresult.add((tri2, tri1))
                    # tcresult.add((tri2, tri1, eventType))
                else:
                    continue

    return tiresult, tcresult, airesult, acresult


# def decode_role(outputs, predoutputs, config):
#     argresult = set()
#     roleresult=set()
#     data = outputs['ac']
#     sample_index = set(data[:, 0])
#     for i in sample_index:
#         shuju = data[data[:, 0] == i]
#         role_index = set(shuju[:, 1])
#         for role in role_index:
#             # 挑选出arg
#             arg_index = []
#             for arg in predoutputs['ai']:
#                 if arg[0] == i and arg[-1] == role:
#                     arg_index.append((arg[1], arg[2]))
#             tri_index = []
#             for tri in predoutputs['tc']:
#                 if tri[0] == i:
#                     tri_index.append((tri[1], tri[2], tri[-1]))
#             sample_list = shuju[shuju[:, 1] == role][:, 2:5]
#             # decode1 UL \UR\LR
#             UL_data = sample_list[sample_list[:, 2] == 0]
#             UR_data = sample_list[sample_list[:, 2] == 1]
#             LL_data = sample_list[sample_list[:, 2] == 2]
#             LR_data = sample_list[sample_list[:, 2] == 3]
#             for t in tri_index:
#                 UL_trigger_word = UL_data[UL_data[:, 0] == t[0]]
#                 for a in arg_index:
#                     # 找到所有的ul起始点
#                     ul_data=UL_trigger_word[UL_trigger_word[:, 1] == a[0]]
#                     if len(ul_data) != 0:
#                         ulflag = (ul_data[0] == [t[0], a[0], 0]).min()
#                     else:
#                         ulflag=False
#                     ur_data = UR_data[(UR_data[:, 0] == t[0]) & (UR_data[:, 1] == a[1])]
#                     if len(ur_data) != 0:
#                         urflag = (ur_data[0] == [t[0], a[1], 1]).min()
#                     else:
#                         urflag=False
#                     ll_data = LL_data[(LL_data[:, 0] == t[1]) & (LL_data[:, 1] == a[0])]
#                     if len(ll_data) != 0:
#                         ll_flag = (ll_data[0] == [t[1], a[0], 2]).min()
#                     else:
#                         ll_flag=False
#                     lr_data = LR_data[(LR_data[:, 0] == t[1]) & (LR_data[:, 1] == a[1])]
#                     if len(lr_data) != 0:
#                         lr_flag = (lr_data[0] == [t[1], a[1], 3]).min()
#                     else:
#                         lr_flag=False
#                     flag=ulflag and urflag and ll_flag and lr_flag
#                     if flag:
#                         argresult.add((i,a[0],a[1],t[-1]))
#                         roleresult.add((i,a[0],a[1],t[-1],role))
#     return argresult,roleresult


def decode_arg(outputs, config):
    result = set()
    data = outputs['ai']
    sample_index = set(data[:, 0])
    for i in sample_index:
        shuju = data[data[:, 0] == i]
        role_index = set(shuju[:, 1])
        for role in role_index:
            sample_list = list(set(shuju[shuju[:, 1] == role][:, 2]))
            for m in find_continuous_subarrays(sample_list, "role"):
                result.add(tuple([i] + list(m) + [role]))
    return result


# 解码事件类型
def docode_event(outputs, config):
    finaloutputs = set()
    data = outputs['tc']
    # 样本索引--三位数组的第一列
    sample_index = set(data[:, 0])
    for i in sample_index:
        shuju = data[data[:, 0] == i]
        event_index = set(shuju[:, 1])
        for index, value in config.vocab.new_tri_labels.items():
            if len((set(value) & event_index)) == len(value):
                select_tri = []
                for k in value:
                    select_tri.append(shuju[shuju[:, 1] == k][:, 2])
                intersection_set = set(select_tri[0])
                for lst in select_tri[1:]:
                    intersection_set &= set(lst)
                intersection_list = list(intersection_set)
                eventTypeid = config.vocab.label2id(index, "tri")
                for m in find_continuous_subarrays(intersection_list, "tri"):
                    finaloutputs.add(tuple([i] + list(m) + [eventTypeid]))
    return finaloutputs


def new_decode_tri(outputs):
    finaloutputs = set()
    data = outputs['ti']
    # 样本索引--二位数组的第一列
    sample_index = set(data[:, 0])
    for i in sample_index:
        shuju = data[data[:, 0] == i]
        sample_list = list(set(shuju[:, 1]))
        for m in find_continuous_subarrays(sample_list, "tri"):
            finaloutputs.add(tuple([i] + list(m)))
    return finaloutputs


def decode_tri(outputs):
    trigger_word_index = []
    tri_word = []
    for i in outputs["ti"]:
        trigger_word = i[1]
        if trigger_word not in trigger_word_index:
            trigger_word_index.append(trigger_word)
            tri_word.append(i)
    # 创建一个默认字典，默认值为一个空列表
    result_dict = defaultdict(list)
    # 遍历数组，将第二列的值添加到字典中对应的第一列键的列表中
    for row in np.stack(tri_word, axis=0):
        key = row[0]
        value = row[1]
        if value not in result_dict[key]:
            result_dict[key].append(value)
    # 对字典中每个键对应的列表进行排序
    for key in result_dict:
        result_dict[key].sort()
    # 转换为普通字典
    result_dict = dict(result_dict)
    final_result_dict = set()
    for key, value in result_dict.items():
        for m in find_continuous_subarrays(value):
            final_result_dict.add(tuple([key] + list(m)))
    final_result_dict
    return final_result_dict


# 参数r:真实标签的数量 p：预测标签的数量 c:正确预测的标签的数量
def calculate_f1(r, p, c):
    if r == 0 or p == 0 or c == 0:
        return 0, 0, 0
    r = c / r
    p = c / p
    f1 = (2 * r * p) / (r + p)
    return f1, r, p


def find_continuous_subarrays(arr, triorarg):
    if not arr:
        return []
    final_result = []
    result = []
    temp = [arr[0]]
    for i in range(1, len(arr)):
        if arr[i] == arr[i - 1] + 1:
            temp.append(arr[i])
        else:
            result.append(temp)
            temp = [arr[i]]
    result.append(temp)  # Don't forget to add the last subarray
    for i in result:
        if (len(i)) == 1:
            final_result.append((i[0], i[0]))
        else:
            if triorarg == "tri":
                tri_length = i[-1] - i[0]
                if tri_length <= 5:
                    final_result.append((i[0], i[-1]))
            else:
                tri_length = i[-1] - i[0]
                final_result.append((i[0], i[-1]))
    return final_result
# 把事件表示为想要的输出格式
# def output(inputs,results,vocab):
#
#     final_results=[]
#     #把8个样本对应的句子，每个句子的token_id转化为token，恢复为原来的句子
#     tokenizer = AutoTokenizer.from_pretrained("./bert-base-chinese", cache_dir="./cache/")
#     input_ids=inputs.cpu().tolist()
#     sentence_id=[]
#     for id in input_ids:
#         newid=[]
#         for i in range(1,len(id)):
#             if(id[i]!=0 and id[i]!=102):
#                 newid.append(id[i])
#         sentence_id.append(newid)
#     sentences=[]
#
#     for i in range(len(sentence_id)):
#         final_results.append({})
#
#     #数据集中的逗号是英文的逗号
#     for index,item in enumerate(sentence_id):
#         # sentences.append(''.join(tokenizer.convert_ids_to_tokens(item)))
#         sentence=''.join(tokenizer.convert_ids_to_tokens(item))
#         #不能这么写，初始化时候final_results的大小为0，直接用索引访问会报错
#         # final_results[index]={}
#         final_results[index]["content"]=sentence
#         final_results[index]["id"]=index
#
#     trigger_set=results["tc"]
#     argument_set=results["ac"]
#     ceshi=[]
#     for i in argument_set:
#         if i[0]==2:
#             ceshi.append(i)
#
#     for item in final_results:
#         item["events"]=[]
#         sentence=item["content"]
#         #出发词和事件类型处理
#         for i in trigger_set:
#             if i[0]==item["id"]:
#                 newdict={}
#                 trigger=sentence[i[1]:i[2]+1]
#                 eventType=vocab.tri_id2label[i[3]]
#                 newdict["type"]=eventType
#                 newdict["trigger"]={}
#                 newdict["trigger"]["span"]=list((i[1],i[2]+1))
#                 newdict["trigger"]["word"]=trigger
#                 newdict["arg"]={}
#                 item["events"].append(newdict)
#         for i in argument_set:
#             if i[0]==item["id"]:
#                 for event in item["events"]:
#                     eventTypeid=vocab.label2id(event["type"], "tri")
#                     if i[3]==eventTypeid:
#                         # 同一事件类型 同一角色，论元位置有很多，如何解决？
#                         # (2, 138, 141, 1, 6), (2, 86, 89, 1, 6), (2, 144, 145, 1, 0), (2, 107, 108, 1, 0), (2, 72, 76, 1, 6),
#                         role=vocab.rol_id2label[i[4]]
#                         event["arg"][role]=[]
#                         xindict={}
#                         xindict["span"]=list((i[1],i[2]+1))
#                         xindict["word"]=sentence[i[1]:i[2]+1]
#                         event["arg"][role].append(xindict)
#     # print(final_results)
#     # for shuju in final_results:
#     #     with open("./shuju.json", 'w', encoding='utf_8', newline='') as f:
#     #         json.dump(shuju, f, ensure_ascii=False)
