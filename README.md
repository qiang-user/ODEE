# ODEE论文复现

ODEE: A One-Stage Object Detection Framework for Overlapping and Nested Event Extraction

本文尝试对ODEE论文进行复现，但是并未达到原文的效果，以下是我的流程，欢迎大家指出复现中可能出现的错误。

## 1. 参考论文及github链接

ODEE论文全称为：ODEE: A One-Stage Object Detection Framework for Overlapping and Nested Event Extraction，github链接为[NingJinzhong/ODEE (github.com)](https://github.com/NingJinzhong/ODEE)

ODRTE的github链接：[NingJinzhong/ODRTE (github.com)](https://github.com/NingJinzhong/ODRTE)

oneEE论文：https://arxiv.org/pdf/2209.02693.pdf  ，github链接为https://github.com/Cao-Hu/OneEE

## 2. 代码复现过程
- 使用的bert-base-chinese在oneEE的github中提及了。
- 数据预处理过程参考oneEE代码的流程，在mydata_loader.py文件中
- 模型构建和解码部分参考ODRTE代码的流程，其中模型搭建在eemodel.py,结果解码在utils.py中。

## 3. Dataset

- FewFC: Chinese Financial Event Extraction dataset. The original dataset can be accessed at [this repo](https://github.com/TimeBurningFish/FewFC). Here we follow the settings of [CasEE](https://github.com/JiaweiSheng/CasEE). Note that the data is avaliable at /data/fewFC, and we adjust data format for simplicity of data loader. To run the code on other dataset, you could also adjust the data as the data format presented.
- [ge11: GENIA Event Extraction (GENIA), 2011](https://2011.bionlp-st.org/home/genia-event-extraction-genia)
- [ge13: GENIA Event Extraction (GENIA), 2013](http://bionlp.dbcls.jp/projects/bionlp-st-ge-2013/wiki/Overview)

## 4. 遇到的问题及思考

- 自己搭建模型仅在fewFC数据集上进行尝试，未能复现出跟模型相近的结果
- 在fewFC数据集上尝试使用过关系采样对数据进行处理，效果比没有使用关系采样的稍好一点，但是论文中没有提及关系的采样，尽是我参考ODRTE后的尝试
- no-relation-sample.log是没有进行关系采样的训练日志，sample-epoch-100.log是进行关系采样的训练日志。

## 5. 致谢

复现的代码仅供参考，我的联系方式2445634146@qq.com，欢迎大家指正复现的代码中可能存在的问题。
