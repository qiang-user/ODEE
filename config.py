import json


class Config:
    def __init__(self, args):
        with open(args.config, "r", encoding="utf-8") as f:
            # json.load() 函数将文件内容解析为 JSON 格式，然后将其转换为相应的 Python 数据结构（通常是字典或列表），
            # 并将其赋值给变量 config
            config = json.load(f)

        self.dataset = config["dataset"]
        self.bert_hid_size = config["bert_hid_size"]
        self.tri_hid_size = config["tri_hid_size"]
        self.eve_hid_size = config["eve_hid_size"]
        self.arg_hid_size = config["arg_hid_size"]
        # self.node_type_size = config["node_type_size"]
        self.event_sample = config["event_sample"]
        # self.layers = config["layers"]

        self.dropout = config["dropout"]
        # self.graph_dropout = config["graph_dropout"]

        self.epochs = config["epochs"]
        self.warm_epochs = config["warm_epochs"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.grad_clip_norm = config["grad_clip_norm"]
        self.gamma = config["gamma"]

        self.bert_name = config["bert_name"]
        self.bert_learning_rate = config["bert_learning_rate"]

        self.seed = config["seed"]

        # args.__dict__ 是 args 对象的一个属性，用于存储 args对象的所有属性和它们的值的字典
        for k, v in args.__dict__.items():
            if v is not None:
                self.__dict__[k] = v

    # 初始化Config类的时候，该方法会自动执行，返回一个字符串，包含了该实例的所有属性及其对应的值：
    # 返回值的样子dict_items([('name', 'John'), ('age', 30)])
    # self.__dict__ 是一个字典，用于存储对象的所有实例变量及其对应的值
    def __repr__(self):
        return "{}".format(self.__dict__.items())
