import torch,os,yaml
import numpy as np

def read_yaml(yaml_path):
    # 读取Yaml文件方法
    with open(yaml_path, encoding="utf-8", mode="r") as f:
        result = yaml.load(stream=f, Loader=yaml.FullLoader)
        return result

def seed_torch(seed):
    # random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # set cpu random seed
    torch.cuda.manual_seed(seed)# set gpu random seed
    torch.cuda.manual_seed_all(seed)# set all gpus random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False