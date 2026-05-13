import json
import os

class SDweather_config:

    def __init__(self, config_name):

        self.config_name = config_name
        self.data_config = self.load_config()

    def load_config(self):
        path = os.path.join(os.path.dirname(__file__), self.config_name + '.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f'未找到数据集配置文件: {path}')
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        print('数据集配置文件加载成功')
        return config
