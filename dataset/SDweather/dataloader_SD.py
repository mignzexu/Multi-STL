import os
import numpy as np
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset
from .range.data_seq import DataSeq
from Instrument.standardizer import Load_Standardizer
from tqdm import tqdm


class SD_Dataloader(Dataset):
    def __init__(self, configs, mode):
        super().__init__()

        self.configs = configs
        self.mode = mode
        self.base_channel = ["CR", "CP", "LT", "TR", "CW"]
        self.data_dir = os.path.join(self.configs.data_dir, "Fusion_RPLTW")
        self.data = None
        self.standardizer = Load_Standardizer(self.configs).standardizer
        self.data_idx = []
        self.data_range = self.get_data_range()
        print(self.data_range)

        if hasattr(self.configs, "region"):
            self.region_idx = self.regional_interception()
        else:
            self.region_idx = None
        
        #定义输入类别的时候，注意将输出类别按顺序相邻：in: [a, b, c, d] out: [b, c]
        self.category = self.configs.in_category
        self.label_cate = self.configs.out_category
        self.total_seq = self.configs.total_seq
        self.total_length = sum(self.total_seq)
        self.stride = self.configs.data_stride
        self.label_idx = self.get_lable_idx()
        
        self.load_data()
        # self.replace_nan_with_zero()
        self.standardizing()
        print(f"{self.mode} 数据集已加载: {len(self.data_idx)} 个样本")

    def get_data_range(self):

        if self.mode == 'train':
            r = self.configs.train_range
        elif self.mode == 'valid':
            r = self.configs.valid_range
        elif self.mode == 'test':
            r = self.configs.test_range
        else:
            raise ValueError("Mode should be 'train', 'valid', or 'test'.")
        
        range_mode, range_inf = r.split(':')
        data_range = DataSeq(range_inf, self.data_dir, range_mode)
        
        return data_range.data_list
    
    def load_data(self):
        range_len = len(self.data_range)
        base_seq = 0
        
        # 使用 tqdm 包装循环，总数 = range_len
        for i in tqdm(range(range_len), desc=f" {self.mode} 数据集加载进度"):
            pre = self.data_range[i]
            pre_length = pre[1][1] + 1 - pre[1][0]
            
            if i == range_len - 1:
                #处理最后一个元素
                if pre_length >= self.total_length:
                    data = np.load(pre[3]).astype(np.float32)
                    base_seq = self.fusion_data(data, pre[2], pre[0], pre[1], base_seq, compensation=True)
            else:
                aft = self.data_range[i + 1]
                aft_length = aft[1][1] + 1 - aft[1][0]

                if pre[1][1] == 143 and aft[1][0] == 0 and datetime.strptime(pre[0], "%Y%m%d") + timedelta(days=1) == datetime.strptime(aft[0], "%Y%m%d"):
                    if pre_length + aft_length < self.total_length:
                        continue
                    else:
                        if aft_length >= self.total_length:
                            data = np.load(pre[3]).astype(np.float32)
                            base_seq = self.fusion_data(data, pre[2], pre[0], pre[1], base_seq, compensation=False)
                        else:
                            data = np.concatenate((np.load(pre[3]).astype(np.float32), np.load(aft[3]).astype(np.float32)), axis=0)
                            base_seq = self.fusion_data(data, pre[2], pre[0], pre[1], base_seq, compensation=True)
                else:
                    if pre_length < self.total_length:
                        continue
                    else:
                        data = np.load(pre[3]).astype(np.float32)
                        base_seq = self.fusion_data(data, pre[2], pre[0], pre[1], base_seq, compensation=True)
                
            # 可选择在进度条中更新当前处理的日期描述
            # tqdm.write(f"{pre[0]} 数据已加载。")


    
    # def load_data(self):
    #     range_len = len(self.data_range)
    #     base_seq = 0
    #     for i in range(range_len - 1):

    #         pre = self.data_range[i]
    #         aft = self.data_range[i + 1]
    #         pre_lengh = pre[1][1] + 1 - pre[1][0]
    #         aft_lengh = aft[1][1] + 1 - aft[1][0]

    #         if pre[1][1] == 143 and aft[1][0] == 0 and datetime.strptime(pre[0], "%Y%m%d") + timedelta(days=1) == datetime.strptime(aft[0], "%Y%m%d"):
                
    #             if pre_lengh + aft_lengh < self.total_length :
    #                 continue
    #             else:
    #                 if aft_lengh >= self.total_length :
    #                     data = np.load(pre[3]).astype(np.float32)
    #                     base_seq = self.fusion_data(data, pre[2], pre[0], pre[1], base_seq, compensation=False)

    #                 else:

    #                     data = np.concatenate((np.load(pre[3]).astype(np.float32), np.load(aft[3]).astype(np.float32)), axis=0)
    #                     base_seq = self.fusion_data(data, pre[2], pre[0], pre[1], base_seq, compensation=True)
    #         else:
    #             if pre_lengh < self.total_length:
    #                 continue
    #             else:
    #                 data = np.load(pre[3]).astype(np.float32)
    #                 base_seq = self.fusion_data(data, pre[2], pre[0], pre[1], base_seq, compensation=True)
            
    #         print(f"{pre[0]}数据已加载。")

    #     pre = self.data_range[-1]
    #     pre_lengh = pre[1][1] + 1 - pre[1][0]
    #     if pre_lengh >= self.total_length:
    #         data = np.load(pre[3]).astype(np.float32)
    #         base_seq = self.fusion_data(data, pre[2], pre[0], pre[1], base_seq, compensation=True)

    #     print(f"{pre[0]}数据已加载。")
        

    def fusion_data(self, data, data_range, day, real_range, base_seq, compensation=True):

        if self.region_idx is not None:
            data = data[:, :,self.region_idx[0][0]:self.region_idx[0][1], self.region_idx[1][0]:self.region_idx[1][1]]
        data = self.tailor_data(data, data_range)
        data = self.reorder_channels(data, self.base_channel, self.category)
        
        total_len = data.shape[0]
        if compensation:
            last_valid_start = total_len - self.total_length
        else:
            last_valid_start = total_len

        for i in range(0, last_valid_start, self.stride):
            start_idx = base_seq + i
            time = datetime.strptime(day, "%Y%m%d") + timedelta(minutes=(real_range[0] + i)*10)
            self.data_idx.append((start_idx, time.strftime("%Y%m%d_%H%M")))

        if self.data is None:
            self.data = torch.from_numpy(data)
        else:
            self.data = torch.cat((self.data, torch.from_numpy(data)), dim=0)
        
        base_seq += total_len

        return base_seq

    def replace_nan_with_zero(self):
        if self.data is not None:
            self.data = torch.nan_to_num(self.data, nan=0.0)
    
    def standardizing(self):

        if self.mode == "train":
            self.standardizer.cal_params(self.data, self.configs.label_idx)
            self.data = self.standardizer.standardizing(self.data)
        elif self.mode == "test" or self.mode == "valid":
            self.standardizer.data_params()
            self.data = self.standardizer.standardizing(self.data)
        else:
            raise ValueError("mode must be train, test or valid")

    @staticmethod
    def tailor_data(data, range):
        if range[0] == range[1] == "-":
            return data
        elif range[0] != "-" and range[1] == "-":
            return data[range[0]:, ... ]
        elif range[0] == "-" and range[1] != "-":
            return data[:range[1] + 1, ... ]
        else:
            raise ValueError(f"数据范围 {range} 格式错误.")

    
    def regional_interception(self):

        lat_min = float(self.configs.region["lat"].split('-')[0])
        lat_max = float(self.configs.region["lat"].split('-')[1])
        lon_min = float(self.configs.region["lon"].split('-')[0])
        lon_max = float(self.configs.region["lon"].split('-')[1])

        region_idx = [[],[]]

        region_idx[0].append(int((39 - lat_max) * 100))
        region_idx[0].append(int((39 - lat_min) * 100))

        region_idx[1].append(int((lon_min - 114) * 100))  
        region_idx[1].append(int((lon_max - 114) * 100) + 1)

        if self.mode == "train":
            self.configs.img_size[0] = region_idx[0][1] - region_idx[0][0]
            self.configs.img_size[1] = region_idx[1][1] - region_idx[1][0]
        
        return region_idx

    def get_lable_idx(self):
        label_idx = []
        if len(self.label_cate) == 1 and self.label_cate[0] in self.category:
            label_idx.append(self.category.index(self.label_cate[0]))
            label_idx.append(label_idx[0] + 1)
            self.configs.label_idx = label_idx
        elif len(self.label_cate) > 1:
            idx = []
            for item in self.label_cate:
                if item in self.category:
                    idx.append(self.category.index(item))
                else:
                    raise ValueError(f"标签类别 {item} 不在输入类别中")
            
            if all(b - a == 1 for a, b in zip(idx, idx[1:])):
                label_idx = [min(idx), max(idx) + 1]
                self.configs.label_idx = label_idx
            else:
                raise ValueError(f"输出类别 {self.label_cate} 必须在输入中相邻顺序排列。")
        else:
            raise ValueError(f"请设置输出类别 {self.label_cate}。")
        
        return label_idx
    
    @staticmethod
    def reorder_channels(arr, old_order, new_order):
        """
        按 new_order 指定的通道名顺序进行选择和重排。
        """
        for ch in new_order:
            if ch not in old_order:
                raise ValueError(f"通道 {ch!r} 不在 old_order 中")

        index_map = [old_order.index(ch) for ch in new_order]
        return arr[:, index_map, ...]

    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, idx):
        inf = self.data_idx[idx]
        sample = self.data[inf[0]:(inf[0] + self.total_seq[0]), :, :, :]
        label = self.data[(inf[0] + self.total_seq[0]):(inf[0] + self.total_length), :, :, :]

        return sample, label
    

if __name__ == '__main__':

    from argparse import Namespace

    configs = {
        "train_range": 'point:r1',
        "in_category": ["CW", "CR", "LT", "TR"],
        "out_category": ["CW"],
        "data_dir": r"/shares/weather",
        "total_seq": [18, 18],
        "img_size": [500, 900],
        "region": {
            "lon" : "117-118.8",
            "lat" : "37-38"
        },
        "is_terrain": True,
        "data_stride": 1,
        "std_method": "z_score_sd"
    }

    configs = Namespace(**configs)

    dataset = SD_Dataloader(configs, "train")
    print(configs.img_size)
    # print(dataset.data_idx)
    print(len(dataset.data_idx))
    print(dataset.data.shape)
    print(dataset.configs.std_params)
    sample, label = dataset.__getitem__(len(dataset.data_idx) - 1)
    print(sample.shape)
    print(label.shape)
    print(dataset.data.mean())
    print(dataset.data.std())
    print(dataset.data.min())
    print(dataset.data.max())
