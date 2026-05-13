import os
import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from Instrument.standardizer import Load_Standardizer




class WB_Dataloader(Dataset):
    def __init__(self, configs, mode):

        self.configs = configs
        self.mode = mode
        self.data_dir = self.configs.data_dir
        self.category = self.configs.in_category
        self.label_cate = self.configs.out_category
        self.total_seq = self.configs.total_seq
        self.resolusion = self.configs.resolusion
        self.stride = self.configs.data_stride

        self.data = None
        self.data_idx = []

        if self.mode == 'train':
            self.data_range = self.configs.train_range
        elif self.mode == 'valid':
            self.data_range = self.configs.valid_range
        elif self.mode == 'test':
            self.data_range = self.configs.test_range
            self.total_seq[1] = self.configs.test_seq
        else:
            raise ValueError("Mode should be 'train', 'valid', or 'test'.")
        
        #定义输入类别的时候，注意将输出类别按顺序相邻：in: [a, b, c, d] out: [b, c]

        self.standardizer = Load_Standardizer(self.configs).standardizer
        
        self.data_name = {
            "tcc" : "total_cloud_cover",
            "tp" : "total_precipitation",
            "t2m" : "2m_temperature",
            "z500": "geopotential_500",
            "t850": "temperature_850",
            "u10" : "10m_u_component_of_wind",
            "v10" : "10m_v_component_of_wind",
            "w10" : "10m_wind_speed"
            }
        
        if self.mode =='train':
            self.get_lable_idx()
            
        self.get_data()
        self.standardizing()
        print(f"{mode} 数据集已加载: {len(self.data_idx)} 个样本")

    def get_data(self):

        if self.resolusion == "5.625":
            reso_dir = os.path.join(self.data_dir, "WB", "5_625deg")
        elif self.resolusion == "2.8125":
            reso_dir = os.path.join(self.data_dir, "WB", "2_8125deg")
        elif self.resolusion == "1.40625":
            reso_dir = os.path.join(self.data_dir, "WB", "1_40625deg")
        else:
            raise ValueError(f"分辨率{self.resolusion}不存在")
        
        start_time = int(self.data_range[0])
        end_time = int(self.data_range[1])
        base_line = 0

        for time in range(start_time, end_time + 1):
            year_data = None
            for cate in self.category:

                if cate in {"orography", "lsm"}:
                    data = xr.open_dataset(os.path.join(reso_dir, "constants", f"constants_{self.resolusion}deg.nc"))
                    dataset = data[cate].values

                elif cate == "w10":
                    data_u = xr.open_dataset(os.path.join(reso_dir, '10m_u_component_of_wind', f"10m_u_component_of_wind_{str(time)}_{self.resolusion}deg.nc"))
                    data_v = xr.open_dataset(os.path.join(reso_dir, '10m_v_component_of_wind', f"10m_v_component_of_wind_{str(time)}_{self.resolusion}deg.nc"))
                    dataset = np.sqrt(data_u["u10"].values**2 + data_v["v10"].values**2)
                    dataset = self.enhance_gust_forecast(dataset)

                else:
                    data = xr.open_dataset(os.path.join(reso_dir, self.data_name[cate], f"{self.data_name[cate]}_{str(time)}_{self.resolusion}deg.nc"))
                    dataset = data[cate].values

                if dataset.ndim == 3 :
                    dataset = np.expand_dims(dataset, axis=1)
                    dataset = torch.from_numpy(dataset)
                elif dataset.ndim == 2:
                    dataset = torch.from_numpy(dataset)
                    dataset = dataset.unsqueeze(0).unsqueeze(0)

                if year_data is None:
                    year_data = dataset
                else:
                    if dataset.shape[0] == year_data.shape[0]:
                        year_data = torch.cat((year_data, dataset), dim=1)
                    elif dataset.shape[0] > year_data.shape[0]:
                        year_data = year_data.expand(dataset.shape[0], -1, -1, -1)
                        year_data = torch.cat((year_data, dataset), dim=1)
                    else:
                        dataset = dataset.expand(year_data.shape[0], -1, -1, -1)
                        year_data = torch.cat((year_data, dataset), dim=1)

            base_line = self.get_data_idx(year_data, time, base_line)
            if self.data is None:
                self.data = year_data
            else:
                self.data = torch.cat((self.data, year_data), dim=0)
            print(f"{str(time)}年数据完成加载")

        self.data = self.adjust_geo_view(self.data)

    @staticmethod
    def enhance_gust_forecast(wind_array):
        wind = wind_array.copy()
        gust = np.zeros_like(wind)
        gust += (wind <= 3.3) * wind * 1.5
        gust += ((wind > 3.3) & (wind <= 5.4)) * wind * 1.8
        gust += ((wind > 5.4) & (wind <= 7.9)) * wind * 1.75
        gust += ((wind > 7.9) & (wind <= 10.7)) * wind * 1.6
        gust += ((wind > 10.7) & (wind <= 13.8)) * wind * 1.6
        gust += (wind > 13.8) * wind * 1.6
        return gust
    

    @staticmethod
    def adjust_geo_view(data):
        """
        调整气象数据的地理视角 (PyTorch 版本)。
        """
        # 0. 确保数据是 Tensor 类型 (如果还是 Numpy 或 List，会自动转换)
        # 这一步也会强制将数据从硬盘(如果是lazy load)读取到内存中，避免之前的报错
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)

        # 1. 获取宽度 W (最后一个维度)
        W = data.shape[-1]
        
        # 2. 上下翻转 (针对 H 维度/倒数第2维)
        # 注意: torch.flip 的 dims 参数必须是列表或元组，不能是单个 int
        data = torch.flip(data, dims=[-2])
        
        # 3. 经度平移 (针对 W 维度/倒数第1维)
        # 操作: 向右滚动一半宽度
        shift = W // 2
        data = torch.roll(data, shifts=shift, dims=-1)
        
        return data

    def get_data_idx(self, data, year, base_line):
        total_len = data.shape[0]
        last_valid_start = total_len - sum(self.total_seq) + 1
        
        for i in range(0, last_valid_start, self.stride):
            start_idx = base_line + i
            label = f"{year}_{i}"
            self.data_idx.append((start_idx, label))
        
        base_line += total_len

        return base_line

    def standardizing(self):

        if self.mode == "train":
            self.standardizer.cal_params(self.data, self.configs.label_idx)
            self.data = self.standardizer.standardizing(self.data)
        elif self.mode == "test" or self.mode == "valid":
            self.standardizer.data_params()
            self.data = self.standardizer.standardizing(self.data)
        else:
            raise ValueError("mode must be train, test or valid")

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

    def __len__(self):
        return len(self.data_idx)
    
    def __getitem__(self, idx):
        start_idx = self.data_idx[idx]
        sample = self.data[start_idx[0]:(start_idx[0] + self.total_seq[0]),:,:,:]
        label = self.data[(start_idx[0] + self.total_seq[0]):(start_idx[0] + sum(self.total_seq)), :,:, :]

        return sample, label



if __name__ == '__main__':
    from argparse import Namespace
    configs = {
        "data_dir": "/scratch/mingze/data",
        "train_range": ["2010", "2012"],
        "valid_range": ["2016", "2016"],
        "test_range" : ["2017", "2018"],
        "in_category": ["tp", "tcc", "u", "v", "orography"],
        "out_category": ["tp"],
        "total_seq": [12, 12],
        "resolusion": "5.625",
        "data_stride": 1,
        "metrics": [ "mae", "mse", "rmse"],
        "threshold": [[0.0001, 0.001, 0.0005], [0.00002, 0.0003, 0.0004]],
        "std_method": "z_score"
    }

    configs = Namespace(**configs)

    train_dataset = WB_Dataloader(configs, "train")
    inputs, labels = train_dataset.__getitem__(59)
    print(inputs.shape)
    print(labels.shape)
    print(train_dataset.data.shape)
    print(len(train_dataset.data_idx))
    print(train_dataset.data.mean())
    print(train_dataset.data.std())
    print(configs)
