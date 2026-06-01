import os
import json
import shutil
import fcntl
import numpy as np
from numpy.lib.format import open_memmap
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset
from .range.data_seq import DataSeq
from Instrument.standardizer import Load_Standardizer
from tqdm import tqdm



class Mem_Loader:
    """
    将 SDweather 拆分数据写入 /dev/shm 中的 .npy memmap 缓存。

    输出：
        /dev/shm/SD/{data_config}/{mode}.npy
        /dev/shm/SD/{data_config}/{mode}_idx.json

    数据形状：
        [T, C, H, W]
    """

    def __init__(self, configs, mode):
        self.configs = configs
        self.mode = mode
        self.data_config = getattr(self.configs, "data_config", "default")
        self.data_dir = self.configs.data_dir
        self.static_channels = ["CT"]
        self.optional_channels = ["CL"]

        self.category = list(self.configs.in_category)
        self.dynamic_channels = [ch for ch in self.category if ch not in self.static_channels]
        self.dynamic_ch_idx = [i for i, ch in enumerate(self.category) if ch not in self.static_channels]
        self.static_ch_idx = [i for i, ch in enumerate(self.category) if ch in self.static_channels]

        self.total_seq = list(self.configs.total_seq)
        self.total_length = sum(self.total_seq)
        self.stride = int(self.configs.data_stride)
        self.img_size = list(self.configs.img_size)

        self.region_idx = None
        if hasattr(self.configs, "region"):
            self.region_idx = self._compute_region_idx()

        self.data_range = self._get_data_range()
        self.data_idx = []

        self.cache_dir = os.path.join("/dev/shm", "SD", str(self.data_config))
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, f"{self.mode}.npy")
        self.idx_path = os.path.join(self.cache_dir, f"{self.mode}_idx.json")

        self._generate_cache()

        print(f"\n{self.mode} 缓存生成完成")
        print(f"  data: {self.cache_path}")
        print(f"  idx : {self.idx_path}")
        print(f"  samples: {len(self.data_idx)}")

    def _get_data_range(self):
        if self.mode == 'train':
            r = self.configs.train_range
        elif self.mode == 'valid':
            r = self.configs.valid_range
        elif self.mode == 'test':
            r = self.configs.test_range
        else:
            raise ValueError("Mode should be 'train', 'valid', or 'test'.")

        range_mode, range_inf = r.split(':')
        data_seq = DataSeq(range_inf, self.data_dir, range_mode, 
                          channels=self.dynamic_channels, 
                          optional_channels=self.optional_channels)
        return data_seq.data_list

    def _compute_region_idx(self):
        lat_min = float(self.configs.region["lat"].split('-')[0])
        lat_max = float(self.configs.region["lat"].split('-')[1])
        lon_min = float(self.configs.region["lon"].split('-')[0])
        lon_max = float(self.configs.region["lon"].split('-')[1])

        region_idx = [[], []]
        region_idx[0].append(int((lat_min - 34) * 100))
        region_idx[0].append(int((lat_max - 34) * 100))
        region_idx[1].append(int((lon_min - 114) * 100))
        region_idx[1].append(int((lon_max - 114) * 100) + 1)

        self.img_size[0] = region_idx[0][1] - region_idx[0][0]
        self.img_size[1] = region_idx[1][1] - region_idx[1][0]

        return region_idx

    def _generate_cache(self):
        segments = self._collect_valid_segments()
        total_timesteps = sum(seg["length"] for seg in segments)

        if total_timesteps == 0:
            raise ValueError(f"{self.mode} 没有有效数据段")

        channels = len(self.category)
        h, w = self.img_size
        expected_shape = (total_timesteps, channels, h, w)
        expected_bytes = np.prod(expected_shape) * np.dtype("float32").itemsize

        print("\n" + "=" * 80)
        print(f"开始生成 {self.mode} 缓存")
        print(f"数据段数: {len(segments)}")
        print(f"缓存路径: {self.cache_path}")
        print(f"缓存 shape: {expected_shape}")
        print(f"预计大小: {self._format_bytes(expected_bytes)}")
        print("=" * 80)

        self._print_shm_usage("生成前 /dev/shm 使用情况")

        cache = open_memmap(
            self.cache_path,
            mode="w+",
            dtype="float32",
            shape=expected_shape,
        )

        print("加载动态模态...")
        offset = 0
        for seg in tqdm(segments, desc=f"{self.mode} 动态模态写入进度"):
            data = self._load_dynamic_segment(seg)
            seg_len = data.shape[0]
            cache[offset:offset + seg_len, self.dynamic_ch_idx] = data
            self._generate_segment_idx(seg_len, seg, offset)
            offset += seg_len
            del data

        if offset != total_timesteps:
            raise RuntimeError(
                f"写入结束后 offset={offset}，但预期 total_timesteps={total_timesteps}。"
            )

        print("广播静态模态...")
        self._broadcast_static_channels(cache, total_timesteps)

        cache.flush()
        self._save_data_idx()
        self._print_shm_usage(f"{self.mode} 写入后 /dev/shm 使用情况")

    def _collect_valid_segments(self):
        """
        从 data_list 中收集有效数据段。

        data_list 格式: [[day, real_range, modalities], ...]
        modalities 格式: [[path, data_range], ...]
        """
        segments = []
        range_len = len(self.data_range)

        for i in range(range_len):
            day, real_range, modalities = self.data_range[i]
            pre_length = real_range[1] + 1 - real_range[0]

            if i == range_len - 1:
                if pre_length >= self.total_length:
                    segments.append({
                        "day": day,
                        "real_range": real_range,
                        "modalities": modalities,
                        "length": pre_length,
                        "compensation": True,
                        "merge_modalities": None,
                    })
            else:
                next_day, next_real_range, next_modalities = self.data_range[i + 1]
                aft_length = next_real_range[1] + 1 - next_real_range[0]

                is_continuous = (
                    real_range[1] == 143 and next_real_range[0] == 0 and
                    datetime.strptime(day, "%Y%m%d") + timedelta(days=1) == datetime.strptime(next_day, "%Y%m%d")
                )

                if is_continuous:
                    if pre_length + aft_length < self.total_length:
                        continue
                    if aft_length >= self.total_length:
                        segments.append({
                            "day": day,
                            "real_range": real_range,
                            "modalities": modalities,
                            "length": pre_length,
                            "compensation": False,
                            "merge_modalities": None,
                        })
                    else:
                        segments.append({
                            "day": day,
                            "real_range": real_range,
                            "modalities": modalities,
                            "length": pre_length + aft_length,
                            "compensation": True,
                            "merge_modalities": next_modalities,
                        })
                else:
                    if pre_length < self.total_length:
                        continue
                    segments.append({
                        "day": day,
                        "real_range": real_range,
                        "modalities": modalities,
                        "length": pre_length,
                        "compensation": True,
                        "merge_modalities": None,
                    })

        return segments

    def _load_dynamic_segment(self, seg_info):
        modalities = seg_info["modalities"]
        merge_modalities = seg_info["merge_modalities"]
        target_t = seg_info["length"]

        channel_data = []
        for idx, (path, data_range) in enumerate(modalities):
            if path == "-":
                data = np.zeros((target_t, 1, self.img_size[0], self.img_size[1]), dtype=np.float32)
            else:
                data = np.load(path).astype(np.float32)

                if merge_modalities is not None:
                    merge_path, merge_range = merge_modalities[idx]
                    if merge_path != "-":
                        merge_data = np.load(merge_path).astype(np.float32)
                        data = np.concatenate((data, merge_data), axis=0)

                data = self._tailor_data(data, data_range)

                if self.region_idx is not None:
                    if data.ndim == 3:
                        data = data[:,
                                    self.region_idx[0][0]:self.region_idx[0][1],
                                    self.region_idx[1][0]:self.region_idx[1][1]]
                    elif data.ndim == 4:
                        data = data[:,
                                    :,
                                    self.region_idx[0][0]:self.region_idx[0][1],
                                    self.region_idx[1][0]:self.region_idx[1][1]]

                if data.ndim == 3:
                    data = data[:, np.newaxis, :, :]

            channel_data.append(data)

        return np.concatenate(channel_data, axis=1)

    def _broadcast_static_channels(self, cache, total_timesteps):
        for ch_idx in self.static_ch_idx:
            ch = self.category[ch_idx]
            terrain = self._load_terrain(ch)
            cache[:, ch_idx] = terrain

    def _load_terrain(self, channel):
        if channel == "CT":
            terrain_path = os.path.join(self.data_dir, "CT", "CT.npy")
            terrain = np.load(terrain_path).astype(np.float32)
            if terrain.ndim == 1:
                terrain = terrain.reshape(500, 900)
            if self.region_idx is not None:
                terrain = terrain[self.region_idx[0][0]:self.region_idx[0][1],
                                  self.region_idx[1][0]:self.region_idx[1][1]]
            return terrain
        raise ValueError(f"未知的静态通道: {channel}")

    def _generate_segment_idx(self, seg_len, seg_info, offset):
        if seg_info["compensation"]:
            last_valid_start = seg_len - self.total_length
        else:
            last_valid_start = seg_len

        for i in range(0, last_valid_start, self.stride):
            start_idx = offset + i
            time = datetime.strptime(seg_info["day"], "%Y%m%d") + timedelta(
                minutes=(seg_info["real_range"][0] + i) * 10
            )
            self.data_idx.append([int(start_idx), time.strftime("%Y%m%d_%H%M")])

    def _save_data_idx(self):
        with open(self.idx_path, "w", encoding="utf-8") as f:
            json.dump(self.data_idx, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _tailor_data(data, data_range):
        if data_range[0] == data_range[1] == "-":
            return data
        elif data_range[0] != "-" and data_range[1] == "-":
            return data[data_range[0]:, ...]
        elif data_range[0] == "-" and data_range[1] != "-":
            return data[:data_range[1] + 1, ...]
        elif data_range[0] != "-" and data_range[1] != "-":
            return data[data_range[0]:data_range[1] + 1, ...]
        else:
            raise ValueError(f"数据范围 {data_range} 格式错误.")

    @staticmethod
    def _format_bytes(num_bytes):
        num_bytes = float(num_bytes)
        units = ["B", "KB", "MB", "GB", "TB"]
        for unit in units:
            if num_bytes < 1024.0:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.2f} PB"

    @staticmethod
    def _print_shm_usage(title):
        total, used, free = shutil.disk_usage("/dev/shm")
        print(
            f"{title}: "
            f"total={Mem_Loader._format_bytes(total)}, "
            f"used={Mem_Loader._format_bytes(used)}, "
            f"free={Mem_Loader._format_bytes(free)}"
        )


class SD_Dataloader(Dataset):
    """
    SDweather 数据集，使用 mmap 软加载方式。
    """

    def __init__(
        self,
        configs,
        mode,
        standardize=True,
        compute_standardizer=None,
        return_label_only=False,
        copy_numpy=True,
        auto_build_cache=True,
    ):
        self.configs = configs
        self.mode = mode

        self.cache_root = "/dev/shm/SD"
        self.standardize = standardize
        self.return_label_only = return_label_only
        self.copy_numpy = copy_numpy
        self.auto_build_cache = auto_build_cache

        self.data_config = getattr(self.configs, "data_config", "default")
        self.category = list(self.configs.in_category)
        self.label_cate = list(self.configs.out_category)
        self.total_seq = list(self.configs.total_seq)
        self.total_length = sum(self.total_seq)
        self.stride = int(self.configs.data_stride)
        self.img_size = list(self.configs.img_size)

        if hasattr(self.configs, "region"):
            self._compute_region_idx()

        self.get_label_idx()

        self.cache_dir = os.path.join(self.cache_root, str(self.data_config))
        self.data_path = os.path.join(self.cache_dir, f"{self.mode}.npy")
        self.idx_path = os.path.join(self.cache_dir, f"{self.mode}_idx.json")

        self._ensure_cache_exists()

        self.data = None
        self.data_idx = self._load_data_idx()

        data = self._open_data()
        self.data_shape = tuple(data.shape)
        self._check_data_shape()

        self.standardizer = None
        if self.standardize:
            self.standardizer = Load_Standardizer(self.configs).standardizer
            if compute_standardizer is None:
                compute_standardizer = self.mode == "train"
            self._init_standardizer(compute_standardizer=compute_standardizer)

        print(
            f"{self.mode} mmap 数据集已加载: "
            f"{len(self.data_idx)} 个样本, data_shape={self.data_shape}"
        )

    def _cache_files_ready(self):
        if not os.path.isdir(self.cache_dir):
            return False
        if not os.path.exists(self.data_path):
            return False
        if not os.path.exists(self.idx_path):
            return False
        return True

    def _ensure_cache_exists(self):
        if self._cache_files_ready():
            print(f"{self.mode} 缓存已存在，直接读取: {self.data_path}")
            return

        if not self.auto_build_cache:
            raise FileNotFoundError(
                f"{self.mode} 缓存不存在，且 auto_build_cache=False。\n"
                f"缺失文件可能是:\n"
                f"  {self.data_path}\n"
                f"  {self.idx_path}"
            )

        os.makedirs(self.cache_dir, exist_ok=True)
        lock_path = os.path.join(self.cache_dir, f".{self.mode}_cache_build.lock")

        try:
            with open(lock_path, "w") as lock_file:
                print(f"{self.mode} 缓存不存在，等待/获取构建锁: {lock_path}")
                fcntl.flock(lock_file, fcntl.LOCK_EX)

                if self._cache_files_ready():
                    print(f"{self.mode} 缓存已由其他进程生成，直接读取: {self.data_path}")
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    return

                print(f"{self.mode} 缓存不存在，开始生成到: {self.cache_dir}")
                Mem_Loader(self.configs, mode=self.mode)

                if not self._cache_files_ready():
                    raise RuntimeError(
                        f"{self.mode} 缓存生成后仍不完整，请检查生成器。\n"
                        f"data_path={self.data_path}\n"
                        f"idx_path={self.idx_path}"
                    )

                print(f"{self.mode} 缓存生成完成: {self.data_path}")
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        except ImportError:
            print("当前环境不支持 fcntl 文件锁，将直接尝试生成缓存。")
            if not self._cache_files_ready():
                Mem_Loader(self.configs, mode=self.mode)
            if not self._cache_files_ready():
                raise RuntimeError(
                    f"{self.mode} 缓存生成后仍不完整，请检查生成器。\n"
                    f"data_path={self.data_path}\n"
                    f"idx_path={self.idx_path}"
                )

    def _open_data(self):
        if self.data is None:
            self.data = np.load(self.data_path, mmap_mode="r")
        return self.data

    def _compute_region_idx(self):
        lat_min = float(self.configs.region["lat"].split('-')[0])
        lat_max = float(self.configs.region["lat"].split('-')[1])
        lon_min = float(self.configs.region["lon"].split('-')[0])
        lon_max = float(self.configs.region["lon"].split('-')[1])

        region_idx = [[], []]
        region_idx[0].append(int((lat_min - 34) * 100))
        region_idx[0].append(int((lat_max -34) * 100))
        region_idx[1].append(int((lon_min - 114) * 100))
        region_idx[1].append(int((lon_max - 114) * 100) + 1)

        self.img_size[0] = region_idx[0][1] - region_idx[0][0]
        self.img_size[1] = region_idx[1][1] - region_idx[1][0]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["data"] = None
        return state

    def _load_data_idx(self):
        with open(self.idx_path, "r", encoding="utf-8") as f:
            data_idx = json.load(f)

        checked_idx = []
        for item in data_idx:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError(
                    f"idx 文件格式错误，期望 [start_idx, label]，但得到: {item}"
                )
            start_idx, label = item
            checked_idx.append((int(start_idx), str(label)))

        return checked_idx

    def _check_data_shape(self):
        if len(self.data_shape) != 4:
            raise ValueError(
                f"{self.data_path} 应该是 [T, C, H, W]，"
                f"但得到 shape={self.data_shape}"
            )

        total_t, channels, height, width = self.data_shape

        if channels != len(self.category):
            raise ValueError(
                f"缓存数据 channel={channels}，但 len(configs.in_category)="
                f"{len(self.category)}，category={self.category}"
            )

        if [height, width] != self.img_size:
            raise ValueError(
                f"缓存数据 H/W={(height, width)}，但 configs.img_size={self.img_size}"
            )

        if len(self.data_idx) > 0:
            max_start = max(item[0] for item in self.data_idx)
            max_needed = max_start + self.total_length
            if max_needed > total_t:
                raise ValueError(
                    f"data_idx 越界: max_start={max_start}, "
                    f"total_length={self.total_length}, total_t={total_t}"
                )

    def _init_standardizer(self, compute_standardizer):
        if compute_standardizer:
            data = torch.from_numpy(np.array(self._open_data()))
            self.standardizer.cal_params(data, self.configs.label_idx)
            del data
        else:
            self.standardizer.data_params()

    def get_label_idx(self):
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

    def _to_tensor(self, array):
        array = np.asarray(array, dtype=np.float32)
        if self.copy_numpy:
            array = array.copy()
        return torch.from_numpy(array)

    def _standardize_window(self, tensor):
        if not self.standardize:
            return tensor
        tensor = self.standardizer.standardizing(tensor)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32)
        return tensor

    def __len__(self):
        return len(self.data_idx)

    def __getitem__(self, idx):
        data = self._open_data()
        start_idx, label_name = self.data_idx[idx]

        input_len = int(self.total_seq[0])
        pred_len = int(self.total_seq[1])
        total_len = input_len + pred_len

        sample_np = data[start_idx:start_idx + input_len, :, :, :]
        label_np = data[start_idx + input_len:start_idx + total_len, :, :, :]

        sample = self._to_tensor(sample_np)
        label = self._to_tensor(label_np)

        sample = self._standardize_window(sample)
        label = self._standardize_window(label)

        if self.return_label_only:
            label_start, label_end = self.configs.label_idx
            label = label[:, label_start:label_end, :, :]

        return sample, label


if __name__ == '__main__':
    from argparse import Namespace

    configs = {
        "data_config": "test",
        "train_range": "point:test",
        "in_category": ["CW", "CR", "CL", "CT"],
        "out_category": ["CW"],
        "data_dir": "/shares/weather/Split_Data",
        "total_seq": [18, 18],
        "img_size": [500, 900],
        "region": {
            "lon": "117-118.8",
            "lat": "37-38"
        },
        "data_stride": 1,
        "std_method": "z_score_sd"
    }

    configs = Namespace(**configs)

    print("=" * 80)
    print("测试 Mem_Loader 缓存生成")
    print("=" * 80)
    Mem_Loader(configs, mode="train")

    print("\n" + "=" * 80)
    print("测试 SD_Dataloader mmap 加载")
    print("=" * 80)
    dataset = SD_Dataloader(configs, "train")

    print(f"\n数据集信息:")
    print(f"  img_size: {configs.img_size}")
    print(f"  samples: {len(dataset.data_idx)}")
    print(f"  data_shape: {dataset.data_shape}")

    sample, label = dataset.__getitem__(len(dataset.data_idx) - 1)
    print(f"\n样本信息:")
    print(f"  sample.shape: {sample.shape}")
    print(f"  label.shape: {label.shape}")
