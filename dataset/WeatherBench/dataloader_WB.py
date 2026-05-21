import os
import json
import shutil
import calendar
from argparse import Namespace
import numpy as np
import xarray as xr
from numpy.lib.format import open_memmap
import torch
from torch.utils.data import Dataset
from Instrument.standardizer import Load_Standardizer

class Mem_Dataloader:
    """
    将 WeatherBench 数据逐年写入 /dev/shm 中的 .npy memmap 缓存。

    输出：
        /dev/shm/M/{data_config}/{mode}.npy
        /dev/shm/M/{data_config}/{mode}_idx.json

    数据形状：
        [T, C, H, W]

    其中：
        T = 所选年份总小时数，闰年 366*24，平年 365*24
        C = len(configs.in_category)
        H, W = configs.img_size
    """

    def __init__(self, configs, mode):
        self.configs = configs
        self.mode = mode

        self.data_dir = self.configs.data_dir
        self.data_dev = self.configs.data_config

        self.category = list(self.configs.in_category)
        self.label_cate = list(self.configs.out_category)
        self.total_seq = list(self.configs.total_seq)

        self.img_size = tuple(int(x) for x in self.configs.img_size)
        if len(self.img_size) != 2:
            raise ValueError(
                f"configs.img_size 应该是 [H, W]，但得到 {self.configs.img_size}"
            )
        self.img_h, self.img_w = self.img_size

        self.resolusion = self.configs.resolusion
        self.stride = int(self.configs.data_stride)

        self.data_idx = []
        self.data = None

        if self.mode == "train":
            self.data_range = self.configs.train_range
        elif self.mode == "valid":
            self.data_range = self.configs.valid_range
        elif self.mode == "test":
            self.data_range = self.configs.test_range
            self.total_seq[1] = int(getattr(self.configs, "test_seq", self.total_seq[1]))
        else:
            raise ValueError("mode should be 'train', 'valid', or 'test'.")

        if self.mode == "train":
            self.get_lable_idx()

        self.cache_dir = os.path.join("/dev/shm", "M", str(self.data_dev))
        os.makedirs(self.cache_dir, exist_ok=True)

        self.cache_path = os.path.join(self.cache_dir, f"{self.mode}.npy")
        self.idx_path = os.path.join(self.cache_dir, f"{self.mode}_idx.json")

        # WeatherBench 目录名与 nc 内变量名映射。
        # 你 configs 里写的是 ["tp", "tcc", "u", "v", "orography"]，
        # 这里把 u/v 映射到 WeatherBench 10m 风变量 u10/v10。
        self.variable_map = {
            "tcc": ("total_cloud_cover", "tcc"),
            "tp": ("total_precipitation", "tp"),
            "t2m": ("2m_temperature", "t2m"),

            "z500": ("geopotential_500", "z500"),
            "t850": ("temperature_850", "t850"),

            "u": ("10m_u_component_of_wind", "u10"),
            "v": ("10m_v_component_of_wind", "v10"),
            "u10": ("10m_u_component_of_wind", "u10"),
            "v10": ("10m_v_component_of_wind", "v10"),

            "orography": ("constants", "orography"),
            "lsm": ("constants", "lsm"),
        }

        self.generate_cache()

        print(f"\n{self.mode} 缓存生成完成")
        print(f"  data: {self.cache_path}")
        print(f"  idx : {self.idx_path}")
        print(f"  samples: {len(self.data_idx)}")

    def generate_cache(self):
        reso_dir = self.get_reso_dir()

        start_year = int(self.data_range[0])
        end_year = int(self.data_range[1])
        years = list(range(start_year, end_year + 1))

        if len(years) == 0:
            raise ValueError(f"{self.mode} data_range 不合法: {self.data_range}")

        channels = self.compute_total_channels()
        total_time = sum(self.year_hours(year) for year in years)

        expected_shape = (total_time, channels, self.img_h, self.img_w)
        expected_bytes = np.prod(expected_shape) * np.dtype("float32").itemsize

        print("\n" + "=" * 80)
        print(f"开始生成 {self.mode} 缓存")
        print(f"年份范围: {years[0]} - {years[-1]}")
        print(f"缓存路径: {self.cache_path}")
        print(f"缓存 shape: {expected_shape}")
        print(f"预计大小: {self.format_bytes(expected_bytes)}")
        print("=" * 80)

        self.print_shm_usage("生成前 /dev/shm 使用情况")

        cache = open_memmap(
            self.cache_path,
            mode="w+",
            dtype="float32",
            shape=expected_shape,
        )

        offset = 0
        base_line = 0

        for year in years:
            year_data = self.load_one_year_data(reso_dir, year)
            year_data = self.adjust_geo_view(year_data)

            offset, base_line = self.write_year_to_cache(
                cache=cache,
                year=year,
                year_data=year_data,
                offset=offset,
                base_line=base_line,
            )

            del year_data

            self.print_shm_usage(f"{year} 年写入后 /dev/shm 使用情况")

        if offset != total_time:
            raise RuntimeError(
                f"写入结束后 offset={offset}，但预期 total_time={total_time}。"
            )

        cache.flush()
        self.save_data_idx()

        # 这里只保留 mmap 句柄，不会把完整数据加载进 Python 内存。
        self.data = np.load(self.cache_path, mmap_mode="r")

    def get_reso_dir(self):
        if self.resolusion == "5.625":
            return os.path.join(self.data_dir, "WB", "5_625deg")
        if self.resolusion == "2.8125":
            return os.path.join(self.data_dir, "WB", "2_8125deg")
        if self.resolusion == "1.40625":
            return os.path.join(self.data_dir, "WB", "1_40625deg")

        raise ValueError(f"分辨率 {self.resolusion} 不存在")

    def compute_total_channels(self):
        """
        当前每个变量默认都是 1 个 channel。
        如果后续某个变量本身有多层 channel，可以在这里扩展。
        """
        return len(self.category)

    @staticmethod
    def is_leap_year(year):
        return calendar.isleap(int(year))

    @classmethod
    def year_hours(cls, year):
        return 366 * 24 if cls.is_leap_year(year) else 365 * 24

    def load_one_year_data(self, reso_dir, year):
        """
        加载单年数据，返回 numpy.float32:
            [T, C, H, W]

        这个函数只会在内存中保留当前 year 的数据。
        """

        expected_t = self.year_hours(year)
        year_data = None

        for cate in self.category:
            dataset = self.load_one_category(reso_dir, year, cate)
            dataset = self.ensure_tchw(dataset, expected_t=expected_t, cate=cate)

            if year_data is None:
                year_data = dataset
            else:
                year_data = self.concat_channel_with_broadcast(
                    year_data=year_data,
                    dataset=dataset,
                    cate=cate,
                    year=year,
                )

            del dataset

        year_data = np.asarray(year_data, dtype=np.float32)

        if year_data.shape[0] == 1 and expected_t > 1:
            year_data = np.broadcast_to(
                year_data,
                (expected_t, year_data.shape[1], year_data.shape[2], year_data.shape[3]),
            ).copy()

        actual_t = year_data.shape[0]
        if actual_t != expected_t:
            raise ValueError(
                f"{year} 年数据 T={actual_t}，但根据闰年判断预期 T={expected_t}。"
                f"请检查 nc 文件是否为逐小时数据，或者是否缺失/多出时间步。"
            )

        if year_data.shape[-2:] != self.img_size:
            raise ValueError(
                f"{year} 年加载后的空间尺寸为 {year_data.shape[-2:]}，"
                f"但 configs.img_size={self.img_size}。"
            )

        print(f"{year} 年加载完成: shape={year_data.shape}")
        return year_data

    def load_one_category(self, reso_dir, year, cate):
        """
        加载单个变量。

        动态变量一般返回：
            [T, H, W]

        常量变量返回：
            [H, W]
        """

        if cate == "w10":
            return self.load_w10(reso_dir, year)

        if cate not in self.variable_map:
            raise ValueError(
                f"变量 {cate} 不在 variable_map 中。"
                f"请检查 configs.in_category 或补充 variable_map。"
            )

        folder_name, var_name = self.variable_map[cate]

        if cate in {"orography", "lsm"}:
            path = os.path.join(
                reso_dir,
                "constants",
                f"constants_{self.resolusion}deg.nc",
            )
        else:
            path = os.path.join(
                reso_dir,
                folder_name,
                f"{folder_name}_{str(year)}_{self.resolusion}deg.nc",
            )

        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}")

        with xr.open_dataset(path) as data:
            if var_name in data:
                dataset = data[var_name].values
            elif cate in data:
                dataset = data[cate].values
            else:
                raise KeyError(
                    f"文件 {path} 中找不到变量 {var_name} 或 {cate}。"
                    f"该文件包含变量: {list(data.data_vars)}"
                )

        return np.asarray(dataset, dtype=np.float32)

    def load_w10(self, reso_dir, year):
        """
        由 u10/v10 计算 w10，并做 gust 增强。
        """

        u_folder = "10m_u_component_of_wind"
        v_folder = "10m_v_component_of_wind"

        path_u = os.path.join(
            reso_dir,
            u_folder,
            f"{u_folder}_{str(year)}_{self.resolusion}deg.nc",
        )
        path_v = os.path.join(
            reso_dir,
            v_folder,
            f"{v_folder}_{str(year)}_{self.resolusion}deg.nc",
        )

        if not os.path.exists(path_u):
            raise FileNotFoundError(f"数据文件不存在: {path_u}")
        if not os.path.exists(path_v):
            raise FileNotFoundError(f"数据文件不存在: {path_v}")

        with xr.open_dataset(path_u) as data_u, xr.open_dataset(path_v) as data_v:
            if "u10" not in data_u:
                raise KeyError(f"{path_u} 中找不到变量 u10，包含变量: {list(data_u.data_vars)}")
            if "v10" not in data_v:
                raise KeyError(f"{path_v} 中找不到变量 v10，包含变量: {list(data_v.data_vars)}")

            u = data_u["u10"].values.astype(np.float32, copy=False)
            v = data_v["v10"].values.astype(np.float32, copy=False)

            wind = np.sqrt(u ** 2 + v ** 2).astype(np.float32, copy=False)
            gust = self.enhance_gust_forecast(wind)

        return np.asarray(gust, dtype=np.float32)

    def ensure_tchw(self, dataset, expected_t, cate):
        """
        统一为 [T, C, H, W]。

        [T, H, W] -> [T, 1, H, W]
        [H, W]    -> [1, 1, H, W]
        """

        dataset = np.asarray(dataset, dtype=np.float32)

        if dataset.ndim == 3:
            if dataset.shape[0] != expected_t:
                raise ValueError(
                    f"变量 {cate} 的 T={dataset.shape[0]}，但预期 T={expected_t}。"
                )
            dataset = np.expand_dims(dataset, axis=1)

        elif dataset.ndim == 2:
            dataset = dataset[np.newaxis, np.newaxis, :, :]

        else:
            raise ValueError(
                f"变量 {cate} 的维度不支持: ndim={dataset.ndim}, shape={dataset.shape}"
            )

        if dataset.shape[-2:] != self.img_size:
            raise ValueError(
                f"变量 {cate} 的空间尺寸为 {dataset.shape[-2:]}，"
                f"但 configs.img_size={self.img_size}。"
            )

        return np.asarray(dataset, dtype=np.float32)

    def concat_channel_with_broadcast(self, year_data, dataset, cate, year):
        """
        沿 channel 维拼接。

        year_data: [T1, C1, H, W]
        dataset:   [T2, C2, H, W]

        如果某个变量是常量，T=1，则 broadcast 到另一方的 T。
        """

        t1 = year_data.shape[0]
        t2 = dataset.shape[0]

        if year_data.shape[-2:] != dataset.shape[-2:]:
            raise ValueError(
                f"{year} 年变量 {cate} 空间尺寸不一致: "
                f"year_data={year_data.shape[-2:]}, dataset={dataset.shape[-2:]}"
            )

        if t1 == t2:
            return np.concatenate((year_data, dataset), axis=1)

        if t1 == 1 and t2 > 1:
            year_data = np.broadcast_to(
                year_data,
                (t2, year_data.shape[1], year_data.shape[2], year_data.shape[3]),
            )
            return np.concatenate((year_data, dataset), axis=1)

        if t2 == 1 and t1 > 1:
            dataset = np.broadcast_to(
                dataset,
                (t1, dataset.shape[1], dataset.shape[2], dataset.shape[3]),
            )
            return np.concatenate((year_data, dataset), axis=1)

        raise ValueError(
            f"{year} 年变量 {cate} 的 T 维无法对齐: "
            f"year_data T={t1}, dataset T={t2}"
        )

    def write_year_to_cache(self, cache, year, year_data, offset, base_line):
        """
        将单年数据写入 cache。

        cache[offset : offset + T_year] = year_data
        """

        expected_t = self.year_hours(year)
        actual_t = year_data.shape[0]

        if actual_t != expected_t:
            raise ValueError(
                f"{year} 年 year_data.shape[0]={actual_t}，但预期为 {expected_t}。"
            )

        if year_data.shape[1] != cache.shape[1]:
            raise ValueError(
                f"{year} 年 channel={year_data.shape[1]}，但 cache channel={cache.shape[1]}。"
            )

        if year_data.shape[-2:] != self.img_size:
            raise ValueError(
                f"{year} 年数据空间尺寸为 {year_data.shape[-2:]}，"
                f"但 configs.img_size={self.img_size}。"
            )

        end = offset + actual_t

        if end > cache.shape[0]:
            raise RuntimeError(
                f"{year} 年写入越界: offset={offset}, actual_t={actual_t}, "
                f"cache.shape[0]={cache.shape[0]}"
            )

        cache[offset:end] = year_data

        base_line = self.get_data_idx_from_year(
            total_len=actual_t,
            year=year,
            base_line=base_line,
        )

        print(
            f"{year} 年写入完成: "
            f"cache[{offset}:{end}], shape={year_data.shape}"
        )

        return end, base_line

    def get_data_idx_from_year(self, total_len, year, base_line):
        """
        按每一年内部生成样本索引，不允许样本跨年。

        data_idx 中 start_idx 是全局时间轴索引。
        """

        last_valid_start = total_len - sum(self.total_seq) + 1

        if last_valid_start <= 0:
            raise ValueError(
                f"{year} 年 total_len={total_len}，total_seq={self.total_seq}，"
                f"无法生成样本。"
            )

        for i in range(0, last_valid_start, self.stride):
            start_idx = base_line + i
            label = f"{year}_{i}"
            self.data_idx.append([int(start_idx), label])

        base_line += total_len
        return base_line

    def save_data_idx(self):
        with open(self.idx_path, "w", encoding="utf-8") as f:
            json.dump(self.data_idx, f, ensure_ascii=False, indent=2)

    @staticmethod
    def enhance_gust_forecast(wind_array):
        wind = wind_array.astype(np.float32, copy=True)
        gust = np.zeros_like(wind, dtype=np.float32)

        gust += (wind <= 3.3) * wind * 1.5
        gust += ((wind > 3.3) & (wind <= 5.4)) * wind * 1.8
        gust += ((wind > 5.4) & (wind <= 7.9)) * wind * 1.75
        gust += ((wind > 7.9) & (wind <= 10.7)) * wind * 1.6
        gust += ((wind > 10.7) & (wind <= 13.8)) * wind * 1.6
        gust += (wind > 13.8) * wind * 1.6

        return gust.astype(np.float32, copy=False)

    @staticmethod
    def adjust_geo_view(data):
        """
        NumPy 版本地理视角调整。

        输入:
            [T, C, H, W]

        操作:
            1. H 维度上下翻转。
            2. W 维度滚动一半宽度。
        """

        data = np.asarray(data, dtype=np.float32)

        width = data.shape[-1]

        data = np.flip(data, axis=-2)
        data = np.roll(data, shift=width // 2, axis=-1)

        return np.asarray(data, dtype=np.float32)

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
                raise ValueError(
                    f"输出类别 {self.label_cate} 必须在输入中相邻顺序排列。"
                )

        else:
            raise ValueError(f"请设置输出类别 {self.label_cate}。")

    @staticmethod
    def format_bytes(num_bytes):
        num_bytes = float(num_bytes)
        units = ["B", "KB", "MB", "GB", "TB"]
        for unit in units:
            if num_bytes < 1024.0:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.2f} PB"

    @staticmethod
    def print_shm_usage(title):
        total, used, free = shutil.disk_usage("/dev/shm")
        print(
            f"{title}: "
            f"total={Mem_Dataloader.format_bytes(total)}, "
            f"used={Mem_Dataloader.format_bytes(used)}, "
            f"free={Mem_Dataloader.format_bytes(free)}"
        )


class WB_Dataloader(Dataset):
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

        self.cache_root = "/dev/shm/M"
        self.standardize = standardize
        self.return_label_only = return_label_only
        self.copy_numpy = copy_numpy
        self.auto_build_cache = auto_build_cache

        self.data_dir = self.configs.data_dir
        self.data_dev = self.configs.data_config

        self.category = list(self.configs.in_category)
        self.label_cate = list(self.configs.out_category)
        self.total_seq = list(self.configs.total_seq)

        self.resolusion = self.configs.resolusion
        self.stride = self.configs.data_stride

        self.img_size = tuple(int(x) for x in self.configs.img_size)
        if len(self.img_size) != 2:
            raise ValueError(
                f"configs.img_size 应该是 [H, W]，但得到 {self.configs.img_size}"
            )

        if self.mode == "train":
            self.data_range = self.configs.train_range
        elif self.mode == "valid":
            self.data_range = self.configs.valid_range
        elif self.mode == "test":
            self.data_range = self.configs.test_range
            self.total_seq[1] = int(getattr(self.configs, "test_seq", self.total_seq[1]))
        else:
            raise ValueError("mode should be 'train', 'valid', or 'test'.")

        self.get_lable_idx()

        self.cache_dir = os.path.join(self.cache_root, str(self.data_dev))
        self.data_path = os.path.join(self.cache_dir, f"{self.mode}.npy")
        self.idx_path = os.path.join(self.cache_dir, f"{self.mode}_idx.json")

        # 核心新增逻辑：
        # 如果缓存存在，直接读取；
        # 如果缓存不存在，先生成，再读取。
        self.ensure_cache_exists()

        self.data = None
        self.data_idx = self.load_data_idx()

        data = self._open_data()
        self.data_shape = tuple(data.shape)
        self.check_data_shape()

        self.standardizer = None
        if self.standardize:
            self.standardizer = Load_Standardizer(self.configs).standardizer

            if compute_standardizer is None:
                compute_standardizer = self.mode == "train"

            self.init_standardizer(compute_standardizer=compute_standardizer)

        print(
            f"{self.mode} mmap 数据集已加载: "
            f"{len(self.data_idx)} 个样本, data_shape={self.data_shape}"
        )

    def cache_files_ready(self):
        """
        检查当前 mode 的缓存是否完整存在。

        不能只检查 /dev/shm/M/{data_config} 文件夹是否存在，
        因为文件夹存在不代表 train.npy / train_idx.json 已经生成完成。
        """

        if not os.path.isdir(self.cache_dir):
            return False

        if not os.path.exists(self.data_path):
            return False

        if not os.path.exists(self.idx_path):
            return False

        try:
            data = np.load(self.data_path, mmap_mode="r")
            if data.ndim != 4:
                return False

            with open(self.idx_path, "r", encoding="utf-8") as f:
                idx = json.load(f)

            if not isinstance(idx, list):
                return False

        except Exception:
            return False

        return True

    def ensure_cache_exists(self):
        """
        如果 /dev/shm/M/{data_config}/{mode}.npy 和 {mode}_idx.json 已经存在，
        直接使用。

        如果不存在，则调用 Mem_Dataloader(configs, mode) 生成缓存。

        为了避免 DDP 多进程同时生成同一个文件，这里加了一个简单文件锁。
        注意：
            这个锁依赖 Linux fcntl，适合 /dev/shm 场景。
        """

        if self.cache_files_ready():
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
            import fcntl

            with open(lock_path, "w") as lock_file:
                print(f"{self.mode} 缓存不存在，等待/获取构建锁: {lock_path}")
                fcntl.flock(lock_file, fcntl.LOCK_EX)

                # 拿到锁之后要再次检查。
                # 因为可能别的进程已经在等待期间生成好了。
                if self.cache_files_ready():
                    print(f"{self.mode} 缓存已由其他进程生成，直接读取: {self.data_path}")
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
                    return

                print(f"{self.mode} 缓存不存在，开始生成到: {self.cache_dir}")

                # 这里调用你前面写好的 open_memmap 生成器。
                # 要确保 Mem_Dataloader 类已经 import 或在同一个文件中。
                Mem_Dataloader(self.configs, mode=self.mode)

                if not self.cache_files_ready():
                    raise RuntimeError(
                        f"{self.mode} 缓存生成后仍不完整，请检查生成器。\n"
                        f"data_path={self.data_path}\n"
                        f"idx_path={self.idx_path}"
                    )

                print(f"{self.mode} 缓存生成完成: {self.data_path}")

                fcntl.flock(lock_file, fcntl.LOCK_UN)

        except ImportError:
            # 极少数非 Linux 环境没有 fcntl。
            # 但 /dev/shm 基本就是 Linux，所以正常不会走到这里。
            print("当前环境不支持 fcntl 文件锁，将直接尝试生成缓存。")

            if not self.cache_files_ready():
                Mem_Dataloader(self.configs, mode=self.mode)

            if not self.cache_files_ready():
                raise RuntimeError(
                    f"{self.mode} 缓存生成后仍不完整，请检查生成器。\n"
                    f"data_path={self.data_path}\n"
                    f"idx_path={self.idx_path}"
                )

    def _open_data(self):
        """
        懒加载 mmap。

        注意：
            np.load(..., mmap_mode='r') 不会把完整 .npy 读入进程内存。
        """

        if self.data is None:
            self.data = np.load(self.data_path, mmap_mode="r")

        return self.data

    def __getstate__(self):
        """
        DataLoader 多 worker 时，避免把 mmap 句柄 pickle 到 worker。
        worker 进程里会重新 _open_data()。
        """

        state = self.__dict__.copy()
        state["data"] = None
        return state

    def load_data_idx(self):
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

    def check_data_shape(self):
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

        if (height, width) != self.img_size:
            raise ValueError(
                f"缓存数据 H/W={(height, width)}，但 configs.img_size={self.img_size}"
            )

        if len(self.data_idx) > 0:
            max_start = max(item[0] for item in self.data_idx)
            max_needed = max_start + sum(self.total_seq)

            if max_needed > total_t:
                raise ValueError(
                    f"data_idx 越界: max_start={max_start}, "
                    f"sum(total_seq)={sum(self.total_seq)}, total_t={total_t}"
                )

    def init_standardizer(self, compute_standardizer):
        """
        标准化参数初始化。

        train:
            可以计算参数。

        valid/test:
            应该读取 train 已保存的参数。

        注意：
            这里不执行 self.data = self.standardizer.standardizing(self.data)，
            否则会把完整 mmap 数据变成一份新的内存数据。
        """

        if compute_standardizer:
            self.standardizer.cal_params(self._open_data(), self.configs.label_idx)
        else:
            self.standardizer.data_params()

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
                raise ValueError(
                    f"输出类别 {self.label_cate} 必须在输入中相邻顺序排列。"
                )

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

        sample_np = data[
            start_idx : start_idx + input_len,
            :,
            :,
            :,
        ]

        label_np = data[
            start_idx + input_len : start_idx + total_len,
            :,
            :,
            :,
        ]

        sample = self._to_tensor(sample_np)
        label = self._to_tensor(label_np)

        sample = self._standardize_window(sample)
        label = self._standardize_window(label)

        if self.return_label_only:
            label_start, label_end = self.configs.label_idx
            label = label[:, label_start:label_end, :, :]

        return sample, label

if __name__ == "__main__":
    configs = {
        "data_dir": "/scratch/mingze/data",
        "data_config": "test",

        "train_range": ["2010", "2012"],
        "valid_range": ["2016", "2016"],
        "test_range": ["2017", "2018"],

        # 注意：
        # 这里你写的是 u/v。
        # 代码里已经将 u -> u10, v -> v10。
        "in_category": ["tp", "tcc", "u", "v", "orography"],
        "out_category": ["tp"],

        "total_seq": [12, 12],
        "img_size": [32, 64],
        "resolusion": "5.625",
        "data_stride": 1,

        "metrics": ["mae", "mse", "rmse"],
        "threshold": [[0.0001, 0.001, 0.0005]],
        "std_method": "z_score",
    }

    configs = Namespace(**configs)

    # 依次生成 train / valid / test。
    # 如果你只想先测试 train，可以只保留第一行。
    Mem_Dataloader(configs, mode="train")
    Mem_Dataloader(configs, mode="valid")
    Mem_Dataloader(configs, mode="test")