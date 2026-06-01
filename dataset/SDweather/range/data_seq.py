import os
import json
from datetime import datetime, timedelta
from itertools import product


class DataSeq:
    """
    根据给定日期范围生成数据段索引。

    方案 A:
    1. 常规动态模态，也就是不在 optional_channels 中的 channels，必须存在。
    2. 每一天先对常规动态模态的真实时间范围取交集。
    3. optional 模态：
       - 如果当天完全没有文件，不影响交集，后续由 Mem_Loader 用 0 填充。
       - 如果当天有文件，则也参与最终交集。
    4. static 模态不应该传入这里；由 Mem_Loader 最后广播。
    """

    def __init__(self, range_name, data_path, mode, channels=None, optional_channels=None):
        self.range = self.get_range_file(range_name, mode)
        self.data_list = []
        self.data_path = data_path

        self.channels = list(channels) if channels else []
        self.optional_channels = set(optional_channels) if optional_channels else set()
        self.required_channels = [
            ch for ch in self.channels
            if ch not in self.optional_channels
        ]

        if not self.channels:
            raise ValueError("channels 不能为空。")

        if not self.required_channels:
            raise ValueError(
                "方案 A 需要至少一个常规动态模态，也就是至少一个不在 optional_channels 中的 channel。"
            )

        self.cal_seq()

    def get_range_file(self, range_name, mode):
        if mode == "point":
            range_file = os.path.join(os.path.dirname(__file__), range_name + ".json")
            if not os.path.exists(range_file):
                raise FileNotFoundError(f"未找到指定区间文件: {range_name}.json")
            with open(range_file, "r", encoding="utf-8") as f:
                file = json.load(f)
        elif mode == "range":
            date = range_name.split("-")
            file = [f"{date[0]}_0-{date[1]}_143"]
        else:
            raise ValueError(f"未支持的模式: {mode}")
        return file

    def _get_channel_path(self, channel, year):
        return os.path.join(self.data_path, channel, str(year))

    def _parse_file_range(self, filename, channel):
        base = filename.replace(".npy", "")
        parts = base.split("_")
        start_idx = int(parts[-2])
        end_idx = int(parts[-1])
        return start_idx, end_idx

    def _list_channel_files(self, channel, day, year):
        """
        返回某个 channel 在某一天的所有文件信息。

        文件名沿用原逻辑：
            CW20200101_0_143.npy
            CR20200101_20_143.npy
            CL20200101_0_143.npy

        即：
        - channel 后 8 位是日期
        - 最后两个下划线字段是 start_idx / end_idx
        """
        channel_dir = self._get_channel_path(channel, year)
        if not os.path.isdir(channel_dir):
            return []

        files = []
        for filename in sorted(os.listdir(channel_dir)):
            if not filename.endswith(".npy"):
                continue

            file_day = filename[len(channel):len(channel) + 8]
            if file_day != day:
                continue

            try:
                start_idx, end_idx = self._parse_file_range(filename, channel)
            except (ValueError, IndexError):
                continue

            if start_idx > end_idx:
                continue

            files.append({
                "path": os.path.join(channel_dir, filename),
                "start": start_idx,
                "end": end_idx,
            })

        return files

    @staticmethod
    def _overlap_len(a_start, a_end, b_start, b_end):
        start = max(a_start, b_start)
        end = min(a_end, b_end)
        if start > end:
            return 0
        return end - start + 1

    def _filter_overlap_files(self, files, req_start, req_end):
        return [
            f for f in files
            if self._overlap_len(f["start"], f["end"], req_start, req_end) > 0
        ]

    def _choose_best_file_combination(self, channel_to_files, req_start, req_end):
        """
        在每个参与交集的 channel 中各选一个文件，使最终交集长度最大。

        这一步是真正替代原先“以 self.channels[0] 为参考”的核心。
        现在不再让第一个通道决定范围，而是所有常规动态模态 + 有数据的 optional
        共同决定最终 real_range。
        """
        channels = list(channel_to_files.keys())
        file_lists = [channel_to_files[ch] for ch in channels]

        best = None
        best_score = None

        for combo in product(*file_lists):
            inter_start = req_start
            inter_end = req_end

            for file_info in combo:
                inter_start = max(inter_start, file_info["start"])
                inter_end = min(inter_end, file_info["end"])

            if inter_start > inter_end:
                continue

            length = inter_end - inter_start + 1

            # 排序规则：
            # 1. 交集越长越好
            # 2. 开始时间越早越好
            # 3. 结束时间越早越好，保证结果确定
            score = (length, -inter_start, -inter_end)

            if best is None or score > best_score:
                best = {
                    "channels": channels,
                    "combo": combo,
                    "inter_start": inter_start,
                    "inter_end": inter_end,
                }
                best_score = score

        return best

    @staticmethod
    def _build_data_range(file_start, file_end, inter_start, inter_end):
        """
        将最终交集 [inter_start, inter_end] 转为该文件内部的裁剪范围。

        返回格式兼容 Mem_Loader._tailor_data:

            ["-", "-"]           不裁剪
            [start_offset, "-"]  从 start_offset 裁到末尾
            ["-", end_offset]    从开头裁到 end_offset
            [start_offset, end_offset]
        """
        if inter_start < file_start or inter_end > file_end:
            raise ValueError(
                f"交集 [{inter_start}, {inter_end}] 超出文件范围 [{file_start}, {file_end}]"
            )

        data_start = "-" if inter_start == file_start else inter_start - file_start
        data_end = "-" if inter_end == file_end else inter_end - file_start
        return [data_start, data_end]

    def cal_seq(self):
        for r in self.range:
            start_inf, end_inf = r.split("-")
            start_inf = start_inf.split("_")
            end_inf = end_inf.split("_")

            start_day, start_idx = start_inf[0], int(start_inf[1])
            end_day, end_idx = end_inf[0], int(end_inf[1])

            start_date = datetime.strptime(start_day, "%Y%m%d")
            end_date = datetime.strptime(end_day, "%Y%m%d")

            if start_date == end_date:
                self._process_day(start_date, start_idx, end_idx)
            else:
                self._process_day(start_date, start_idx, 143)

                current = start_date + timedelta(days=1)
                while current < end_date:
                    self._process_day(current, 0, 143)
                    current += timedelta(days=1)

                self._process_day(end_date, 0, end_idx)

    def _process_day(self, date, req_start, req_end):
        """
        处理单日的一个请求时间窗口。

        req_start / req_end 是这一天内希望使用的时间范围，例如：
            first day:  start_idx - 143
            middle day: 0 - 143
            last day:   0 - end_idx
            single day: start_idx - end_idx
        """
        if req_start > req_end:
            return

        day = date.strftime("%Y%m%d")
        year = date.strftime("%Y")

        channel_to_files = {}
        missing_optional_channels = set()

        # 1. 常规动态模态必须存在，并参与交集。
        for ch in self.required_channels:
            files = self._list_channel_files(ch, day, year)
            files = self._filter_overlap_files(files, req_start, req_end)
            if not files:
                return
            channel_to_files[ch] = files

        # 2. optional 模态：
        #    - 当天没有文件：不参与交集，后面补 0。
        #    - 当天有文件：参与交集。
        for ch in self.channels:
            if ch not in self.optional_channels:
                continue

            all_files = self._list_channel_files(ch, day, year)
            if not all_files:
                missing_optional_channels.add(ch)
                continue

            files = self._filter_overlap_files(all_files, req_start, req_end)
            if not files:
                # 方案 A：
                # optional 当天有数据，但与当前请求窗口无交集，
                # 那么最终交集为空，这一天丢掉。
                return

            channel_to_files[ch] = files

        # 3. 在所有参与交集的通道中，寻找最大公共时间交集。
        best = self._choose_best_file_combination(
            channel_to_files,
            req_start=req_start,
            req_end=req_end,
        )
        if best is None:
            return

        inter_start = best["inter_start"]
        inter_end = best["inter_end"]
        real_range = [inter_start, inter_end]

        selected_by_channel = {
            ch: file_info
            for ch, file_info in zip(best["channels"], best["combo"])
        }

        # 4. 按 self.channels 的顺序生成 modalities，保持和 Mem_Loader 的通道顺序一致。
        modalities = []
        for ch in self.channels:
            if ch in missing_optional_channels:
                modalities.append(["-", ["-", "-"]])
                continue

            file_info = selected_by_channel.get(ch)
            if file_info is None:
                # 理论上不会走到这里，除非 channels / optional_channels 配置不一致。
                if ch in self.optional_channels:
                    modalities.append(["-", ["-", "-"]])
                    continue
                return

            data_range = self._build_data_range(
                file_start=file_info["start"],
                file_end=file_info["end"],
                inter_start=inter_start,
                inter_end=inter_end,
            )
            modalities.append([file_info["path"], data_range])

        self.data_list.append([day, real_range, modalities])


if __name__ == "__main__":
    data_seq = DataSeq(
        range_name="r1",
        data_path="/shares/weather/Split_Data",
        mode="point",
        channels=["CW", "CR", "CL"],
        optional_channels=["CL"],
    )

    print(f"数据段数量: {len(data_seq.data_list)}")
    print(f"通道: {data_seq.channels}")
    print(f"常规动态模态: {data_seq.required_channels}")
    print(f"optional 模态: {sorted(data_seq.optional_channels)}")
    print()

    for i, item in enumerate(data_seq.data_list[:3]):
        print(f"=== 数据段 {i} ===")
        print(f"  日期: {item[0]}")
        print(f"  时间范围: {item[1]}")
        print(f"  模态数量: {len(item[2])}")
        for j, modality in enumerate(item[2]):
            print(f"    模态{j}: path={modality[0]}")
            print(f"           range={modality[1]}")
        print()

    with open("data_seq_debug.json", "w", encoding="utf-8") as f:
        json.dump(data_seq.data_list, f, ensure_ascii=False, indent=2)
    print("已保存到 data_seq_debug.json")