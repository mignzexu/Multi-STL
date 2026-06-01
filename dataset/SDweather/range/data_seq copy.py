import os
import json
from datetime import datetime, timedelta


# 计算数据序列的长度

class DataSeq:
    def __init__(self, range_name, data_path, mode):

        self.range = self.get_range_file(range_name, mode) #["20240504_60-20240505_80", "20240505_90-20240506_90"]
        self.data_list = [] #[[real_day, real_seq, data_seq, path], ...]
        self.data_path = data_path
        self.cal_seq()

    def get_range_file(self, range_name, mode):
        if mode == "point":
            
            range_file = os.path.join(os.path.dirname(__file__), range_name + '.json')
            if not os.path.exists(range_file):
                raise FileNotFoundError(f'未找到指定区间文件: {range_name}.json')
            with open(range_file, 'r', encoding='utf-8') as f:
                file = json.load(f)

        elif mode == "range":

            date = range_name.split("-")
            file = [f"{date[0]}_0-{date[1]}_143"]

        else:
            raise ValueError(f"未支持的模式: {mode}")
        
        return file
    
    def cal_seq(self):
        data_path = self.data_path
        for r in self.range:
            start_inf, end_inf = r.split("-")
            start_inf = start_inf.split("_")
            end_inf = end_inf.split("_")
            start_day , start_idx = start_inf[0], int(start_inf[1])
            end_day, end_idx = end_inf[0], int(end_inf[1])

            start_date = datetime.strptime(start_day, "%Y%m%d")
            end_date = datetime.strptime(end_day, "%Y%m%d")

            #首天计算

            if start_date == end_date:
                file_dir = os.path.join(data_path, start_date.strftime("%Y"))
                day = start_date.strftime("%Y%m%d")
                inf = [day, [], []]
                for files in os.listdir(file_dir):
                    if files[7:15] == day:
                        file_path = os.path.join(file_dir, files)
                        inf.append(file_path)
                        files_inf = files.split("_")

                        start_real = int(files_inf[2])
                        end_real = int(files_inf[3][:-4])

                        if start_real > end_idx or end_real < start_idx:
                            inf = []
                            break

                        if start_real < start_idx :
                            real_idx = start_idx - start_real
                            inf[1].append(start_idx)
                            inf[2].append(real_idx)
                        else:
                            inf[1].append(start_real)
                            inf[2].append("-")

                        if end_real < end_idx:
                            inf[1].append(end_real)
                            inf[2].append("-")

                        elif inf[2][0] == "-":
                            inf[1].append(end_idx)
                            inf[2].append(end_idx - start_real)

                        else:
                            inf[1].append(end_real)
                            inf[2].append("-")
                        break
                
                if len(inf) > 3:
                    self.data_list.append(inf)


            else:

                file_dir = os.path.join(data_path, start_date.strftime("%Y"))
                day = start_date.strftime("%Y%m%d")
                for files in os.listdir(file_dir):
                    if files[7:15] == day:
                        file_path = os.path.join(file_dir, files)
                        files_inf = files.split("_")
                        start_real = int(files_inf[2])
                        end_real = int(files_inf[3][:-4])
                        if end_real < start_idx:
                            break
                        if start_real < start_idx :
                            real_idx = start_idx - start_real
                            self.data_list.append([day, [start_idx, end_real], [real_idx, "-"], file_path])
                        elif start_real == start_idx:
                                self.data_list.append([day, [start_idx, end_real], ["-", "-"], file_path])
                        else:
                            self.data_list.append([day, [start_real, end_real], ["-", "-"], file_path])
                        break

                start_date += timedelta(days=1)

                #中间天计算

                while start_date < end_date:
                    file_dir = os.path.join(data_path, start_date.strftime("%Y"))
                    day = start_date.strftime("%Y%m%d")
                    for files in os.listdir(file_dir):
                        if files[7:15] == day:
                            file_path = os.path.join(file_dir, files)
                            files_inf = files.split("_")
                            self.data_list.append([day, [int(files_inf[2]), int(files_inf[3][:-4])], ["-", "-"], file_path])
                            break
                    start_date += timedelta(days=1)

                #最后一天计算

                file_dir = os.path.join(data_path, end_date.strftime("%Y"))
                day = end_date.strftime("%Y%m%d")
                for files in os.listdir(file_dir):
                    if files[7:15] == day:
                        file_path = os.path.join(file_dir, files)
                        files_inf = files.split("_")
                        start_real = int(files_inf[2])
                        end_real = int(files_inf[3][:-4])
                        if start_real > end_idx:
                            break
                        if end_real < end_idx :
                            self.data_list.append([day, [start_real, end_real], ["-", "-"], file_path])
                        elif end_real == end_idx:
                            self.data_list.append([day, [start_real, end_idx], ["-", "-"], file_path])
                        else:
                            real_idx = end_idx - start_real
                            self.data_list.append([day, [start_real, end_idx], ["-", real_idx], file_path])
                        break





if __name__ == "__main__":
    # data_seq = DataSeq("20250101-20251223", "/scratch/mingze/data", "range")
    data_seq = DataSeq("effective", "/shares/weather/Fusion_RPLTW", "point")
    print(data_seq.data_list)
    print(len(data_seq.data_list))

    with open("mid.json", "w") as f:
        json.dump(data_seq.data_list, f, indent=4)
