import os
import json
import torch
import argparse
import numpy as np
import sys
import argparse
from argparse import Namespace



def create_args():
    parser = argparse.ArgumentParser(description='Multi-STL: 多源数据融合降水临近预报模型')

    parser.add_argument('--ex_name', '-ex', type=str, default='TAU_wind', help='本次实验的名称')
    parser.add_argument('--work_dir', '-wd', type=str, default='work_dirs', help='工作目录')
    parser.add_argument('--point_dir', '-pd', type=str, default=None, help='指定可视化目录')
    parser.add_argument('--input', action='store_true', default=False, help='输入可视化')
    parser.add_argument('--contrast', action='store_true', default=False, help='对比可视化')
    parser.add_argument('--gif', action='store_true', default=False, help='样本gif')
    
    args = parser.parse_args()

    return args


class visualizer():
    def __init__(self, configs, vis_configs):
        self.configs = configs
        self.total_seq = self.configs.total_seq
        self.label_idx = self.configs.label_idx
        self.v_cfg = vis_configs
        self.obj_path = os.path.join(self.v_cfg.work_dir, self.v_cfg.ex_name)
        if self.v_cfg.point_dir is None:
            self.save_dir = os.path.join(self.obj_path, "vis")
        else:
            self.save_dir = self.v_cfg.point_dir

        self.is_gif = self.v_cfg.gif

        if self.configs.dataset == "SDweather":
            from dataset.SDweather import SD_Painter, SD_Dataloader

            self.original_data = SD_Dataloader(self.configs, mode='test')
            self.viser = SD_Painter(self.configs)

        elif self.configs.dataset == "weatherbench":

            from dataset.WeatherBench import vis_WB, WB_Dataloader

            self.original_data = WB_Dataloader(self.configs, mode='test')
            self.viser = vis_WB(self.configs, self.save_dir)
        else:
            raise NotImplementedError("不存在该数据集的可视化方法")
        self.original_data.standardizer.data_params()
        self.original_data.standardizer.metric_params()

        self.pred_idx = self.get_idx()
        
        self.input_data = None
        self.pred_data = None
        self.label_data = None

    def visualize(self):
        """执行可视化"""

        if self.v_cfg.input:
            self.get_input()
            self.viser.ploter(
                category=self.configs.in_category,
                name_inf=self.pred_idx,
                pred_data=None,
                label_data=self.input_data,
                out_gif=self.is_gif,
                mode="input"
            )
        else:
            if self.v_cfg.contrast:
                self.get_pred()
                self.get_label()
                self.viser.ploter(
                    category=self.configs.out_category,
                    name_inf=self.pred_idx,
                    pred_data=self.pred_data,
                    label_data=self.label_data,
                    out_gif=self.is_gif,
                    mode="pred"
                )
            else:
                self.get_pred()
                self.viser.ploter(
                    category=self.configs.out_category,
                    name_inf=self.pred_idx,
                    pred_data=self.pred_data,
                    label_data=None,
                    out_gif=self.is_gif,
                    mode="pred"
                )


    def get_idx(self):

        with open(os.path.join(self.obj_path, 
                               "outputs", 
                               'out_label.json'
                               ), 'r') as f:
            pred_idx = json.load(f)
        
        return pred_idx

    def get_input(self):
        input_data = []
        for idx in self.pred_idx: # [start, caption]
            data_seq = np.array(self.original_data.data[
                idx[0]:(idx[0] + self.total_seq[0]),:,:,:])
            input_data.append(data_seq)
        dataset = torch.from_numpy(np.stack(input_data, axis=0))  # (samp, T_in, cate, H, W)
        self.input_data = dataset.numpy()
    
    def get_pred(self):
        self.pred_data = np.load(os.path.join(self.obj_path,"outputs",'out_data.npy'))
        
    def get_label(self):
        label_data = []
        for idx in self.pred_idx: # [start, caption]
            data_seq = np.array(self.original_data.data[
                (idx[0] + self.total_seq[0]):(idx[0] + sum(self.total_seq)),
                self.label_idx[0]:self.label_idx[1],:,:])
            label_data.append(data_seq)
        dataset = torch.from_numpy(np.stack(label_data, axis=0))  # (samp, T_out, cate, H, W)
        self.label_data = dataset.numpy()  # (samp, T_out, cate, H, W)

if __name__ == "__main__":

    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)

    vis_configs = create_args()

    obj_path = os.path.join(vis_configs.work_dir, vis_configs.ex_name, "obj_config.json")
    if os.path.exists(obj_path):
        with open(obj_path, "r", encoding="utf-8") as f:
            obj_configs = json.load(f)
            obj_configs = Namespace(**obj_configs)
    else:
        raise ValueError(f"工程{vis_configs.ex_name}不存在")

    vis = visualizer(obj_configs, vis_configs)
    vis.visualize()
