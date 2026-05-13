import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap




class vis_WB():
    def __init__(self, configs, save_path):

        self.configs = configs
        self.label_idx = self.configs.label_idx
        self.total_seq = self.configs.total_seq
        self.out_size = tuple((512, 256))
        self.interp = cv2.INTER_CUBIC
        self.overlay_weight = 0.9
        self.map_linewidth_coast = 2
        self.map_linewidth_country = 1

        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.h_gap_1, self.h_gap_2, self.w_gap = self.build_gaps()
        self.colors = None
        self.base_f = self.map_plot()    
        
    def color_config(self, category):
        """配置颜色映射"""
        colors = {}
        for cate in category:
            colors[cate] = {}

            if cate == "w10" or cate == "u10" or cate == "v10":
                
                levels=(1.6, 3.4, 5.5, 8, 10.9, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7)
                colors[cate]["levels"] = np.asarray(levels, dtype=np.float32)
                colors[cate]["N"] = len(levels)

                mycolors_cr=('#b4dfff', '#7fc9ff', '#2ea7ec', '#00baab', '#92bc00', '#dfbd01',
                     '#fe7d0b', '#f84115', '#dd032c', '#9f0201', '#630e00')
                cols_rgb = np.stack([self.hex_to_rgb01(c) for c in mycolors_cr], axis=0)  # (N,3) RGB
                colors[cate]["color"] = cols_rgb[:, ::-1].astype(np.float32)  # (N,3) BGR

            elif cate == "tcc":  # 添加TCC的颜色配置

                levels = (0.0, 0.3, 0.6, 0.8, 0.84, 0.88, 0.92, 0.96, 1.0) # 0到1之间的等间距10个等级
                colors[cate]["levels"] = np.asarray(levels, dtype=np.float32)
                colors[cate]["N"] = len(levels)

                mycolors_cr =  (
                    '#ffffff', # 0.00 纯白 (晴空)
                    '#e5f5f9', # 0.15 极浅青
                    '#bae4bc', # 0.30 嫩绿 (保留一点原有的绿色调，显得色彩丰富)
                    '#7bccc4', # 0.45 青绿
                    '#43a2ca', # 0.60 湖蓝 (替换了原本的橘黄色，进入均值区)
                    '#0868ac', # 0.70 深湖蓝
                    '#084081', # 0.80 深蓝 (高值区开始)
                    '#053061', # 0.90 藏青
                    '#021a36'  # 1.00 墨蓝 (完全覆盖)
                    )
                cols_rgb = np.stack([self.hex_to_rgb01(c) for c in mycolors_cr], axis=0)  # (N,3) RGB
                colors[cate]["color"] = cols_rgb[:, ::-1].astype(np.float32)  # (N,3) BGR
            
            # elif cate == "tp" or cate == "total_precipitation" or cate == "precip":
            #     # 单位是 m，不是 mm
            #     # 下面这些阈值分别对应：
            #     # 0, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50 mm
            #     # 换成 m 后就是：
            #     levels = (
            #         0.0,
            #         0.0001,
            #         0.0005,
            #         0.001,
            #         0.002,
            #         0.005,
            #         0.01,
            #         0.02,
            #         0.03,
            #         0.05
            #     )
            #     colors[cate]["levels"] = np.asarray(levels, dtype=np.float32)
            #     colors[cate]["N"] = len(levels)

            #     # 降水建议用“浅色到深蓝/紫”的风格，比较符合气象习惯
            #     mycolors_cr = (
            #         '#ffffff',  # 0
            #         '#d9f0ff',  # 0.1 mm
            #         '#a6d8ff',  # 0.5 mm
            #         '#6fb6ff',  # 1 mm
            #         '#3c8df0',  # 2 mm
            #         '#1f78b4',  # 5 mm
            #         '#225ea8',  # 10 mm
            #         '#253494',  # 20 mm
            #         '#4b0082',  # 30 mm
            #         '#800026'   # 50 mm+
            #     )
            #     cols_rgb = np.stack([self.hex_to_rgb01(c) for c in mycolors_cr], axis=0)
            #     colors[cate]["color"] = cols_rgb[:, ::-1].astype(np.float32)
            elif cate == "tp" or cate == "precip" or cate == "total_precipitation":
                # 单位: m
                # 0, 0.1, 0.5, 1, 2, 5, 10, 20, 30, 50 mm
                levels = (
                    0.0,
                    0.00003,
                    0.00006,
                    0.00010,
                    0.00015,
                    0.00025,
                    0.00040,
                    0.00060,
                    0.00085,
                    0.00120,
                    0.00170,
                    0.00250,
                    0.00350,
                    0.00500,
                    0.00750,
                    0.01000,
                    0.01500,
                    0.02200,
                    0.03200,
                    0.05000
                )
                colors[cate]["levels"] = np.asarray(levels, dtype=np.float32)
                colors[cate]["N"] = len(levels)

                # 借参考图的“雾面感”，但保持降水的顺序色轴
                mycolors_cr = (
                    '#ffffff',  # 0
                    '#edf4fb',  # 0.03 mm
                    '#deebf7',  # 0.06 mm
                    '#cfe1f2',  # 0.10 mm
                    '#bed6ec',  # 0.15 mm
                    '#a9c8e4',  # 0.25 mm
                    '#91b8da',  # 0.40 mm
                    '#78a6cf',  # 0.60 mm
                    '#6193c4',  # 0.85 mm
                    '#4d82bb',  # 1.20 mm
                    '#3e72b2',  # 1.70 mm
                    '#3363aa',  # 2.50 mm
                    '#2b569f',  # 3.50 mm
                    '#244a94',  # 5.00 mm
                    '#223f88',  # 7.50 mm
                    '#24357c',  # 10.0 mm
                    '#2a2d73',  # 15.0 mm
                    '#34306d',  # 22.0 mm
                    '#443266',  # 32.0 mm
                    '#5a305e'   # 50.0 mm+
                )
                cols_rgb = np.stack([self.hex_to_rgb01(c) for c in mycolors_cr], axis=0)
                colors[cate]["color"] = cols_rgb[:, ::-1].astype(np.float32)

            elif cate == "t2m":

                # 参考图片范围大致在 240K 到 310K+
                # 这里设置锚点，覆盖深蓝(冷) -> 浅蓝 -> 浅黄(舒适) -> 橙 -> 深红(热)
                levels = (240.0, 250.0, 260.0, 270.0, 273.15, 280.0, 290.0, 300.0, 310.0, 320.0)
                colors[cate]["levels"] = np.asarray(levels, dtype=np.float32)
                colors[cate]["N"] = len(levels)

                mycolors_cr = (
                    '#313695', # 240.0 深蓝
                    '#4575b4', # 250.0 蓝
                    '#74add1', # 260.0 浅蓝
                    '#abd9e9', # 270.0 极浅蓝
                    '#e0f3f8', # 273.15 (0度) 接近白/青
                    '#ffffbf', # 280.0 浅黄
                    '#fee090', # 290.0 浅橙
                    '#fdae61', # 300.0 橙色
                    '#f46d43', # 310.0 红橙
                    '#a50026'  # 320.0 深红
                )
                # 提示：你现有的插值代码会自动在这些颜色之间进行平滑过渡（Gradient），
                # 从而实现"连续平滑色轴"的效果，而不会像等高线图那样分层明显。
                cols_rgb = np.stack([self.hex_to_rgb01(c) for c in mycolors_cr], axis=0)
                colors[cate]["color"] = cols_rgb[:, ::-1].astype(np.float32)

            elif cate == "orography":
                # 统计数据: Max ~4910m.
                # 配置策略: 20个节点，从0m到5000m，涵盖平原到雪山
                # 负值处理: 你的数据最小值-29m会被自动归类为第0级的颜色(深绿)，符合陆地低洼地特征
                
                # 20个水位等级 (Levels)
                levels = (
                    0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 
                    1000.0, 1300.0, 1600.0, 1900.0, 2200.0, 2500.0, 2800.0, 3100.0, 3400.0, 3700.0, 4000.0     
                )
                
                colors[cate]["levels"] = np.asarray(levels, dtype=np.float32)
                colors[cate]["N"] = len(levels)

                # 20种颜色：从原本的 Hex 列表中精心挑选的渐变色
                mycolors_cr = (
                    '#dbf3fa', # 0-100    极浅蓝 (低洼湿地)
                    '#baddb8', # 100-200  灰绿
                    '#8bc98d', # 200-300  浅草绿
                    '#58b366', # 300-400  鲜草绿
                    '#2d9c44', # 400-500  深绿 (森林)
                    '#b8bd3d', # 500-600  黄绿 (灌木/过渡)
                    '#d9c55b', # 600-700  土黄
                    '#e3a64b', # 700-800  橙褐
                    '#bd7e3e', # 800-900  浅棕
                    '#8c5a32', # 900-1000 深棕 (高山基座)
                    "#328c5f", # 1000m 以上 
                    '#7b7c9e', # 1000-1300 灰紫 (岩石阴影/雪线之下)
                    '#6266a8', # 1300-1600 蓝紫
                    '#4b57b5', # 1600-1900 深蓝 (视觉深邃感)
                    '#4169e1', # 1900-2200 皇室蓝 (RoyalBlue)
                    '#5ca0eb', # 2200-2500 亮中蓝
                    '#77caff', # 2500-2800 天蓝
                    '#92e5ff', # 2800-3100 浅青
                    '#baffff', # 3100-3400 冰川青
                    '#e0ffff', # 3400-3700 近白 (霜)
                    '#ffffff'  # 3700-4000+ 纯白 (积雪/峰顶)
                )
                
                # 转换为 OpenCV 所需的 BGR 格式
                cols_rgb = np.stack([self.hex_to_rgb01(c) for c in mycolors_cr], axis=0) # (N,3) RGB
                colors[cate]["color"] = cols_rgb[:, ::-1].astype(np.float32) # (N,3) BGR

            else:
                raise NotImplementedError("Not Implemented")
        self.colors = colors

    def ploter(self, category, name_inf, pred_data = None, label_data=None, out_gif=False, mode="pred"):

        """绘制可视化样本"""
        if label_data is not None and pred_data is not None:
            vis_data = np.stack((label_data, pred_data), axis=3) # (samp, T, cate, 2, H, W)
        elif pred_data is not None and label_data is None:
            vis_data = np.expand_dims(pred_data, axis=3)
        elif pred_data is None and label_data is not None:
            vis_data = np.expand_dims(label_data, axis=3)
        else:
            raise ValueError("请至少提供预测数据或标签数据进行可视化")

        self.color_config(category)
        # vis_data = self.adjust_geo_view(vis_data)

        sample = vis_data.shape[0]
        T = vis_data.shape[1]

        for samp in range(sample):

            caption = name_inf[samp][1]

            samp_dir = os.path.join(self.save_path, caption)
            if not os.path.exists(samp_dir):
                os.makedirs(samp_dir)
            print(f"正在生成样本{caption}。")

            for t in range(T):
                self.splicer(vis_data[samp, t, :, :, :, :], category, samp_dir, label=f"{mode}_{t+1}")

            if out_gif:
                self.plot_gif(samp_dir, caption)


    def splicer(self, vis_data, category , save_path, label): #vis_data: (cate, 2, H, W)
        """将可视化数据拼接成图片"""
        section = []
        lenth = vis_data.shape[1] # 2
        for c, cate in enumerate(category):
            cate_section = []
            for i in range(lenth):
                if cate == "w10" :  
                    cate_section.append(self.plot_wind(
                        vis_data[c, i, :, :], 
                        self.colors[cate]["levels"], 
                        self.colors[cate]["color"], 
                        self.colors[cate]["N"]
                        )) # (H, W, 3)
                    
                elif cate == "tcc":  # 添加TCC的可视化方法
                    cate_section.append(self.plot_tcc(
                        vis_data[c, i, :, :], 
                        self.colors[cate]["levels"], 
                        self.colors[cate]["color"], 
                        self.colors[cate]["N"]
                        )) # (H, W, 3)
                    
                elif cate == "t2m":
                    cate_section.append(self.plot_t2m(
                        vis_data[c, i, :, :],
                        self.colors[cate]["levels"],
                        self.colors[cate]["color"],
                        self.colors[cate]["N"]
                    ))

                elif cate == "tp" or cate == "total_precipitation" or cate == "precip":
                    cate_section.append(self.plot_tp(
                        vis_data[c, i, :, :],
                        self.colors[cate]["levels"],
                        self.colors[cate]["color"],
                        self.colors[cate]["N"]
                    ))

                elif cate == "orography":
                    cate_section.append(self.plot_orography(
                        vis_data[c, i, :, :],
                        self.colors[cate]["levels"],
                        self.colors[cate]["color"],
                        self.colors[cate]["N"]
                    ))
                else:
                    raise NotImplementedError(f"不存在{cate}可视化方法")
            if len(cate_section) == 1:
                section.append(cate_section[0])
            else:
                section.append(np.concatenate([cate_section[0], self.w_gap, cate_section[1]], axis=1)) # (H, W*2 + gapw, 3)
        
        if len(section) == 1:
            output = section[0]
        else:
            output = section[0] 
            if lenth == 2:
                for i in range(1, len(section)):
                    output = np.concatenate([output, self.h_gap_2, section[i]], axis=0) # (H*2 + gaph, W, 3)
            elif lenth == 1:
                for i in range(1, len(section)):
                    output = np.concatenate([output, self.h_gap_1, section[i]], axis=0) # (H + gaph, W, 3)

        cv2.imwrite(os.path.join(save_path, f"{label}.png"), output)

    def plot_wind(self, heatmap_hw, levels, cols_bgr, N):
        """绘制风场可视化"""

        # ---------- 1) 输入检查 ----------
        heatmap = np.asarray(heatmap_hw, dtype=np.float32)
        if heatmap.ndim != 2:
            raise ValueError(f"heatmap 必须是二维[H,W]，收到 shape={heatmap.shape}")

        # ---------- 2) 标量场插值到目标大小 ----------
        hm = cv2.resize(heatmap, self.out_size, interpolation=self.interp)

        # ---------- 3) levels + mycolors 分段线性插值上色 ----------
        v = hm
        out = np.empty((hm.shape[0], hm.shape[1], 3), dtype=np.float32)

        nan_mask = ~np.isfinite(v)

        # 左端/右端饱和
        out[v <= levels[0]] = cols_bgr[0]
        out[v >= levels[-1]] = cols_bgr[-1]

        # 中间区间
        mid = (v > levels[0]) & (v < levels[-1]) & (~nan_mask)
        if np.any(mid):
            vv = v[mid]
            i = np.searchsorted(levels, vv, side="right") - 1
            i = np.clip(i, 0, N - 2)
            t = (vv - levels[i]) / (levels[i + 1] - levels[i] + 1e-12)
            c0 = cols_bgr[i]
            c1 = cols_bgr[i + 1]
            out[mid] = (1.0 - t)[:, None] * c0 + t[:, None] * c1

        # NaN 用底图替代
        if np.any(nan_mask):
            out[nan_mask] = self.base_f[nan_mask]

        hm_f = out  # float[0,1]

        # ---------- 4) 融合 + 保留底图线条 ----------
        cam = self.overlay_weight * hm_f + (1.0 - self.overlay_weight) * self.base_f
        keep_bg = (255.0 * self.base_f[:, :, 0]) < 100.0
        cam[keep_bg, :] = self.base_f[keep_bg, :]

        cam = cam / (np.max(cam) + 1e-8)
        out_bgr = np.uint8(255 * cam)

        return out_bgr
    
    def plot_tcc(self, heatmap_hw, levels, cols_bgr, N):
        """绘制TCC（Total Cloud Cover）可视化"""
        
        # ---------- 1) 输入检查 ----------
        heatmap = np.asarray(heatmap_hw, dtype=np.float32)
        if heatmap.ndim != 2:
            raise ValueError(f"heatmap 必须是二维[H,W]，收到 shape={heatmap.shape}")

        # ---------- 2) 标量场插值到目标大小 ----------
        hm = cv2.resize(heatmap, self.out_size, interpolation=self.interp)

        # ---------- 3) levels + mycolors 分段线性插值上色 ----------
        v = hm
        out = np.empty((hm.shape[0], hm.shape[1], 3), dtype=np.float32)

        nan_mask = ~np.isfinite(v)

        # 左端/右端饱和
        out[v <= levels[0]] = cols_bgr[0]
        out[v >= levels[-1]] = cols_bgr[-1]

        # 中间区间
        mid = (v > levels[0]) & (v < levels[-1]) & (~nan_mask)
        if np.any(mid):
            vv = v[mid]
            i = np.searchsorted(levels, vv, side="right") - 1
            i = np.clip(i, 0, N - 2)
            t = (vv - levels[i]) / (levels[i + 1] - levels[i] + 1e-12)
            c0 = cols_bgr[i]
            c1 = cols_bgr[i + 1]
            out[mid] = (1.0 - t)[:, None] * c0 + t[:, None] * c1

        # NaN 用底图替代
        if np.any(nan_mask):
            out[nan_mask] = self.base_f[nan_mask]

        hm_f = out  # float[0,1]

        # ---------- 4) 融合 + 保留底图线条 ----------
        cam = self.overlay_weight * hm_f + (1.0 - self.overlay_weight) * self.base_f
        keep_bg = (255.0 * self.base_f[:, :, 0]) < 100.0
        cam[keep_bg, :] = self.base_f[keep_bg, :]

        cam = cam / (np.max(cam) + 1e-8)
        out_bgr = np.uint8(255 * cam)

        return out_bgr
    
    def plot_tp(self, heatmap_hw, levels, cols_bgr, N):
        """
        绘制降水 (Total Precipitation) 可视化
        注意：输入单位为 m
        """
        heatmap = np.asarray(heatmap_hw, dtype=np.float32)
        if heatmap.ndim != 2:
            raise ValueError(f"heatmap 必须是二维[H,W]，收到 shape={heatmap.shape}")

        hm = cv2.resize(heatmap, self.out_size, interpolation=self.interp)

        v = hm
        out = np.empty((hm.shape[0], hm.shape[1], 3), dtype=np.float32)
        nan_mask = ~np.isfinite(v)

        out[v <= levels[0]] = cols_bgr[0]
        out[v >= levels[-1]] = cols_bgr[-1]

        mid = (v > levels[0]) & (v < levels[-1]) & (~nan_mask)
        if np.any(mid):
            vv = v[mid]
            i = np.searchsorted(levels, vv, side="right") - 1
            i = np.clip(i, 0, N - 2)
            t = (vv - levels[i]) / (levels[i + 1] - levels[i] + 1e-12)
            c0 = cols_bgr[i]
            c1 = cols_bgr[i + 1]
            out[mid] = (1.0 - t)[:, None] * c0 + t[:, None] * c1

        if np.any(nan_mask):
            out[nan_mask] = self.base_f[nan_mask]

        hm_f = out

        cam = self.overlay_weight * hm_f + (1.0 - self.overlay_weight) * self.base_f
        keep_bg = (255.0 * self.base_f[:, :, 0]) < 100.0
        cam[keep_bg, :] = self.base_f[keep_bg, :]

        cam = cam / (np.max(cam) + 1e-8)
        out_bgr = np.uint8(255 * cam)

        return out_bgr

    def plot_t2m(self, heatmap_hw, levels, cols_bgr, N):
        """
        绘制 t2m (2m Temperature) 可视化
        使用线性插值实现连续平滑色轴效果
        """
        # ---------- 1) 输入检查 ----------
        heatmap = np.asarray(heatmap_hw, dtype=np.float32)
        if heatmap.ndim != 2:
            raise ValueError(f"heatmap 必须是二维[H,W]，收到 shape={heatmap.shape}")

        # ---------- 2) 标量场插值到目标大小 ----------
        hm = cv2.resize(heatmap, self.out_size, interpolation=self.interp)

        # ---------- 3) levels + mycolors 分段线性插值上色 (Continuous Interpolation) ----------
        v = hm
        out = np.empty((hm.shape[0], hm.shape[1], 3), dtype=np.float32)
        
        nan_mask = ~np.isfinite(v)

        # 左端/右端饱和处理
        out[v <= levels[0]] = cols_bgr[0]
        out[v >= levels[-1]] = cols_bgr[-1]

        # 中间区间插值
        # 这里的逻辑保证了颜色是平滑过渡的，而不是阶梯状的
        mid = (v > levels[0]) & (v < levels[-1]) & (~nan_mask)
        if np.any(mid):
            vv = v[mid]
            # 找到数值所在的区间索引
            i = np.searchsorted(levels, vv, side="right") - 1
            i = np.clip(i, 0, N - 2)
            
            # 计算在该区间内的比例 t (0.0 到 1.0)
            t = (vv - levels[i]) / (levels[i + 1] - levels[i] + 1e-12)
            
            c0 = cols_bgr[i]
            c1 = cols_bgr[i + 1]
            
            # 混合颜色：根据 t 值在 c0 和 c1 之间进行线性混合
            out[mid] = (1.0 - t)[:, None] * c0 + t[:, None] * c1

        # NaN 用底图替代
        if np.any(nan_mask):
            out[nan_mask] = self.base_f[nan_mask]

        hm_f = out  # float[0,1]

        # ---------- 4) 融合 + 保留底图线条 ----------
        cam = self.overlay_weight * hm_f + (1.0 - self.overlay_weight) * self.base_f
        keep_bg = (255.0 * self.base_f[:, :, 0]) < 100.0
        cam[keep_bg, :] = self.base_f[keep_bg, :]

        cam = cam / (np.max(cam) + 1e-8)
        out_bgr = np.uint8(255 * cam)

        return out_bgr
    
    def plot_orography(self, heatmap_hw, levels, cols_bgr, N):
        """
        绘制地形 (Topography) 可视化 - 专业版
        """
        # ---------- 1) 输入检查 ----------
        heatmap = np.asarray(heatmap_hw, dtype=np.float32)
        if heatmap.ndim != 2:
            raise ValueError(f"heatmap 必须是二维[H,W]，收到 shape={heatmap.shape}")

        # ---------- 2) 插值到目标大小 ----------
        hm = cv2.resize(heatmap, self.out_size, interpolation=self.interp)

        # ---------- 3) 连续平滑上色 (Continuous Interpolation) ----------
        v = hm
        out = np.empty((hm.shape[0], hm.shape[1], 3), dtype=np.float32)
        
        nan_mask = ~np.isfinite(v)

        # 边界处理：
        # 你的最小值 -28.9 会小于 levels[0] (0.0)，因此会被赋值为 cols_bgr[0] (深绿色)
        # 这在地理上是正确的，表示低于海平面的洼地（如吐鲁番盆地）显示为深色低地
        out[v <= levels[0]] = cols_bgr[0]
        out[v >= levels[-1]] = cols_bgr[-1]

        # 中间区间插值
        mid = (v > levels[0]) & (v < levels[-1]) & (~nan_mask)
        if np.any(mid):
            vv = v[mid]
            # 找到区间索引
            i = np.searchsorted(levels, vv, side="right") - 1
            i = np.clip(i, 0, N - 2)
            
            # 计算平滑系数 t
            t = (vv - levels[i]) / (levels[i + 1] - levels[i] + 1e-12)
            
            c0 = cols_bgr[i]
            c1 = cols_bgr[i + 1]
            
            # 线性混合颜色，实现20个颜色节点间的丝滑过渡
            out[mid] = (1.0 - t)[:, None] * c0 + t[:, None] * c1

        # NaN 处理
        if np.any(nan_mask):
            out[nan_mask] = self.base_f[nan_mask]

        hm_f = out 

        # ---------- 4) 融合 ----------
        # 由于颜色分级多达20级，细节很丰富，我们稍微增加底图线条的透视度
        topo_weight = 0.92 
        cam = topo_weight * hm_f + (1.0 - topo_weight) * self.base_f
        
        # 强制保留黑色海岸线，增加专业感
        keep_bg = (255.0 * self.base_f[:, :, 0]) < 80.0
        cam[keep_bg, :] = self.base_f[keep_bg, :]

        cam = cam / (np.max(cam) + 1e-8)
        out_bgr = np.uint8(255 * cam)

        return out_bgr
    
    def map_plot(self):
        """绘制地图底图"""

        fig = plt.figure(figsize=(8, 4))
        ax = fig.add_axes([0., 0., 1., 1.])
        ax.set_axis_off()

        m = Basemap() # 标准世界地图
        # m = Basemap(projection='cyl', 
        #         lon_0=180,              
        #         llcrnrlon=0, urcrnrlon=360,  
        #         llcrnrlat=-90, urcrnrlat=90,
        #         resolution='c') # 太平洋中心地图
        m.drawcoastlines(linewidth=2)
        m.drawcountries(linewidth=1)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        plt.close(fig)

        base = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)                        # BGR uint8
        base = cv2.resize(base, self.out_size, interpolation=cv2.INTER_AREA) # 目标尺寸

        return base.astype(np.float32) / 255.0                        # float[0,1]
    
    @staticmethod
    def hex_to_rgb01(hx: str) -> np.ndarray:
            hx = hx.lstrip('#')
            return np.array([int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)],
                            dtype=np.float32) / 255.0

    def build_gaps(self):
        """
        构建gap
        """
        gapw = 10
        gaph = 10
        W, H = self.out_size  # out_size = (512, 256)

        # 横向 gap：左右图之间
        w_gap = np.full((H, gapw, 3), 255, dtype=np.uint8)

        # 纵向 gap：上下图之间（基于“已横向拼接”后的宽度）
        h_gap_1 = np.full((gaph, W, 3), 255, dtype=np.uint8)
        total_w = W * 2 + gapw
        h_gap_2 = np.full((gaph, total_w, 3), 255, dtype=np.uint8)

        return h_gap_1, h_gap_2, w_gap
    
    @staticmethod
    def adjust_geo_view(data):
        """
        调整气象数据的地理视角。
        """
        # 1. 获取宽度 W (最后一个维度)
        W = data.shape[-1]
        
        # 2. 上下翻转 (针对 H 维度/倒数第2维)
        # 将 lat: -90...90 (南极在第一行) 翻转为 lat: 90...-90 (北极在第一行)
        data = np.flip(data, axis=-2)
        
        # 3. 经度平移 (针对 W 维度/倒数第1维)
        # 原始: [0, ..., 180, ..., 360] (0度在左边)
        # 目标: [-180, ..., 0, ..., 180] (0度在中间)
        # 操作: 向右滚动一半宽度
        shift = W // 2
        data = np.roll(data, shift, axis=-1)
        
        return data

    def plot_gif(self, dir_path, label):
        """生成 GIF 动图"""
        image_path = []

        for i in range(1, self.total_seq[1]+1):
            image_path.append(os.path.join(dir_path, f"pred_{i}.png"))

        images = [Image.open(x) for x in image_path]

        images[0].save(os.path.join(os.path.dirname(dir_path), f"{label}_browse.gif"), save_all=True, append_images=images[1:], duration=500, loop=0)
