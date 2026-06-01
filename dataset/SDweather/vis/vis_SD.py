import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import font_manager
from matplotlib import ticker
from cartopy.io.shapereader import Reader
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from datetime import datetime, timedelta
from PIL import Image



class SD_Painter():

    def __init__(self, configs):
        self.configs = configs
        self.category = self.configs.out_category
        self.input_seq_len = self.configs.total_seq[0] if hasattr(self.configs, "total_seq") else 0
        vis_dir = os.path.dirname(os.path.abspath(__file__))
        font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
        self.title_font = font_manager.FontProperties(fname=font_path) if os.path.exists(font_path) else None
        province_path = os.path.join(vis_dir, '山东省_省', '山东省_省.shp')
        city_path = os.path.join(vis_dir, '山东省_市', '山东省_市.shp')
        self.save_path = os.path.join(os.path.abspath(self.configs.obj_dir), "vis")
        self.total_seq = self.configs.total_seq
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if hasattr(self.configs, "region"):
            self.lat_min = float(self.configs.region["lat"].split('-')[0])
            self.lat_max = float(self.configs.region["lat"].split('-')[1])
            self.lon_min = float(self.configs.region["lon"].split('-')[0])
            self.lon_max = float(self.configs.region["lon"].split('-')[1])
        else:
            self.lat_min = 34.0
            self.lat_max = 39.0
            self.lon_min = 114.0
            self.lon_max = 123.0


        self.province = cfeature.ShapelyFeature(
            Reader(province_path).geometries(),
            ccrs.PlateCarree(), 
            edgecolor='k',
            facecolor='none'
            )
        
        self.city = cfeature.ShapelyFeature(
            Reader(city_path).geometries(),
            ccrs.PlateCarree(), 
            edgecolor='k',
            facecolor='none'
            )


    def ploter(self, category, name_inf, pred_data = None, label_data=None, out_gif=False, mode="pred"):
        contrast_mode = mode != "input" and pred_data is not None and label_data is not None
        file_prefix = "input" if mode == "input" else "pred"
        title_prefix = "Input" if mode == "input" else "Pred"
        plot_data = label_data if mode == "input" else pred_data
        if plot_data is None and label_data is not None:
            plot_data = label_data
        if plot_data is None:
            raise ValueError(f"No plot data provided for mode={mode}")

        sample = plot_data.shape[0]
        T = plot_data.shape[1]

        for cate in range(len(category)):
            category_name = category[cate]
            save_dir = self.save_path

            for samp in range(sample):

                date = name_inf[samp]
                input_start_time = self.cal_date(date)
                plot_start_time = self.get_plot_start_time(input_start_time, mode)

                samp_dir = os.path.join(save_dir, f"{input_start_time.strftime('%Y%m%d-%H%M')}")
                if not os.path.exists(samp_dir):
                    os.makedirs(samp_dir)
                print(f"正在生成{category_name}样本{input_start_time.strftime('%Y%m%d-%H%M')}。")

                for t in range(T):
                    time = plot_start_time

                    if category_name =="CW" :
                        compare_data = None
                        if contrast_mode:
                            if label_data is None:
                                raise ValueError("Contrast mode requires label_data")
                            compare_data = label_data[samp, t, cate, :, :]
                        self.plot_wind(
                            plot_data[samp, t, cate, :, :],
                            t,
                            samp_dir,
                            time,
                            compare_data=compare_data,
                            file_prefix=file_prefix,
                            title_prefix=title_prefix,
                        )
                    else:
                        raise NotImplementedError("Not Implemented")
                          
                    plot_start_time += timedelta(minutes=10)

                if out_gif: 
                    self.plot_gif(samp_dir, os.path.basename(samp_dir))

    def cal_date(self, date):
        if isinstance(date, (list, tuple)):
            if len(date) < 2:
                raise ValueError(f"Unsupported SDweather sample identifier: {date}")
            date = date[1]

        if isinstance(date, str) and "_" in date:
            return datetime.strptime(date, "%Y%m%d_%H%M")

        start_day = datetime.strptime(f"{date.split('-')[0]}0000", "%Y%m%d%H%M")
        time_delta = timedelta(minutes=int(date.split("-")[1]) * 10)
        start_time = start_day + time_delta

        return start_time

    def get_plot_start_time(self, input_start_time, mode):
        if mode == "input":
            return input_start_time
        return input_start_time + timedelta(minutes=self.input_seq_len * 10)

    def build_wind_style(self):
        mycolors_cr = ('#b4dfff', '#7fc9ff', '#2ea7ec', '#00baab', '#92bc00', '#dfbd01',
                       '#fe7d0b', '#f84115', '#dd032c', '#9f0201', '#630e00')
        cmap = colors.ListedColormap(mycolors_cr)
        levels = [1.6, 3.4, 5.5, 8, 10.9, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]
        norm = colors.BoundaryNorm(levels, len(mycolors_cr))
        return cmap, norm, levels

    def format_cn_time(self, time):
        if not isinstance(time, datetime):
            raise ValueError(f"Unsupported time value: {time}")
        return time.strftime("%Y-%m-%d %H:%M")

    def build_axis_ticks(self, start, end):
        step = (end - start) / 10
        return np.arange(start, end + step * 0.5, step)

    def setup_geo_axis(self, ax, show_left_labels=True):
        try:
            ax.add_feature(self.province, lw=0.5)
            ax.add_feature(self.city, lw=0.8)
        except Exception as e:
            print(f"⚠️ 无法绘制 shapefile 边界: {e}")

        ax.set_extent([self.lon_min, self.lon_max, self.lat_min, self.lat_max], crs=ccrs.PlateCarree())

        lon_labels = self.build_axis_ticks(self.lon_min, self.lon_max)
        lat_labels = self.build_axis_ticks(self.lat_min, self.lat_max)
        ax.set_xticks(lon_labels, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_labels, crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.tick_params(labelsize=10)

        gridliner = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=False,
            linewidth=0.6,
            color='gray',
            alpha=0.6,
            linestyle='--'
        )
        gridliner.xlocator = ticker.FixedLocator(lon_labels)
        gridliner.ylocator = ticker.FixedLocator(lat_labels)

    def draw_panel_title(self, ax, title_prefix, time):
        if title_prefix == 'True':
            cur_color = 'red'
        elif title_prefix == 'Pred':
            cur_color = 'blue'
        elif title_prefix == 'Input':
            cur_color = 'green'
        else:
            raise ValueError(f"Unsupported title prefix: {title_prefix}")
        ax.text(
            0.49,
            1.06,
            title_prefix,
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            color=cur_color,
            fontsize=22,
            fontweight='bold'
        )
        ax.text(
            0.51,
            1.06,
            self.format_cn_time(time),
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            color='black',
            fontsize=20,
            fontproperties=self.title_font
        )

    def draw_wind_panel(self, ax, wind_data, title_prefix, title_color, time, cmap, norm, show_left_labels=True):
        if wind_data.ndim != 2:
            raise ValueError("data must be 2D for plotting")

        self.setup_geo_axis(ax, show_left_labels=show_left_labels)
        image = ax.imshow(
            wind_data,
            cmap=cmap,
            norm=norm,
            aspect='auto',
            extent=[self.lon_min, self.lon_max, self.lat_min, self.lat_max],
            origin='lower'
        )
        self.draw_panel_title(ax, title_prefix, time)
        return image

    def plot_wind(self, input_data, t, save_path, time, compare_data=None, file_prefix="pred", title_prefix="Pred"):

        save_path = os.path.join(save_path, f"{file_prefix}_{t+1}.png")

        projection = ccrs.PlateCarree()
        cmap, norm, levels = self.build_wind_style()
        data_aspect = input_data.shape[1] / input_data.shape[0]
        panel_height = 5.6
        panel_width = panel_height * data_aspect

        if compare_data is not None:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(panel_width * 2 + 2.2, panel_height + 1.4),
                subplot_kw={'projection': projection},
                constrained_layout=True
            )
            self.draw_wind_panel(axes[0], compare_data, 'True', 'blue', time, cmap, norm, show_left_labels=True)
            img = self.draw_wind_panel(axes[1], input_data, 'Pred', 'blue', time, cmap, norm, show_left_labels=False)
            colorbar = fig.colorbar(img, ax=axes, location='right', fraction=0.04, pad=0.03, ticks=levels)
        else:
            fig, ax = plt.subplots(
                figsize=(panel_width + 1.6, panel_height + 1.2),
                subplot_kw={'projection': projection},
                constrained_layout=True
            )
            img = self.draw_wind_panel(ax, input_data, title_prefix, 'blue', time, cmap, norm, show_left_labels=True)
            colorbar = fig.colorbar(img, ax=ax, location='right', fraction=0.04, pad=0.03, ticks=levels)

        colorbar.set_label('CW Value (m/s)', fontproperties=self.title_font, fontsize=11)
        for tick_label in colorbar.ax.get_yticklabels():
            tick_label.set_fontproperties(self.title_font)

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def plot_gif(self, dir_path, label):
        """生成 GIF 动图"""
        image_path = []

        for i in range(1, self.total_seq[1]+1):
            image_path.append(os.path.join(dir_path, f"pred_{i}.png"))

        images = [Image.open(x) for x in image_path]

        images[0].save(os.path.join(os.path.dirname(dir_path), f"{label}_browse.gif"), save_all=True, append_images=images[1:], duration=500, loop=0)
