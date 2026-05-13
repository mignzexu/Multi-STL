import datetime
import json
import os


class Logger:
    def __init__(self, configs, mode="train"):
        self.configs = configs
        self.filepath = os.path.join(self.configs.obj_dir, "log.txt")
        self.thr_num = max([len(i) for i in self.configs.threshold])
        self.metric_lenth = 3 if self.thr_num < 3 else self.thr_num

        self.out_category = self.configs.out_category
        
        # 定义占位符（用于后续查找位置进行插入）
        self.best_vl = "(★) Best valid loss  :  "
        self.best_ve = "(★) Best valid epoch :  "
        self.best_val = float('inf')
        
        self.train_header = ["Epoch", "Train loss", "Valid loss", "LR"]
        self.train_length = [7, 12, 12, 10]
        self.total_width = 100
        self.num_lines = None

        for cate in self.out_category:
            self.train_header.insert(-1, f"{cate.upper()} verification")
            self.train_length.insert(-1, 60)

        if mode == "train":

            with open(self.filepath, "w", encoding="utf-8") as f:
                self._plot_config(f)

            # 2. 写入 Test 标题和占位符 (留白)
            self._init_test_placeholder()
            # 3. 写入 Train 标题和表头
            self._init_train_table()

    def _plot_config(self, f):
        """写入顶部简约实验信息"""
        f.write("=" * self.total_width + "\n")
        f.write(f"Experiment: {getattr(self.configs, 'ex_name', 'Unnamed')}\n")
        f.write(f"Time      : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("-" * self.total_width + "\n")

        """核心：写入双栏配置信息"""
        # 分类 Key
        model_keys = ["load_model", "batch_size", "epoch", "learning_rate", 
                      "optimizer", "scheduler", "loss_function"]
        data_keys = ["dataset", "train_range", "test_range", "in_category", 
                     "out_category", "total_seq", "img_size"]

        max_len = max(len(model_keys), len(data_keys))

        f.write(f"{'-Model Configuration':<50}{'-Data Configuration':<50}\n\n") # 中间加个|更好看

        for i in range(max_len):
            # --- 左边：Model Keys 处理 ---
            if i < len(model_keys):
                mk = model_keys[i]
                mv = str(getattr(self.configs, mk, "N/A"))
                # 正常生成左侧内容
                model_label = f"{mk:<15} : {mv:<30}"
            else:
                # 如果 model_keys 用完了，用空格填充，保持对齐
                # 长度 = 15(key) + 3( : ) + 30(value) = 48，或者直接给个固定宽度
                model_label = f"{'':<48}" 

            # --- 右边：Data Keys 处理 ---
            if i < len(data_keys):
                dk = data_keys[i]
                dv = str(getattr(self.configs, dk, "N/A"))
                # 正常生成右侧内容
                data_label = f"{dk:<15} : {dv:<30}"
            else:
                # 如果 data_keys 用完了，右边可以是空的
                data_label = ""

            # 写入文件
            f.write(f"{model_label} | {data_label}\n")

            
        f.write("=" * self.total_width + "\n\n")


    def _init_test_placeholder(self):
        """
        写入测试标题，并留下一个占位符字符串
        """
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(">>> Test Assessment <<<\n\n")
            # 这里的占位符必须独占一行，方便替换
            test_div = "○"
            test_sub_div = "○"
            test_label = "|"
            test_content = "|" 
            for cate in self.out_category:
                capt = f"{cate.upper()} Test"
                test_label += f"{capt:^72}|"
                test_content += " " * 72 + "|"
                test_div += "="*72 + "○"
                test_sub_div += "-"*72 + "○"
            f.write(test_div + "\n")
            f.write(test_label + "\n")
            f.write(test_sub_div + "\n")
            f.write(test_content + "\n")
            f.write(test_div + "\n\n")

    def _init_train_table(self):
        """
        初始化训练过程表格 (三列结构)
        """
        # 表头
        header_str = "|"
        for i, header in enumerate(self.train_header):
            header_str += f"{header:^{self.train_length[i]}}|"  # 居中对齐

        # 分割线
        self.train_sep = "□"
        for i, header in enumerate(self.train_header):
            self.train_sep += f"{'-'*self.train_length[i]}□"

        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(">>> Training Progress <<<\n\n")
            f.write(f"{self.best_vl}" + "\n")
            f.write(f"{self.best_ve}" + "\n")
            f.write(self.train_sep + "\n")
            f.write(header_str + "\n")
            f.write(self.train_sep + "\n")


    def log_train(self, epoch, lr, train_metrics, valid_metrics):

        # --- 步骤 A: 准备 Metrics 列的内容 (多行文本列表) ---
        metric_lines = []
        
        # 1. 处理标量指标 (MAE, MSE, RMSE) -> 放第一行
        for ca, cate in enumerate(self.out_category):
            scalars = []
            events = []
            metric_lines.append([])
            for k, v in valid_metrics[str(epoch)][cate].items():
                if not isinstance(v, list) :
                    val_str = f"{v:.2e}" if isinstance(v, float) else str(v)
                    scalars.append(f"{k.upper():<6}: {val_str:<8}")
                    if len(scalars) == self.metric_lenth:
                        metric_lines[ca].append("   ".join(scalars))
                        scalars.clear()
                    
                else:
                    for idx, val in enumerate(v):
                        val_str = f"{val:.6f}" if isinstance(val, float) else str(val)
                        label = f"{k.upper()}-{idx+1}"
                        events.append(f"{label:<6}: {val_str:<8}")
                    
                    metric_lines[ca].append("   ".join(events))
                    events.clear()

        # --- 步骤 B: 组合打印 ---
        # 即使 metrics 有 5 行，Epoch 和 LR 只显示在第 1 行，后面为空
        if self.num_lines is None:
            self.num_lines = max([len(metric_lines[ca]) for ca in range(len(metric_lines))])


        with open(self.filepath, "a", encoding="utf-8") as f:

            label =f" ◎ Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f.write(f"|{label:<104}|\n")
            f.write(self.train_sep + "\n")

            for i in range(self.num_lines):
                # 只有第一行显示 Epoch 和 LR
                if i == self.num_lines // 2:
                    ep_str = str(epoch)
                    write_list = [ep_str, 
                                  f"{train_metrics[ep_str]['loss']:.6f}", 
                                  f"{valid_metrics[ep_str]['loss']:.6f}",
                                  f"{lr:.2e}"
                                  ]
                else:
                    write_list = ["", "", "", ""]
                
                # 获取当前行的 metric 内容
                for g in range(len(self.out_category)):
                    m_str = metric_lines[g][i] if i < len(metric_lines[g]) else ""
                    write_list.insert(-1, m_str)
                
                write_str = "|"
                for o, item in enumerate(write_list):
                    write_str += f"{item:^{self.train_length[o]}}|"
                f.write(write_str + "\n")
            
            # 每一条记录结束后，画一条分割线 (这是你要求的)
            f.write(self.train_sep + "\n")
        
    
    def best_log(self, epoch, value):

        
        with open(self.filepath, "r", encoding="utf-8") as f:
            content = f.read()
            
            # 3. 执行替换操作
            vl = f"(★) Best valid loss  :  {value:.6f}"
            ve  = f"(★) Best valid epoch :  {epoch}"
            new_content = content.replace(self.best_vl, vl)
            new_content = new_content.replace(self.best_ve, ve)
                
        # 4. 覆盖写入
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        self.best_vl = vl
        self.best_ve = ve

    def best_log1(self, epoch, value):
        with open(self.filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()

        position = next(l for l, line in enumerate(lines) if line[:3] == "(★)")
        lines[position] = f"(★) Best valid loss  :  {value:.6f}\n"
        lines[position+1] = f"(★) Best valid epoch :  {epoch}\n"

        with open(self.filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)


    def test_record(self, test_metrics):
        """
        读取整个文件，找到占位符，将其替换为真正的表格
        """
        metric_lines = []
        for ca, cate in enumerate(self.out_category):
            scalars = []
            events = []
            metric_lines.append([])
            for k, v in test_metrics["0"][cate].items():
                if not isinstance(v, list) :
                    val_str = f"{v:.5e}" if isinstance(v, float) else str(v)
                    scalars.append(f"{k.upper():<6}: {val_str:<12}")
                    if len(scalars) == self.metric_lenth:
                        metric_lines[ca].append("   ".join(scalars))
                        scalars.clear()
                    
                else:
                    for idx, val in enumerate(v):
                        val_str = f"{val:.6f}" if isinstance(val, float) else str(val)
                        label = f"{k.upper()}-{idx+1}"
                        events.append(f"{label:<6}: {val_str:<12}")
                    
                    metric_lines[ca].append("   ".join(events))
                    events.clear()
        
        show_line = []
        cate_len = len(self.out_category)
        line_len = max([len(line) for line in metric_lines])
        for l in range(line_len):
            line = "|"
            for c in range(cate_len):
                if l < len(metric_lines[c]):
                    line += f"{metric_lines[c][l]:^72}|"
                else:
                    line += " " * 72 + "|"
            show_line.append(line+"\n")
        add = ""
        for line in show_line:
            add += line


        # 2. 读取当前文件所有内容
        with open(self.filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        seq = [l for l, line in enumerate(lines) if line[:3] == "○=="]
        seq[0] += 3

        del lines[seq[0]:seq[1]]

        lines.insert(seq[0], add)

        with open(self.filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)



if __name__ == "__main__":
    

    with open("work_dirs/gsta_tp_1/obj_config.json", "r") as f:
        config = json.load(f)
    # 2. 初始化 (此时 log 文件里 Test Metrics 下面是空的)
    logger = Logger(config)

    print("Logger 初始化完成，请查看 txt 文件，此时 Test 区域为空。")

    with open("work_dirs/gsta_tp_1/process/train_metrics.json", "r") as f:
        train_metrics = json.load(f)

    with open("work_dirs/gsta_tp_1/process/valid_metrics.json", "r") as f:
        valid_metrics = json.load(f)

    val_loss = float("inf")
    # 3. 模拟训练过程
    for epoch in range(1, 4):
        
        # 记录一行 (会自动分行显示列表)
        logger.log_train(epoch, 0.005, train_metrics, valid_metrics)
        valid_loss = valid_metrics[f"{epoch}"]["loss"]
        if valid_loss < val_loss:
            val_loss = valid_loss
            logger.best_log1(epoch, valid_loss)

    # 4. 假设训练到一半，或者训练结束后，你进行了测试
    #    得到了测试结果
    final_test_results = {"0":{"tp":{
        "mae": 0.5555,
        "mse": 0.1111,
        "rmse": 0.2222
    }}}

    # 5. 【关键】调用回填方法
    #    这会把文件头部的空白区域填上
    logger.test_record(final_test_results)
