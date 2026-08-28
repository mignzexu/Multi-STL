from utils.cfg_up import config_update



class Model_Instrument:
    def __init__(self, configs, mode):
        self.configs = configs
        self.mode = mode
        self.model_name = self.configs.load_model.lower()
        self.model = None
        self.loss_function = None
        self.load_model()

    def load_model(self):
        if self.model_name == 'gsta':
            from models.SimVP_gSTA import Model, gSTA_Config

            if self.mode == 'train' :
                model_config = gSTA_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == 'msradar':
            from models.MSRadar import Model, MS_RadarFormer_Config

            if self.mode == 'train' :
                model_config = MS_RadarFormer_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "stpanet":
            from models.STPANet import Model, STPANet_Config

            if self.mode == 'train' :
                model_config = STPANet_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "gvbf":
            from models.GVBF import Model, GVBF_Config

            if self.mode == 'train' :
                config_name = self.configs.model_config
                model_config = GVBF_Config(config_name).model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "sbflow":
            from models.SBFlow.config import SBFlow_Config

            if self.mode == 'train' :
                config_name = self.configs.model_config
                model_config = SBFlow_Config(config_name).model_config
                self.configs = config_update(self.configs, model_config)

            from models.SBFlow.Model import Model

            self.model = Model

        elif self.model_name == "drift":
            from models.Drift import Model, Drift_Config

            if self.mode == 'train' :
                config_name = self.configs.model_config
                model_config = Drift_Config(config_name).model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "convlstm":
            from models.ConvLSTM import Model, ConvLSTM_Config

            if self.mode == 'train' :
                model_config = ConvLSTM_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "e3dlstm":
            from models.E3DLSTM import Model, E3DLSTM_Config

            if self.mode == 'train' :
                model_config = E3DLSTM_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "mau":
            from models.MAU import Model, MAU_Config

            if self.mode == 'train' :
                model_config = MAU_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "mim":
            from models.MIM import Model, MIM_Config

            if self.mode == 'train' :
                model_config = MIM_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "mmvp":
            from models.MMVP import Model, MMVP_Config

            if self.mode == 'train' :
                model_config = MMVP_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "predrnnpp":
            from models.PerdRNNpp import Model, PredRNNpp_Config

            if self.mode == 'train' :
                model_config = PredRNNpp_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "phydnet":
            from models.PhyDNet import Model, PhyDNet_Config

            if self.mode == 'train' :
                model_config = PhyDNet_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "poolformer":
            from models.PoolFormer import Model, Poolformer_Config

            if self.mode == 'train' :
                model_config = Poolformer_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "prediff":
            from models.PreDiff import Model, PreDiff_Config

            if self.mode == 'train' :
                model_config = PreDiff_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "predrnn":
            from models.PredRNN import Model, PredRNN_Config

            if self.mode == 'train' :
                model_config = PredRNN_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "predrnnv2":
            from models.PredRNNv2 import Model, PredRNNv2_Config

            if self.mode == 'train' :
                model_config = PredRNNv2_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "simvp_incepu":
            from models.SimVP_IncepU import Model, IncepU_Config

            if self.mode == 'train' :
                model_config = IncepU_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "swinlstm":
            from models.SwinLSTM import Model, SwinLSTM_Config

            if self.mode == 'train' :
                model_config = SwinLSTM_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        elif self.model_name == "tau":
            from models.TAU import Model, TAU_Config

            if self.mode == 'train' :
                model_config = TAU_Config().model_config
                self.configs = config_update(self.configs, model_config)

            self.model = Model

        else:
            raise NotImplementedError(f'{self.model_name} model not implemented')
