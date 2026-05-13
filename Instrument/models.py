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

        else:
            raise NotImplementedError(f'{self.model_name} model not implemented')
