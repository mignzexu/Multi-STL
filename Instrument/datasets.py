from utils import config_update

class Dataset_Instrument:
    def __init__(self, configs):
        self.configs = configs
        self.data_name = self.configs.dataset
        self.train_data = None
        self.valid_data = None
        self.test_data = None

    def load_dataset(self, mode='train'):
        
        if self.data_name == 'SDweather':
            from datasets.SDweather import SD_Dataloader, SDweather_config

            if mode == 'train' : 
                if self.configs.data_config is None:
                    config_name = "config2"
                else:
                    config_name = self.configs.data_config
                data_config = SDweather_config(config_name).data_config
                self.configs = config_update(self.configs, data_config)

            if mode == 'train' or mode == 'retrain':
                self.train_data = SD_Dataloader(self.configs, 'train')
            elif mode == 'valid' :
                self.valid_data = SD_Dataloader(self.configs, 'valid')
            elif mode == 'test':
                self.test_data = SD_Dataloader(self.configs, 'test')
            else:
                raise NotImplementedError(f'{mode} mode not implemented')

        elif self.data_name == 'weatherbench':
            from datasets.WeatherBench import WB_Dataloader, WeatherBench_config
            
            if mode == 'train':

                if self.configs.data_config is None:
                    config_name = "test"
                else:
                    config_name = self.configs.data_config

                data_config = WeatherBench_config(config_name).data_config
                self.configs = config_update(self.configs, data_config)

            if mode == 'train' or mode == 'retrain':
                self.train_data = WB_Dataloader(self.configs, 'train')
            elif mode == 'valid':
                self.valid_data = WB_Dataloader(self.configs, 'valid')
            elif mode == 'test':
                self.test_data = WB_Dataloader(self.configs, 'test')
            else:
                raise NotImplementedError(f'{mode} mode not implemented')
        else:
            raise NotImplementedError(f'{self.data_name} dataset not implemented')
    
