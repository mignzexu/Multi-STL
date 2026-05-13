from utils import Z_Score_SD, Z_Score

class Load_Standardizer:
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.std_method = self.configs.std_method
        self.standardizer = None
        self.load_standardizer()

    def load_standardizer(self):
        if self.std_method == None:
            pass
        elif self.std_method == 'z_score':
            self.standardizer = Z_Score(self.configs)
        elif self.std_method == 'z_score_sd':
            self.standardizer = Z_Score_SD(self.configs)
        # elif self.std_method == 'log1p':
        #     self.standardizer = Log1p_Z_Score(self.configs)
        else:
            raise ValueError('Invalid standardizing method')



    
