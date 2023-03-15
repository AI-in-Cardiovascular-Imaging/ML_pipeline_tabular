from feature_corr.data_borg import DataBorg


class Report(DataBorg):
    def __init__(self):
        super().__init__()

    def __call__(self):
        return self.get_all_features()
