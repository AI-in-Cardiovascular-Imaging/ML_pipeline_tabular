from sklearn.model_selection import train_test_split

from feature_corr.data_borg import DataBorg


class SelectionSplit(DataBorg):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.seed = config.meta.seed
        self.state_name = config.meta.state_name
        self.learn_task = config.selection.learn_task
        self.selection_frac = config.selection.selection_frac
        self.stratify = None
        self.frame = self.get_store('ephemeral', self.state_name, 'ephemeral')

    def __call__(self):
        self.set_stratification()
        self.split_data()

    def set_stratification(self):
        """Set stratification"""
        if self.learn_task == 'binary-classification':
            target_frame = self.frame[self.config.meta.target_label]
            self.stratify = target_frame
        elif self.learn_task == 'multi-classification':
            raise NotImplementedError('Multi-classification not implemented')
        elif self.learn_task == 'regression':
            self.stratify = None

    def split_data(self):
        if 0 < self.set_stratification() < 1:
            selection_train, verification_train = train_test_split(
                self.frame,
                stratify=self.stratify,
                test_size=1 - self.selection_frac,
                random_state=self.seed,
            )
            self.set_store('data', self.state_name, 'selection_train', selection_train)
            self.set_store('data', self.state_name, 'verification_train', verification_train)
            self.set_store('data', self.state_name, 'verification_test', None)

        elif self.selection_frac == 0:  # special mode in which entire train data is used for selection and verification
            verification_train, verification_test = train_test_split(
                self.frame,
                stratify=self.stratify,
                test_size=0.2,
                random_state=self.seed,
            )
            self.set_store('data', self.state_name, 'selection_train', verification_train)
            self.set_store('data', self.state_name, 'verification_train', verification_train)
            self.set_store('data', self.state_name, 'verification_test', verification_test)
        else:
            raise ValueError(f'Value {self.selection_frac} is invalid, must be float in (0, 1)')
