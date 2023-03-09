from sklearn.model_selection import train_test_split

from cardio_parsers.data_borg import DataBorg


class SelectionSplit(DataBorg):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.selection_fracture = self.config.data_split.selection_fracture

    def __call__(self):

        task, stratify = target_statistics(data, self.target_label)

        if 0 < self.explore_frac < 1:
            explore_data, verification_data = train_test_split(
                data, stratify=stratify, test_size=1 - self.explore_frac, random_state=self.seed
            )
            verification_data_test = None
        elif self.explore_frac == 0:  # special mode in which entire train data is used for exploration and verification
            verification_data, verification_data_test = train_test_split(
                data, stratify=stratify, test_size=0.2, random_state=self.seed
            )
            explore_data = verification_data
        else:
            raise ValueError(f'Value {self.explore_frac} is invalid, must be float in (0, 1)')
