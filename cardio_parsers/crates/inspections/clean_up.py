from cardio_parsers.data_borg.data_borg import DataBorg


class CleanUp(DataBorg):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_name = self.config.meta.state_name

    def __call__(self):
        self.set_index_by_label()

    def set_index_by_label(self):
        if isinstance(self.config.inspection.label_as_index, str):
            # print(self.get_ephemeral_data().columns)
            # if self.config.inspection.label_as_index not in self.get_ephemeral_data().columns:
            #     raise ValueError(f'Label {self.config.inspection.label_as_index} not in data')
            data = self.get_ephemeral_data(self.state_name)
            data = data.set_index(str(self.config.inspection.label_as_index))
            self.set_ephemeral_data(self.state_name, data)
