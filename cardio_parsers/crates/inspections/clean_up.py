from cardio_parsers.data_borg.data_borg import DataBorg


class CleanUp(DataBorg):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_name = self.config.meta.state_name
        self.label_as_index = self.config.inspection.label_as_index

    def __call__(self):
        self.set_index_by_label()

    def set_index_by_label(self):
        """Set index by label"""
        if isinstance(self.label_as_index, str):
            if self.label_as_index not in self.get_ephemeral_data(self.state_name).columns:
                raise ValueError(f'Label {self.label_as_index} not in data')
            data = self.get_ephemeral_data(self.state_name)
            data = data.set_index(str(self.config.inspection.label_as_index))
            self.set_ephemeral_data(self.state_name, data)
