from loguru import logger

from feature_corr.data_borg import DataBorg


class CleanUp(DataBorg):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.state_name = self.config.meta.state_name
        self.label_as_index = self.config.inspection.label_as_index
        logger.info(f'Running -> {self.__class__.__name__}')

    def __call__(self):
        self.set_index_by_label()

    def set_index_by_label(self):
        """Set index by label"""
        logger.info(f'Reindex table by name -> {self.label_as_index}')
        if isinstance(self.label_as_index, str):
            if self.label_as_index not in self.get_frame('ephemeral').columns:
                raise ValueError(f'Label {self.label_as_index} not in data')
            data = self.get_frame('ephemeral')
            data = data.set_index(str(self.config.inspection.label_as_index))
            self.set_frame('ephemeral', data)
