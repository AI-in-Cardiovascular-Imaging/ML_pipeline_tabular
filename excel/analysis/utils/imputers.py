# imputer = IterativeImputer(
#     initial_strategy='median', max_iter=100, random_state=self.seed, keep_empty_features=True
# )
# if 'subject' in table.columns:
#     tmp = table['subject']
#     table = table.drop('subject', axis=1)
#     imputed_data = imputer.fit_transform(table)
#     imputed_data = pd.DataFrame(imputed_data, index=table.index, columns=table.columns)
#     table = pd.concat((tmp, imputed_data), axis=1)
# else:
#     imputed_data = imputer.fit_transform(table)
#     table = pd.DataFrame(imputed_data, index=table.index, columns=table.columns)
#
#
# Impute missing metadata if desired
# if self.impute:
#     indicator = MissingIndicator(missing_values=np.nan, features='all')
#     mask_missing_values = indicator.fit_transform(tables)
#     style_df = pd.DataFrame('', index=tables.index, columns=tables.columns)
#     style_df = style_df.mask(mask_missing_values, 'background-color: cyan')
#
#     tables = self.impute_data(tables)
#
#     os.makedirs(self.merged_dir, exist_ok=True)
#     tables.style.apply(lambda _: style_df, axis=None).to_excel(
#         os.path.join(self.merged_dir, f'{self.experiment_name}_highlighted_missing_metadata.xlsx')
#     )
#
# else:  # remove patients with any NaN values
#     logger.info(f'Number of patients before dropping NaN metadata: {len(tables.index)}')
#     tables = tables.dropna(axis=0, how='any')
#     logger.info(f'Number of patients after dropping NaN metadata: {len(tables.index)}')

from sklearn.experimental import enable_iterative_imputer  # because of bug in sklearn
from sklearn.impute import IterativeImputer, MissingIndicator
import pandas as pd
from loguru import logger
from omegaconf import DictConfig


class Imputers:

    def __init__(self) -> None:
        pass

    def drop_missing_data(self, data):
        """Drop patients with any NaN values"""
        logger.info(f'Number of patients before dropping NaN metadata: {len(data.index)}')
        data = data.dropna(axis=0, how='any')
        logger.info(f'Number of patients after dropping NaN metadata: {len(data.index)}')
        return data

    def iterative_imputer(self, data):
        """Impute missing metadata"""
        imputer = IterativeImputer(
            initial_strategy='median',
            max_iter=100,
            random_state=self.seed,
            keep_empty_features=True,
        )
        if 'subject' in data.columns:
            tmp = data['subject']
            data = data.drop('subject', axis=1)
            imputed_data = imputer.fit_transform(data)
            imputed_data = pd.DataFrame(imputed_data, index=data.index, columns=data.columns)
            data = pd.concat((tmp, imputed_data), axis=1)

        return data

    def simple_imputer(self):
        """"""
        pass

    def missing_indicator_imputer(self):
        """"""
        pass

    def knn_imputer(self):
        """"""
        pass


