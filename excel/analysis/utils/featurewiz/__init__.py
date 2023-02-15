# -*- coding: utf-8 -*-
#################################################################
#     featurewiz - advanced feature engineering and best features selection in single line of code
#     Python v3.6+
#     Created by Ram Seshadri
#     Licensed under Apache License v2
################################################################################
# Version
from .featurewiz import (
    EDA_binning_numeric_column_displaying_bins,
    EDA_classify_and_return_cols_by_type,
    EDA_classify_features_for_deep_learning,
    EDA_find_outliers,
    EDA_find_skewed_variables,
    EDA_randomly_select_rows_from_dataframe,
    FE_add_age_by_date_col,
    FE_add_groupby_features_aggregated_to_dataframe,
    FE_capping_outliers_beyond_IQR_Range,
    FE_concatenate_multiple_columns,
    FE_count_rows_for_all_columns_by_group,
    FE_create_categorical_feature_crosses,
    FE_create_interaction_vars,
    FE_discretize_numeric_variables,
    # FE_drop_rows_with_infinity,
    FE_find_and_cap_outliers,
    FE_get_latest_values_based_on_date_column,
    FE_kmeans_resampler,
    FE_split_add_column,
    FE_split_one_field_into_many,
    FE_start_end_date_time_features,
    FE_transform_numeric_columns_to_bins,
    FeatureWiz,
    featurewiz,
    split_data_n_ways,
)
from .ml_models import (
    complex_LightGBM_model,
    complex_XGBoost_model,
    data_transform,
    simple_LightGBM_model,
    simple_XGBoost_model,
)
from .sulov_method import FE_remove_variables_using_SULOV_method
