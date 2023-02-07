import os

from loguru import logger
import numpy as np
from scipy.stats import (
    anderson,
    chi2_contingency,
    f_oneway,
    friedmanchisquare,
    kendalltau,
    kruskal,
    mannwhitneyu,
    normaltest,
    pearsonr,
    shapiro,
    spearmanr,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)


# Normality test (check if data has a Gaussian distribution)
def shapiro_test(data):
    """
    Shapiro-Wilk Test
    Assumptions:        Observations in each sample are independent and identically distributed (iid).
    Interpretation:     H0: the sample has a Gaussian distribution.
                        H1: the sample does not have a Gaussian distribution.
    """
    stat, p = shapiro(data)
    print('Shapiro test ', stat, p)


def normal_test(data):
    """
    D’Agostino’s K^2 Test
    Assumptions:        Observations in each sample are independent and identically distributed (iid).
    Interpretation:     H0: the sample has a Gaussian distribution.
                        H1: the sample does not have a Gaussian distribution.
    """
    stat, p = normaltest(data)
    print('Normal test ', stat, p)


def anderson_darling_test(data):
    """
    Anderson-Darling Test
    Assumptions:        Observations in each sample are independent and identically distributed (iid).
    Interpretation:     H0: the sample has a Gaussian distribution.
                        H1: the sample does not have a Gaussian distribution
    """
    result = anderson(data)
    print('Anderson darling test ', result)


# Correlation test (check if two samples are related)
def pearson_correlation(data_1, data_2):
    """
    Pearson’s Correlation Coefficient (Tests whether two samples have a linear relationship)
    Assumptions:        Observations in each sample are independent and identically distributed (iid).
                        Observations in each sample are normally distributed.
                        Observations in each sample have the same variance.
    Interpretation:     H0: the two samples are independent.
                        H1: there is a dependency between the samples
    """
    corr, p = pearsonr(data_1, data_2)
    print('Pearson correlation ', corr, p)


def spearman_correlation(data_1, data_2):
    """
    Spearman’s Rank Correlation (Tests whether two samples have a monotonic relationship)
    Assumptions:        Observations in each sample are independent and identically distributed (iid).
                        Observations in each sample can be ranked.
    Interpretation:     H0: the two samples are independent.
                        H1: there is a dependency between the samples.
    """
    corr, p = spearmanr(data_1, data_2)
    print('Spearman correlation ', corr, p)


def kendalltau_correlation(data_1, data_2):
    """
    Kendall’s Rank Correlation (Tests whether two samples have a monotonic relationship)
    Assumptions:        Observations in each sample are independent and identically distributed (iid).
                        Observations in each sample can be ranked.
    Interpretation:     H0: the two samples are independent.
                        H1: there is a dependency between the samples.
    """
    corr, p = kendalltau(data_1, data_2)
    print('Kendaltau correlation ', corr, p)


def chi_squared_test(table):
    """
    Chi-Squared Test (Tests whether two categorical variables are related or independent)
    Assumptions:        Observations used in the calculation of the contingency table are independent.
                        25 or more examples in each cell of the contingency table.
    Interpretation      H0: the two samples are independent.
                        H1: there is a dependency between the samples.

    """
    stat, p, dof, expected = chi2_contingency(table)
    print('Chi squared test ', stat, p, dof, expected)


# Parametric Statistical Hypothesis Tests (use to compare data samples)
def t_test(data_1, data_2):
    """
    Student’s t-test (Tests whether the means of two independent samples are significantly different)

    Assumptions:        Observations in each sample are independent and identically distributed (iid).
                        Observations in each sample are normally distributed.
                        Observations in each sample have the same variance.
    Interpretation:     H0: the means of the samples are equal.
                        H1: the means of the samples are unequal.
    """
    stat, p = ttest_ind(data_1, data_2)
    print('t_test ', stat, p)


def paired_student_test(data_1, data_2):
    """
    Paired Student’s t-test (Tests whether the means of two paired samples are significantly different.)

    Assumptions:        Observations in each sample are independent and identically distributed (iid).
                        Observations in each sample are normally distributed.
                        Observations in each sample have the same variance.
                        Observations across each sample are paired.
    Interpretation:     H0: the means of the samples are equal.
                        H1: the means of the samples are unequal.
    """
    stat, p = ttest_rel(data_1, data_2)
    print('paired student test ', stat, p)


def anova(data_1, data_2):
    """
    Analysis of Variance Test (ANOVA) (Tests whether the means of two or more independent samples are significantly different)

    Assumptions:        Observations in each sample are independent and identically distributed (iid).
                        Observations in each sample are normally distributed.
                        Observations in each sample have the same variance.
    Interpretation:     H0: the means of the samples are equal.
                        H1: one or more of the means of the samples are unequal.
    """
    stat, p = f_oneway(data_1, data_2, ...)
    print('Anova', stat, p)


def mannwhitneyu_test(data_1, data_2):
    """
    Nonparametric Statistical Hypothesis Tests (use to compare data samples)

    Mann-Whitney U Test (Tests whether the distributions of two independent samples are equal or not)

    Assumptions:        Observations in each sample are independent and identically distributed (iid).
                        Observations in each sample can be ranked.
    Interpretation:     H0: the distributions of both samples are equal.
                        H1: the distributions of both samples are not equal.
    """
    stat, p = mannwhitneyu(data_1, data_2)
    print('Mannwithneyu test ', stat, p)


def wilcoxon_test(data_1, data_2):
    """
    Wilcoxon Signed-Rank Test (Tests whether the distributions of two paired samples are equal or not)

    Assumptions:        Observations in each sample are independent and identically distributed (iid).
                        Observations in each sample can be ranked.
                        Observations across each sample are paired.
    Interpretation:     H0: the distributions of both samples are equal.
                        H1: the distributions of both samples are not equal.
    """
    stat, p = wilcoxon(data_1, data_2)
    print('Wilcoxon test ', stat, p)


def kruskal_test(data_1, data_2):
    """
    Kruskal-Wallis H Test (Tests whether the distributions of two paired samples are equal or not)

    Assumptions:        Observations in each sample are independent and identically distributed (iid).
                        Observations in each sample can be ranked.
    Interpretation:     H0: the distributions of all samples are equal.
                        H1: the distributions of one or more samples are not equal.
    """
    stat, p = kruskal(data_1, data_2, ...)
    print('Kruskal test ', stat, p)


def friedmanchisquare_test(data_1, data_2):
    """
    Friedman Test (Tests whether the distributions of two or more paired samples are equal or not)

    Assumptions:        Observations in each sample are independent and identically distributed (iid).
                        Observations in each sample can be ranked.
                        Observations across each sample are paired.

    Interpretation:     H0: the distributions of all samples are equal.
                        H1: the distributions of one or more samples are not equal.
    """
    stat, p = friedmanchisquare(data_1, data_2, ...)
    print('Friedmanchisquare test ', stat, p)


if __name__ == '__main__':
    data = np.random.rand(50)
    data = [1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 3, 4, 5, 6, 4, 5, 4, 5, 4]

    # plt.plot(x, data)
    plt.hist(data, bins=7)
    plt.show()

    # data = np.random.normal(20, 1, 5000)
    print(data)
