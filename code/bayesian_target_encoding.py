import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split


class BaseTargetEncoder(object):
    """ Base class for Target Encoding

    Warning: This class should not be used directly.

    Use the following classes instead:
    1) TargetEncoder: for target encoding without regularization
    2) GaussianTargetEncoder: for regression tasks
    3) BetaTargetEncoder: for binary classification tasks
    4) DicheletTargetEncoder: for multiclass classification tasks (NOT IMPLEMENTED YET)
    """
    def __init__(self, category_col, path):
        self.MISSING_VALUE_STRING = '--null_value--'
        self.category_col = category_col
        self.grouped_data = None
        self.priors = {}
        self.posteriors = {}
        self.stat_types = None
        self.encodings = None
        self.missing_imputations = {}
        self.name = None
        self.valid_tasks = set()
        self.path = os.path.join('common_intra_model', 'model_params') if path is None else path


    def infer_task(self, y):
        ''' Infer type of ML task

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            The target values (class labels) as integers

        Returns
        -------
        task_type : string
            One of the set {regression, binary_classification, multiclass_classification}
        '''
        n_labels = len(y)
        unique_labels = set(y)
        cardinality = len(unique_labels)

        # Labels must contain no null values
        assert np.sum(pd.isnull(unique_labels)) == 0

        if cardinality < 2:
            raise ValueError(f'Not enough unique labels: {cardinality} found')
        elif cardinality == 2:
            if unique_labels == {0, 1}:
                task_type = 'binary_classification'
            else:
                raise ValueError(f'Labels must be 0 and 1 for beta encoding: {unique_labels} found')
        elif (cardinality / n_labels) < 0.01:
            # number of unique labels is small relative to sample size
            task_type = 'multiclass_classification'
        else:
            task_type = 'regression'

        return task_type


    def fit(self, X, y, fit_method):
        ''' Build a target encoder from the training set (X, y)

        Steps
        -----
        1. Validate data.
            - Make sure the correct Encoder is being used
            - Check that the data is categorical
        2. Get statistics for each level of the data
        3. Estimate prior parameters
        4. Given a prior and data (evidence), calculate parameters of the posterior distributions
        5. Using the posteriors, calculate the expected value of every stat_type of interest
        6. Save the encodings (expected value of every stat_type)

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The training input samples

        y : array-like of shape (n_samples,)
            The target values (class labels) as integers

        fit_method : string
            Method used to estimate parameters of the prior distribution

        Returns
        -------
        self : BaseTargetEncoder
            Fitted target encoder
        '''
        assert self.MISSING_VALUE_STRING not in X.unique()

        # remove data when label is missing
        X = X[pd.notnull(y)]
        y = y[pd.notnull(y)]

        label_type = self.infer_task(y)
        if label_type not in self.valid_tasks:
            raise ValueError(f'Inferred ML task {label_type}, is not a subset of {self.valid_tasks}')

        df = pd.DataFrame(zip(X, y), columns=['X', 'y'])
        if df['X'].dtype != object:
            raise ValueError(f'{self.category_col} is not an object')

        self.grouped_data = self.__get_group_statistics(df)

        self.priors = self.get_prior_params(df, fit_method)
        self.posteriors = self.get_posterior_params()

        for stat_type in self.stat_types:
            self.encodings[stat_type] = pd.DataFrame(self.get_encodings(stat_type), columns=[stat_type])

            # Fill in missing values using weighted average of values from each level
            tmp_df = pd.merge(
                self.encodings[stat_type],
                self.grouped_data['count'],
                left_index=True,
                right_index=True,
                how='inner'
            )
            self.missing_imputations[stat_type] = np.average(tmp_df[stat_type], weights=tmp_df['count'])

        self.save_params()
        return None


    def transform(self, X, stat_type):
        ''' Transform X using a target encoder

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            The training input samples

        stat_type : string
            Measures of central tendancy and statistical dispersion which describe the distribution.
            e.g. mean, mode, median, variance, skewness, kurtosis

        Returns
        -------
        X_out : Series
            Transformed input
        '''
        self.load_params()
        df_output = pd.merge(
            X,
            self.encodings[stat_type],
            left_on=self.category_col,
            right_index=True,
            how='left'
        )

        # category column is missing data
        df_output.loc[pd.isnull(df_output[self.category_col]), stat_type] = self.encodings[stat_type].loc[self.MISSING_VALUE_STRING, stat_type]

        # category has never been seen in training set
        df_output.loc[pd.isnull(df_output[stat_type]), stat_type] = self.missing_imputations[stat_type]
        return df_output[stat_type]


    def __get_group_statistics(self, df):
        ''' Calculate statistics for each level of the provided DataFrame

        Parameters
        ----------
        df : pandas DataFrame, must include 2 columns (X, y)

        Returns
        -------
        df_grouped: pandas DataFrame
            Aggregated DataFrame created by grouping the input df
        '''
        df_grouped = df[['X', 'y']].groupby('X').agg(['count', 'sum', 'mean', 'median', 'var'])['y']

        # add a new level for missing values
        df_missing = df.loc[pd.isnull(df['X']), 'y']
        missing_row = pd.DataFrame(
            data={
                'sum': df_missing.sum(),
                'count': df_missing.count(),
                'mean': df_missing.mean(),
                'median': df_missing.median(),
                'var': df_missing.var(),
            },
            index=[self.MISSING_VALUE_STRING]
        )

        df_grouped = pd.concat([df_grouped, missing_row], axis=0)
        return df_grouped


    def save_params(self):
        ''' Saves fit parameters to file '''
        encodings_dict = {stat_type: df.to_dict()[stat_type] for stat_type, df in self.encodings.items()}
        params = {
            'encodings': encodings_dict,
            'MISSING_VALUE_STRING': self.MISSING_VALUE_STRING,
            'missing_imputations': self.missing_imputations,
        }
        with open(os.path.join(self.path, f'{self.category_col}_{self.name}_params.json'), 'w') as f:
            json.dump(params, f)
        return None


    def load_params(self):
        ''' Loads fit parameters from file '''
        with open(os.path.join(self.path, f'{self.category_col}_{self.name}_params.json'), 'r') as f:
            params = json.load(f)

        self.encodings = {
            stat_type: pd.DataFrame.from_dict(d, orient='index', columns=[stat_type])
            for stat_type, d in params['encodings'].items()
        }
        self.MISSING_VALUE_STRING = params['MISSING_VALUE_STRING']
        self.missing_imputations = params['missing_imputations']
        return None


class TargetEncoder(BaseTargetEncoder):
    ''' Target Encoding without bayesian adjustments

    If 'mean' is chosen as the stat_type, this is identical to how lightGBM and CatBoost
    handle categorical columns.

    Parameters
    ----------
    category_col : string
        name of the category column

    path : str
        Path to directory where model parameters are saved
    '''
    def __init__(self, category_col, path=None):
        super().__init__(category_col, path)

        self.stat_types = ['mean', 'median', 'var']
        self.encodings = {stat_type: {} for stat_type in self.stat_types}
        self.name = 'target'
        self.valid_tasks = {'regression', 'multiclass_classification', 'binary_classification'}


    def fit(self, X, y, fit_method='default'):
        ''' Build a target encoder from the training set (X, y) '''
        super().fit(X, y, fit_method)
        return None


    def transform(self, X, stat_type):
        ''' Transform X using a target encoder '''
        return super().transform(X, stat_type)


    def get_prior_params(self, df, fit_method):
        ''' Estimate prior parameters '''
        not_null_y = df.loc[pd.notnull(df['y']), 'y']
        return {
            'mean': not_null_y.mean(),
            'var': not_null_y.var(),
        }


    def get_posterior_params(self): return {}


    def get_encodings(self, stat_type):
        ''' Calculate the encoding for a given stat_type

        Returns
        -------
        output : Pandas Series
            Index of series is the name of the level (e.g. CA, AZ for state).
            Values of series are the encoded values
        '''
        return self.grouped_data[stat_type]


class BetaTargetEncoder(BaseTargetEncoder):
    ''' Bayesian Target Encoding for binary classification tasks

    Inspired by the following blog post: http://varianceexplained.org/r/empirical_bayes_baseball/
    Code is adapted, but heavily modified, from: https://www.kaggle.com/mmotoki/avito-target-encoding

    Parameters
    ----------
    category_col : string
        name of the category column

    path : str
        Path to directory where model parameters are saved
    '''
    def __init__(self, category_col, path=None):
        super().__init__(category_col, path)

        self.stat_types = ['mean', 'var']
        self.encodings = {stat_type: {} for stat_type in self.stat_types}
        self.name = 'betaTarget'
        self.valid_tasks = {'binary_classification'}


    def fit(self, X, y, fit_method='moments'):
        ''' Build a target encoder from the training set (X, y) '''
        super().fit(X, y, fit_method)
        return None


    def transform(self, X, stat_type):
        ''' Transform X using a target encoder '''
        return super().transform(X, stat_type)


    def get_prior_params(self, df, fit_method):
        ''' Estimate prior parameters

        Parameters
        ----------
        df : pandas DataFrame, must include a column (y) corresponding to the target

        fit_method : string
            Method used to estimate parameters of the prior

        Returns
        -------
        priors : dict
            Contains all parameters necessary to specify prior distribution
            Maps a parameter to a float.
        '''
        not_null_y = df.loc[pd.notnull(df['y']), 'y']
        data_mean = not_null_y.mean()
        data_median = not_null_y.median()
        data_var = not_null_y.var()

        if fit_method == 'moments':
            # approximate alpha and beta parameters using the method of moments
            alpha = (data_mean ** 2) * (((1 - data_mean) / (data_var ** 2)) - (1 / data_mean))
            beta = alpha * ((1 / data_mean) - 1)
        elif fit_method == 'mle':
            # Maximum Likelihood Estimation
            # https://www.itl.nist.gov/div898/handbook/eda/section3/eda366h.htm
            #
            # Note: don't fit distribution to small samples
            #   i.e. if one example switched from positive to negative, should not affect value by a large amount
            sample_size_threshold = 2 * (1 / data_mean)
            is_large_enough = self.grouped_data['count'] > sample_size_threshold

            # fix the limit that has the least number of examples unless exactly equal, then fix both
            loc, scale = 0, 1
            if data_median == loc:
                # variable loc, fixed scale
                alpha, beta, loc, scale = stats.beta.fit(
                    self.grouped_data.loc[is_large_enough, 'mean'],
                    loc=loc,
                    fscale=scale
                )
            elif data_median == scale:
                # fixed loc, variable scale
                alpha, beta, loc, scale = stats.beta.fit(
                    self.grouped_data.loc[is_large_enough, 'mean'],
                    floc=loc,
                    scale=scale
                )
            else:
                # fixed loc, fixed scale
                alpha, beta, loc, scale = stats.beta.fit(
                    self.grouped_data.loc[is_large_enough, 'mean'],
                    floc=loc,
                    fscale=scale
                )
        else:
            raise ValueError(f'Fit method not implemented yet: {fit_method}')
        return {
            'alpha': alpha,
            'beta': beta,
        }


    def get_posterior_params(self):
        ''' Calculate posterior parameters

        Returns
        -------
        posteriors : dict
            Contains all parameters necessary to specify posterior distributions for each level of the category.
            Maps a parameter to a dataframe.
        '''
        return {
            'alphas': self.priors['alpha'] + self.grouped_data['sum'],
            'betas': self.priors['beta'] + self.grouped_data['count'] - self.grouped_data['sum'],
        }


    def get_encodings(self, stat_type):
        ''' Calculate the encoding for a given stat_type

        Returns
        -------
        output : Pandas Series
            Index of series is the name of the level (e.g. CA, AZ for state).
            Values of series are the encoded values
        '''
        alpha = self.posteriors['alphas']
        beta = self.posteriors['betas']
        if stat_type == 'mean':
            num = alpha
            dem = alpha + beta
        elif stat_type == 'mode':
            num = alpha - 1
            dem = alpha + beta - 2
        elif stat_type == 'median':
            num = alpha - (1 / 3)
            dem = alpha + beta - (2 / 3)
        elif stat_type == 'var':
            num = alpha * beta
            dem = ((alpha + beta) ** 2) * (alpha + beta + 1)
        elif stat_type == 'skewness':
            num = 2 * (beta - alpha) * np.sqrt(alpha + beta + 1)
            dem = (alpha + beta + 2) * np.sqrt(alpha * beta)
        elif stat_type == 'kurtosis':
            num = 6 * ((alpha - beta) ** 2) * (alpha + beta + 1) - (alpha * beta * (alpha + beta + 2))
            dem = alpha * beta * (alpha + beta + 2) * (alpha + beta + 3)
        elif stat_type == 'none':
            # no beta encoding, just the target mean for each category
            num = self.data['n_positives']
            dem = self.data['n_samples']

        return num / dem


class GaussianTargetEncoder(BaseTargetEncoder):
    ''' Bayesian Target Encoding for regression tasks

    Valid encodings are 'mean' and 'var'. Why could capturing the variance be important?
    - Both the expected value and the uncertainty of the estimate might be important.

    There are multiple options for choosing a conjugate prior.
    1) Unknown mean with known variance
        - use Normal prior with Normal conjugate
        - hyperparams (sample mean, sample precision)
    2) Known mean with unknown variance
        - use Normal prior with Inverse-Gamma conjugate
        - hyperparams (alpha, beta), alpha = n_samples / 2, sample variance = beta / alpha, beta = (sum squared deviations) / 2
    3) Unknown mean and unknown variance
        - Use Normal prior with Normal-Inverse-Gamma conjugate
        - hyperparams (alpha, beta)

    For simplicity, we choose the first option which has a single hyperparameter, variance.
    Variance represents the amount of certainty we have in our prior and acts as a regularization parameter.
    The smaller the variance, the more confident we are in the prior, thus the larger the regularization.

    Parameters
    ----------
    category_col : string
        name of the category column

    tau : positive float
        Precision, aka (1 / variance)

    path : str
        Path to directory where model parameters are saved
    '''
    def __init__(self, category_col, tau=1, path=None):
        super().__init__(category_col, path)

        assert tau > 0
        self.priors = {'tau': tau}
        self.stat_types = ['mean', 'var']
        self.encodings = {stat_type: {} for stat_type in self.stat_types}
        self.name = 'gaussianTarget'
        self.valid_tasks = {'regression'}


    def fit(self, X, y, fit_method='default'):
        ''' Build a target encoder from the training set (X, y) '''
        super().fit(X, y, fit_method)
        return None


    def transform(self, X, stat_type):
        ''' Transform X using a target encoder '''
        return super().transform(X, stat_type)


    def get_prior_params(self, df, fit_method):
        ''' Estimate prior parameters

        Parameters
        ----------
        df : pandas DataFrame, must include a column (y) corresponding to the target

        fit_method : string
            Method used to estimate parameters of the prior

        Returns
        -------
        priors : dict
            Contains all parameters necessary to specify prior distribution
            Maps a parameter to a float.
        '''
        not_null_y = df.loc[pd.notnull(df['y']), 'y']
        if fit_method == 'default':
            priors = {
                'mean': not_null_y.mean(),
                'var': not_null_y.var(),
            }
        elif fit_method in {'moments', 'mle'}:
            priors = {
                'mean': self.grouped_data['mean'].mean(),
                'var': self.grouped_data['mean'].var(),
                'tau': 1 / self.grouped_data['mean'].var(),
            }

        # TODO: replace with "|"" operator (requires python 3.9)
        # return self.priors | priors
        return {**self.priors, **priors}


    def get_posterior_params(self):
        ''' Calculate posterior parameters

        Returns
        -------
        posteriors : dict
            Contains all parameters necessary to specify posterior distributions for each level of the category.
            Maps a parameter to a dataframe.
        '''
        posteriors = {
            'tau': self.priors['tau'] + (self.grouped_data['count'] / self.grouped_data['var']),
        }
        return posteriors


    def get_encodings(self, stat_type):
        ''' Calculate the encoding for a given stat_type

        Returns
        -------
        output : Pandas Series
            Index of series is the name of the level (e.g. CA, AZ for state).
            Values of series are the encoded values
        '''
        if stat_type == 'mean':
            num = (self.priors['tau'] * self.priors['mean']) + \
                (self.grouped_data['count'] * self.grouped_data['mean'] / self.grouped_data['var'])
            den = self.posteriors['tau']
        elif stat_type == 'var':
            num = 1.0
            den = self.posteriors['tau']
        elif stat_type == 'precision':
            num = self.posteriors['tau']
            den = 1.0

        return num / den


class DirichletTargetEncoder(BaseTargetEncoder):
    ''' Bayesian Target Encoding for multiclass classification tasks

    Parameters
    ----------
    category_col : string
        name of the category column

    path : str
        Path to directory where model parameters are saved
    '''
    def __init__(self, category_col, path):
        super().__init__(category_col, path)

        self.stat_types = ['mean', 'var']
        self.encodings = {stat_type: {} for stat_type in self.stat_types}
        self.name = 'dirichletTarget'
        self.valid_tasks = {'multiclass_classification'}

        raise NotImplementedError('Can be implemented if there is a need')


    def fit(self, X, y, fit_method='default'):
        ''' Build a target encoder from the training set (X, y) '''
        super().fit(X, y, fit_method)
        return None


    def transform(self, X, stat_type):
        ''' Transform X using a target encoder '''
        return super().transform(X, stat_type)


    def get_prior_params(self, df, fit_method): return {}


    def get_posterior_params(self): return {}


    def get_encodings(self, stat_type): return {}


def make_diagnostic_plot(X, y, cat_col, stat_type, hyperparams={}):
    ''' Plots the bayesian target encoded variable before/after adjustment

    Ref: http://varianceexplained.org/r/empirical_bayes_baseball/
    '''
    base_encoder = BaseTargetEncoder(cat_col)
    target_encoder = TargetEncoder(cat_col)
    task_type = base_encoder.infer_task(y)
    if task_type == 'regression':
        bayesian_encoder = GaussianTargetEncoder(cat_col, **hyperparams)
    elif task_type == 'binary_classification':
        bayesian_encoder = BetaTargetEncoder(cat_col, **hyperparams)
    elif task_type == 'multiclass_classification':
        bayesian_encoder = DirichletTargetEncoder(cat_col, **hyperparams)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    target_encoder.fit(X_train, y_train)
    bayesian_encoder.fit(X_train, y_train)

    target_encodings = target_encoder.encodings[stat_type]
    bayesian_encodings = bayesian_encoder.encodings[stat_type]

    df_target = pd.DataFrame(target_encodings)
    df_target.columns = ['target_value']
    df_bayesian = pd.DataFrame(bayesian_encodings)
    df_bayesian.columns = ['bayesian_value']
    df_tmp = pd.merge(df_target, df_bayesian, left_index=True, right_index=True, how='left')
    df_tmp = pd.merge(df_tmp, target_encoder.grouped_data['count'], left_index=True, right_index=True, how='left')

    xmin, xmax = df_tmp['target_value'].min(), df_tmp['target_value'].max()
    ymin, ymax = df_tmp['bayesian_value'].min(), df_tmp['bayesian_value'].max()
    fig = df_tmp.plot(kind='scatter', x='target_value', y='bayesian_value', c='count', cmap='winter', s=35)
    plt.hlines(
        y=target_encoder.priors[stat_type],
        xmin=xmin,
        xmax=xmax,
        color='r',
        linestyle='--',
        label=f'line of no evidence (prior {stat_type})'
    )
    plt.plot(
        [.95 * min(xmin, ymin), 1.05 * max(xmax, ymax)],
        [.95 * min(xmin, ymin), 1.05 * max(xmax, ymax)],
        color='r',
        label='x=y'
    )
    plt.title(f'{target_encoder.category_col}\nTarget Encoding: {stat_type}')
    plt.xlim([.95 * xmin, 1.05 * xmax])
    plt.ylim([.95 * ymin, 1.05 * ymax])
    plt.xlabel('No adjustment')
    plt.ylabel('With bayesian adjustment')
    plt.legend(loc='best')
    plt.show()
    return fig
