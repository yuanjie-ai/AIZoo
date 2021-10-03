#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project      : aizoo.
# @File         : feature_selector
# @Time         : 2021/9/30 下午6:43
# @Author       : yuanjie
# @WeChat       : 313303303
# @Software     : PyCharm
# @Description  :

"""https://www.cnblogs.com/nolonely/p/6435083.html
1、为什么要做特征选择
在有限的样本数目下，用大量的特征来设计分类器计算开销太大而且分类性能差。
2、特征选择的确切含义
将高维空间的样本通过映射或者是变换的方式转换到低维空间，达到降维的目的，然后通过特征选取删选掉冗余和不相关的特征来进一步降维。
3、特征选取的原则
获取尽可能小的特征子集，不显著降低分类精度、不影响类分布以及特征子集应具有稳定适应性强等特点
"""

import scipy as sp

from functools import partial
from sklearn.utils import shuffle, check_random_state
from sklearn.preprocessing import scale as zscore
from sklearn.model_selection import train_test_split

# ME
from meutils.pipe import *
from aizoo.utils.model_utils import get_imp
from aizoo.utils.check_utils import check_classification


class SimpleFS(object):
    """粗筛
    高缺失率：
    低方差（高度重复值）：0.5%~99.5%分位数内方差为0的初筛
    高相关：特别高的初筛，根据重要性细筛 TODO
    低重要性：
    召回高IV：

    """

    def __init__(self, df: pd.DataFrame, exclude=None):
        self.to_drop_dict = {}
        if exclude is None:
            exclude = []

        assert isinstance(exclude, list)
        assert isinstance(df, pd.DataFrame)

        self.df = df

        exclude += df.select_dtypes(['datetime64[ns]', object]).columns.tolist()
        print(f"Exclude Fetures: {exclude}")

        if exclude:
            self.feats = df.columns.difference(exclude).tolist()
        else:
            self.feats = df.columns.tolist()

    def run(self):
        df = self.df.copy()

        with timer('干掉高缺失'):
            self.to_drop_dict['filter_missing'] = self.filter_missing()

        with timer('干掉低方差'):
            self.to_drop_dict['filter_variance'] = self.filter_variance()

        with timer('干掉高相关'):
            pass

        return df

    def filter_missing(self, feats=None, threshold=0.95):
        """
        :param feat_cols:
        :param threshold:
        :param as_na: 比如把-99当成缺失值
        :return:
        """
        if feats is None:
            feats = self.feats

        to_drop = self.df[feats].isna().mean()[lambda x: x > threshold].index.tolist()
        print('%d features with greater than %0.2f missing values.' % (len(to_drop), threshold))
        return to_drop

    def _filter_variance(self, feat, df):
        var = df[feat][lambda x: x.between(x.quantile(0.005), x.quantile(0.995))].var()
        return '' if var else feat

    def filter_variance(self, feats=None, max_worker=4):
        if feats is None:
            feats = self.feats

        _filter_variance = partial(self._filter_variance, df=self.df)
        with ProcessPoolExecutor(min(max_worker, len(feats))) as pool:
            to_drop = pool.map(_filter_variance, tqdm(feats, 'Filter Variance ...'))
            to_drop = [feat for feat in to_drop if feat]
        print('%d features with 0 variance in 0.5 ~ 99.5 quantile.' % len(to_drop))
        return to_drop


class FS(object):

    def __init__(self, estimator, verbose=0,
                 importance_type='split', importance_normlize=True,
                 percent=95, alpha=0.05, two_step=True,
                 ):
        self.estimator = estimator
        self.estimator_name = str(estimator).lower()
        self.verbose = verbose

        self.importance_type = importance_type  # split/shap/permutaion
        self.importance_type_normlize = importance_normlize

        self.alpha = alpha
        self.two_step = two_step
        self.percent = percent * 100 if 0 <= percent <= 1 else percent

    def fit(self, X, y, n_trials=10, sample_weight=None):

        # setup variables for Boruta
        n_sample, n_feat = X.shape

        # holds the decision about each feature:
        # 0  - default state = tentative in original code
        # 1  - accepted in original code
        # -1 - rejected in original code
        dec_reg = np.zeros(n_feat)

        # counts how many times a given feature was more important than the best of the shadow features
        hit_reg = np.zeros(n_feat)

        # these record the history of the iterations
        imp_history = np.zeros((n_trials, n_feat))  # 记录真实特征历史重要性, 输出排序
        imp_history[:] = np.nan

        # 影子特征历史重要性最大值
        sha_max_history = []

        # main feature selection loop
        for trial in tqdm(range(n_trials), desc='🔥'):
            if not np.any(dec_reg == 0):  # 存在未拒绝的就继续迭代
                break
            # make sure we start with a new tree in each iteration
            seed = np.random.randint(0, 10 ** 6)

            # 通过模型获取shap值
            imp_real, imp_sha = self._add_shadows_get_imps(X, y, dec_reg, sample_weight, seed=seed + 2)
            imp_history[trial] = imp_real  # record importance history

            # get the threshold of shadow importances we will use for rejection
            imp_sha_max = np.percentile(imp_sha, self.percent)
            sha_max_history.append(imp_sha_max)

            # register which feature is more imp than the max of shadows
            hits = np.where(imp_real > imp_sha_max)[0]
            hit_reg[hits] += 1

            # based on hit_reg we check if a feature is doing better than expected by chance
            dec_reg = self._do_tests(dec_reg, hit_reg, trial + 1)

        # we automatically apply R package's rough fix for tentative ones
        confirmed = np.where(dec_reg == 1)[0]
        tentative = np.where(dec_reg == 0)[0]

        # which tentative to keep
        tentative_median = np.median(imp_history[:, tentative], 0)
        tentative_confirmed = np.where(tentative_median > np.median(sha_max_history))[0]
        tentative = tentative[tentative_confirmed]

        # basic result variables
        self.n_features_ = confirmed.shape[0]
        self.support_ = np.zeros(n_feat, dtype=np.bool)
        self.support_[confirmed] = 1
        self.support_weak_ = np.zeros(n_feat, dtype=np.bool)
        self.support_weak_[tentative] = 1

        # ranking, confirmed variables are rank 1
        self.ranking_ = np.ones(n_feat, dtype=np.int)
        # tentative variables are rank 2
        self.ranking_[tentative] = 2
        # selected = confirmed and tentative
        selected = np.hstack((confirmed, tentative))
        # all rejected features are sorted by importance history
        not_selected = np.setdiff1d(np.arange(n_feat), selected)
        # large importance values should rank higher = lower ranks -> *(-1)
        imp_history_rejected = imp_history[:, not_selected] * -1

        # update rank for not_selected features
        if not_selected.shape[0] > 0:
            # calculate ranks in each iteration, then median of ranks across feats
            iter_ranks = self._nanrankdata(imp_history_rejected, axis=1)
            rank_medians = np.nanmedian(iter_ranks, axis=0)
            ranks = self._nanrankdata(rank_medians, axis=0)

            # set smallest rank to 3 if there are tentative feats
            if tentative.shape[0] > 0:
                ranks = ranks - np.min(ranks) + 3
            else:
                # and 2 otherwise
                ranks = ranks - np.min(ranks) + 2
            self.ranking_[not_selected] = ranks
        else:
            # all are selected, thus we set feature supports to True
            self.support_ = np.ones(n_feat, dtype=np.bool)

        self.importance_type_history_ = imp_history
        self.shadows_max_importance_history = sha_max_history

    def _estimator_fit(self, X_train, X_test, y_train, y_test, train_w=None, test_w=None, seed=0):
        # todo: oof estimator
        # check estimator_name
        self.estimator.set_params(random_state=seed)  # 要不要都行，因为数据随机了
        if 'lgb' in self.estimator_name:
            self.estimator.fit(X_train, y_train,
                               sample_weight=train_w,
                               eval_sample_weight=test_w,
                               eval_set=[(X_train, y_train), (X_test, y_test)],
                               eval_metric=None,
                               eval_names=('Train', 'Valid'),
                               verbose=self.verbose,
                               early_stopping_rounds=100)
        else:
            raise ValueError('默认支持lgb，其他模型待支持')

    def _add_shadows_get_imps(self, X, y, dec_reg, sample_weight=None, seed=0):
        # find features that are tentative still
        x_cur_ind = np.where(dec_reg >= 0)[0]
        x_cur = X[:, x_cur_ind]
        x_cur_w = x_cur.shape[1]

        # deep copy the matrix for the shadow matrix
        x_sha = x_cur.copy()

        # make sure there's at least 5 columns in the shadow matrix for
        while (x_sha.shape[1] < 5):
            x_sha = np.c_[x_sha, x_sha]

        # 打乱每列的值
        x_sha = np.apply_along_axis(check_random_state(seed).permutation, 0, x_sha)  # 打乱每列的值

        # get importance of the merged matrix
        imp = self._feature_importances(np.c_[x_cur, x_sha], y, sample_weight, seed + 1)  # (x_cur, x_sha)

        # separate importances of real and shadow features
        imp_sha = imp[x_cur_w:]

        imp_real = np.zeros(X.shape[1])
        imp_real[:] = np.nan
        imp_real[x_cur_ind] = imp[:x_cur_w]

        return imp_real, imp_sha

    def _feature_importances(self, X, y, sample_weight=None, seed=0):

        arrays = [X, y] if sample_weight is None else [X, y, sample_weight]
        args = train_test_split(*arrays, random_state=seed, stratify=y if check_classification(y) else None)
        self._estimator_fit(*args, seed=seed + 1)  # estimator fitted

        imp = get_imp(self.estimator, X, self.importance_type)

        return zscore(imp) if self.importance_type_normlize else imp

    def _nanrankdata(self, X, axis=1):
        """
        Replaces bottleneck's nanrankdata with scipy and numpy alternative.
        """
        ranks = sp.stats.mstats.rankdata(X, axis=axis)
        ranks[np.isnan(X)] = np.nan
        return ranks

    def _do_tests(self, dec_reg, hit_reg, _iter):
        active_features = np.where(dec_reg >= 0)[0]
        hits = hit_reg[active_features]
        # get uncorrected p values based on hit_reg
        to_accept_ps = sp.stats.binom.sf(hits - 1, _iter, .5).flatten()
        to_reject_ps = sp.stats.binom.cdf(hits, _iter, .5).flatten()

        if self.two_step:
            # two step multicor process
            # first we correct for testing several features in each round using FDR
            to_accept = self._fdrcorrection(to_accept_ps, alpha=self.alpha)[0]
            to_reject = self._fdrcorrection(to_reject_ps, alpha=self.alpha)[0]

            # second we correct for testing the same feature over and over again
            # using bonferroni
            to_accept2 = to_accept_ps <= self.alpha / float(_iter)
            to_reject2 = to_reject_ps <= self.alpha / float(_iter)

            # combine the two multi corrections, and get indexes
            to_accept *= to_accept2
            to_reject *= to_reject2
        else:
            # as in th original Boruta, we simply do bonferroni correction
            # with the total n_feat in each iteration
            to_accept = to_accept_ps <= self.alpha / float(len(dec_reg))
            to_reject = to_reject_ps <= self.alpha / float(len(dec_reg))

        # find features which are 0 and have been rejected or accepted
        to_accept = np.where((dec_reg[active_features] == 0) * to_accept)[0]
        to_reject = np.where((dec_reg[active_features] == 0) * to_reject)[0]

        # updating dec_reg
        dec_reg[active_features[to_accept]] = 1
        dec_reg[active_features[to_reject]] = -1
        return dec_reg

    def _fdrcorrection(self, pvals, alpha=0.05):
        """
        Benjamini/Hochberg p-value correction for false discovery rate, from
        statsmodels package. Included here for decoupling dependency on statsmodels.

        Parameters
        ----------
        pvals : array_like
            set of p-values of the individual tests.
        alpha : float
            error rate

        Returns
        -------
        rejected : array, bool
            True if a hypothesis is rejected, False if not
        pvalue-corrected : array
            pvalues adjusted for multiple hypothesis testing to limit FDR
        """
        pvals = np.asarray(pvals)
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
        nobs = len(pvals_sorted)
        ecdffactor = np.arange(1, nobs + 1) / float(nobs)

        reject = pvals_sorted <= ecdffactor * alpha
        if reject.any():
            rejectmax = max(np.nonzero(reject)[0])
            reject[:rejectmax] = True

        pvals_corrected_raw = pvals_sorted / ecdffactor
        pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
        pvals_corrected[pvals_corrected > 1] = 1
        # reorder p-values and rejection mask to original order of pvals
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_
