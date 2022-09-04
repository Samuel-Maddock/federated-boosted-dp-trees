from federated_gbdt.core.binning.feature_binning_param import FeatureBinningParam
from federated_gbdt.core.binning.base_binning import BaseBinning
from federated_gbdt.core.binning.quantile_summaries import SparseQuantileSummaries, QuantileSummaries
from federated_gbdt.core.baseline_constants import DEFAULT_RELATIVE_ERROR

import pandas as pd
import copy
import functools


class NoneType:
    def __eq__(self, obj):
        return isinstance(obj, NoneType)

def get_split_points(data_inst, is_sparse=False, bin_num=32,
                               binning_error=DEFAULT_RELATIVE_ERROR,
                               handle_missing_value=False):
    assert isinstance(data_inst, pd.DataFrame)
    param_obj = FeatureBinningParam(bin_num=bin_num, error=binning_error)
    if handle_missing_value:
        binning_obj = QuantileBinning(params=param_obj, abnormal_list=[NoneType()])
    else:
        binning_obj = QuantileBinning(params=param_obj)
    binning_obj.fit_split_points(data_inst, is_sparse)
    #print('split point results have been defined')
    return binning_obj.get_split_points_result_numpy()

def quantile_summary_factory(is_sparse, param_dict):
    if is_sparse:
        return SparseQuantileSummaries(**param_dict)
    else:
        return QuantileSummaries(**param_dict)


class QuantileBinning(BaseBinning):
    """
    After quantile binning, the numbers of elements in each binning are equal.

    The result of this algorithm has the following deterministic bound:
    If the data_instances has N elements and if we request the quantile at probability `p` up to error
    `err`, then the algorithm will return a sample `x` from the data so that the *exact* rank
    of `x` is close to (p * N).
    More precisely,

    {{{
      floor((p - 2 * err) * N) <= rank(x) <= ceil((p + 2 * err) * N)
    }}}

    This method implements a variation of the Greenwald-Khanna algorithm (with some speed
    optimizations).
    """

    def __init__(self, params: FeatureBinningParam, abnormal_list=None, allow_duplicate=False):
        super(QuantileBinning, self).__init__(params, abnormal_list)
        self.summary_dict = None
        self.allow_duplicate = allow_duplicate

    def fit_split_points(self, data_inst, is_sparse=False):
        """
        Apply the binning method

        Parameters
        ----------
        sparse_dataseries : Data series
            The input sparse vector

        Returns
        -------
        split_points : dict.
            Each value represent for the split points for a feature. The element in each row represent for
            the corresponding split point.
            e.g.
            split_points = {'x1': [0.1, 0.2, 0.3, 0.4 ...],    # The first feature
                            'x2': [1, 2, 3, 4, ...],           # The second feature
                            ...                         # Other features
                            }
        """
        if is_sparse:
            assert isinstance(data_inst, pd.Series)
            header = data_inst.iloc[0].feature_name
        else:
            assert isinstance(data_inst, pd.DataFrame)
            header = list(data_inst.columns)
        # if not isinstance(sparse_dataseries, pd.Series):
        #     raise TypeError('the input data should be data series')

        # LOGGER.debug("in _fit_split_point, cols_map: {}".format(self.bin_inner_param.bin_cols_map))

        self._default_setting(header)
        # self._init_cols(data_instances)
        percent_value = 1.0 / self.bin_num

        # calculate the split points
        percentile_rate = [i * percent_value for i in range(1, self.bin_num)]
        percentile_rate.append(1.0)

        self._fit_split_point(data_inst, is_sparse, percentile_rate)

        # self.fit_category_features(sparse_dataseries)  # can be ignored here
        return self.bin_results.all_split_points   # {fn: [fv_thresholds], ....}

    def get_split_points_result_numpy(self):
        return self.bin_results.get_split_points_array(self.bin_inner_param.transform_bin_names)

    @staticmethod
    def copy_merge(s1, s2):
        new_s1 = copy.deepcopy(s1)
        return new_s1.merge(s2)

    def _fit_split_point(self, data_inst, is_sparse, percentile_rate):
        if self.summary_dict is None:
            f = functools.partial(self.feature_summary,
                                  params=self.params,   # FeatureBinningParam(...)
                                  abnormal_list=self.abnormal_list,
                                  cols_dict=self.bin_inner_param.bin_cols_map,  # {bin_name: bin_idx, ...}
                                  header=self.header,
                                  is_sparse=is_sparse)
            summary_dict = f(data_inst=data_inst)
            summary_dict = dict(summary_dict)

            # LOGGER.debug(f"new summary_dict: {summary_dict}")
            total_count = len(data_inst)
            for _, summary_obj in summary_dict.items():
                summary_obj.set_total_count(total_count)

            self.summary_dict = summary_dict
        else:
            summary_dict = self.summary_dict

        for col_name, summary in summary_dict.items():
            split_point = []
            for percen_rate in percentile_rate:
                s_p = summary.query(percen_rate)
                if not self.allow_duplicate:
                    if s_p not in split_point:
                        split_point.append(s_p)
                else:
                    split_point.append(s_p)
            self.bin_results.put_col_split_points(col_name, split_point)

    @staticmethod
    def feature_summary(data_inst, params, cols_dict, abnormal_list, header, is_sparse):
        summary_dict = {}

        summary_param = {'compress_thres': params.compress_thres,
                         'head_size': params.head_size,
                         'error': params.error,
                         'abnormal_list': abnormal_list}

        for col_name, col_index in cols_dict.items():
            quantile_summaries = quantile_summary_factory(is_sparse=is_sparse, param_dict=summary_param)
            # quantile_summaries = SparseQuantileSummaries(**summary_param)
            summary_dict[col_name] = quantile_summaries

        if is_sparse:
            # pd.Series
            for sv in data_inst:
                data_generator = sv.get_all_data()
                for col_idx, col_value in data_generator:
                    col_name = header[col_idx]
                    if col_name not in cols_dict:
                        continue
                    summary = summary_dict[col_name]
                    summary.insert(col_value)
        else:
            # pd.Dataframe
            for _, inst in data_inst.iterrows():
                for col_name, summary in summary_dict.items():
                    col_index = cols_dict[col_name]
                    summary.insert(inst[col_index])

        result = []
        for features_name, summary_obj in summary_dict.items():
            summary_obj.compress()
            # result.append(((_, features_name), summary_obj))
            result.append((features_name, summary_obj))

        return result

    @staticmethod
    def _query_split_points(summary, percent_rates):
        split_point = []
        for percent_rate in percent_rates:
            s_p = summary.query(percent_rate)
            if s_p not in split_point:
                split_point.append(s_p)
        return split_point

    @staticmethod
    def approxi_quantile(data_instances, params, cols_dict, abnormal_list, header, is_sparse):
        """
        Calculates each quantile information

        Parameters
        ----------
        data_instances : DTable
            The input data

        cols_dict: dict
            Record key, value pairs where key is cols' name, and value is cols' index.

        params : FeatureBinningParam object,
                Parameters that user set.

        abnormal_list: list, default: None
            Specify which columns are abnormal so that will not static when traveling.

        header: list,
            Storing the header information.

        is_sparse: bool
            Specify whether data_instance is in sparse type

        Returns
        -------
        summary_dict: dict
            {'col_name1': summary1,
             'col_name2': summary2,
             ...
             }

        """

        summary_dict = {}

        summary_param = {'compress_thres': params.compress_thres,
                         'head_size': params.head_size,
                         'error': params.error,
                         'abnormal_list': abnormal_list}

        for col_name, col_index in cols_dict.items():
            quantile_summaries = quantile_summary_factory(is_sparse=is_sparse, param_dict=summary_param)
            summary_dict[col_name] = quantile_summaries

        QuantileBinning.insert_datas(data_instances, summary_dict, cols_dict, header, is_sparse)
        for _, summary_obj in summary_dict.items():
            summary_obj.compress()
        return summary_dict

    @staticmethod
    def insert_datas(data_instances, summary_dict, cols_dict, header, is_sparse):

        for iter_key, instant in data_instances:
            if not is_sparse:
                if type(instant).__name__ == 'Instance':
                    features = instant.features
                else:
                    features = instant
                for col_name, summary in summary_dict.items():
                    col_index = cols_dict[col_name]
                    summary.insert(features[col_index])
            else:
                data_generator = instant.features.get_all_data()
                for col_idx, col_value in data_generator:
                    col_name = header[col_idx]
                    summary = summary_dict[col_name]
                    summary.insert(col_value)

    @staticmethod
    def merge_summary_dict(s_dict1, s_dict2):
        if s_dict1 is None and s_dict2 is None:
            return None
        if s_dict1 is None:
            return s_dict2
        if s_dict2 is None:
            return s_dict1

        s_dict1 = copy.deepcopy(s_dict1)
        s_dict2 = copy.deepcopy(s_dict2)

        new_dict = {}
        for col_name, summary1 in s_dict1.items():
            summary2 = s_dict2.get(col_name)
            summary1.merge(summary2)
            new_dict[col_name] = summary1
        return new_dict

    def query_quantile_point(self, query_points, col_names=None):

        if self.summary_dict is None:
            raise RuntimeError("Bin object should be fit before query quantile points")

        if col_names is None:
            col_names = self.bin_inner_param.bin_names

        summary_dict = self.summary_dict

        if isinstance(query_points, (int, float)):
            query_dict = {}
            for col_name in col_names:
                query_dict[col_name] = query_points
        elif isinstance(query_points, dict):
            query_dict = query_points
        else:
            raise ValueError("query_points has wrong type, should be a float, int or dict")

        result = {}
        for col_name, query_point in query_dict.items():
            summary = summary_dict[col_name]
            result[col_name] = summary.query(query_point)
        return result


# class QuantileBinningTool(QuantileBinning):
#     """
#     Use for quantile binning data directly.
#     """
#
#     def __init__(self, bin_nums=consts.G_BIN_NUM, param_obj: FeatureBinningParam = None,
#                  abnormal_list=None, allow_duplicate=False):
#         if param_obj is None:
#             param_obj = FeatureBinningParam(bin_num=bin_nums)
#         super().__init__(params=param_obj, abnormal_list=abnormal_list, allow_duplicate=allow_duplicate)