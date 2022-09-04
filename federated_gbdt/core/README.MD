### Structue of federated_xgboost.core:

* ``binning`` - Binning/Quantile Sketching implementation from FATE
* ``dp_multiq`` - Google implementation of multi-dimensional DP quantiles (see https://arxiv.org/abs/2102.08244)
* ``moments_accountant`` - RDP moments accountant implementation (of the sampled gaussian mechanism (SGM)) from tensorflow.privacy
* ``pure_ldp`` - Various implementations of LDP protocols
* ``baseline_constants.py`` - Leftover code from FEVERLESS, contains parameters for quantile sketching