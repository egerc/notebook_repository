# Fix ts
2026-06-17 17:25:22,054 INFO Fitting model PCA on dataset Mouse Intestine
WARNING: It seems you use rank_genes_groups on the raw count data. Please logarithmize your data before calling rank_genes_groups.
Traceback (most recent call last):
  File "/home/gruengroup/christian/Projects/notebook_repository/notebooks/2026-06-15T09-07-20Z/experiment.py", line 577, in <module>
    main()
  File "/home/gruengroup/christian/Projects/notebook_repository/notebooks/2026-06-15T09-07-20Z/experiment.py", line 536, in main
    for (celltype, predictor_protocol), sample in product(
                                                  ^^^^^^^^
  File "/home/gruengroup/christian/Projects/notebook_repository/notebooks/2026-06-15T09-07-20Z/experiment.py", line 532, in <genexpr>
    predictor.fit(celltype.reference_counts_matrix),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/gruengroup/christian/Projects/notebook_repository/notebooks/2026-06-15T09-07-20Z/.venv/lib64/python3.12/site-packages/nico2_lib/predictors/_nmf/_nmf_pred.py", line 235, in fit
    w_reference = model.fit_transform(
                  ^^^^^^^^^^^^^^^^^^^^
  File "/home/gruengroup/christian/Projects/notebook_repository/notebooks/2026-06-15T09-07-20Z/.venv/lib64/python3.12/site-packages/sklearn/utils/_set_output.py", line 319, in wrapped
    data_to_wrap = f(self, X, *args, **kwargs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/gruengroup/christian/Projects/notebook_repository/notebooks/2026-06-15T09-07-20Z/.venv/lib64/python3.12/site-packages/sklearn/base.py", line 1403, in wrapper
    return fit_method(estimator, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/gruengroup/christian/Projects/notebook_repository/notebooks/2026-06-15T09-07-20Z/.venv/lib64/python3.12/site-packages/sklearn/decomposition/_nmf.py", line 1621, in fit_transform
    W, H, n_iter = self._fit_transform(X, W=W, H=H)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/gruengroup/christian/Projects/notebook_repository/notebooks/2026-06-15T09-07-20Z/.venv/lib64/python3.12/site-packages/sklearn/decomposition/_nmf.py", line 1683, in _fit_transform
    W, H = self._check_w_h(X, W, H, update_H)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/gruengroup/christian/Projects/notebook_repository/notebooks/2026-06-15T09-07-20Z/.venv/lib64/python3.12/site-packages/sklearn/decomposition/_nmf.py", line 1200, in _check_w_h
    raise TypeError(
TypeError: H and W should have the same dtype as X. Got H.dtype = float64 and W.dtype = float64.
