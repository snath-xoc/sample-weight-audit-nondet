Auditing tool for sample weight equivalence over all regressors, classifiers and transformers within scikit-learn. 
Testing is done under the non-deterministic case where we can only compare estimator results in expectation and not absolute value.
The sample weight equivalence assumes equivalence in estimator results between weighted and repeated samples (i.e., a weight of n should be equivalent to n-times repitition)
