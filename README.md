Auditing tool for sample weight equivalence over all regressors, classifiers and transformers within scikit-learn.
Testing is done under the non-deterministic case where we can only compare estimator results in expectation and not absolute value.
The sample weight equivalence assumes equivalence in estimator results between weighted and repeated samples (i.e., a weight of n should be equivalent to n-times repetition)

Getting started:

```
pip install -e .
```

Then run the notebook(s) in the `reports/` folder.

# Sample Weight Documentation

Property Checks
---

If we have a weight-aware function $f$, then certain modifications to our weights $w$ and sample set $X$ can be made, such that the outcome of $f$ does not change:

* 0-weight invariance: Omitting samples and weighing samples as 0 should have the same outcome. For example, for a given input with $R$ denoting rows with 0 weight values.
    
  $f(X,w) = f(X_{i\notin R},w_{i\notin R})$
  
* No-weight invariance: Having no weights is the same having unit weights.

  $f(X) = f(X,1)$
  
* Combining data: if a subset with identical records exist, you can combine them to a single value with a weight equal to the subset's cardinality. For example, take subset $J$ which consists of values $x$ with cardinality $n$.

      $f(X,w) = f([X_{i\notin J},x],[w_{i\notin J},n])$
  
* Splitting data: The inverse of combining, if we split a weighted sample up and apportion the weights, we should arrive at the same outcome. For example if we take $\sum w'=w_j$ for a given row $j$, then:

      $f(X,w) = f([X_{i\notin j},x_j],[w_{i\notin j},w'])$

* Scale-free weights: Rescaling weights by a factor $f$ results in the same outcome.

        $f(X,w) = f(X,w\cdot f)$


Types of weighting
---
Across literature there seem to be 3 types of weighting:

* Precision weighting: This considers the weights to represent the precision of a sample
* Frequency weighting
* Sampling weighting


