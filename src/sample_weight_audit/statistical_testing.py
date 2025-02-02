from scipy.stats import kstest, mannwhitneyu, ttest_ind


def run_1d_test(pred_ref, pred, test_name="kstest"):
    if test_name == "kstest":
        test_result = kstest(pred, pred_ref)
    elif test_name == "ttest":
        test_result = ttest_ind(pred, pred_ref, equal_var=True)
    elif test_name == "mannwhitneyu":
        test_result = mannwhitneyu(pred, pred_ref)
    else:
        raise ValueError(
            f"Test {test_name} is not supported, please use 'kstest', 'ttest' or 'mannwhitneyu'"
        )
    return test_result
