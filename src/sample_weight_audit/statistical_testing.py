from scipy.stats import kstest, mannwhitneyu


def run_statistical_test(pred_ref, pred, test_name="kstest"):
    if test_name == "kstest":
        test_result = kstest(pred, pred_ref)
    elif test_name == "mannwhitneyu":
        test_result = mannwhitneyu(pred, pred_ref)
    else:
        raise ValueError(
            f"Test {test_name} is not supported, please use 'kstest' or 'mannwhitneyu'"
        )
    return test_result
