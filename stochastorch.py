"""
Stochastically rounded operations between Pytorch tensors.

This code was written by Nestor Demeure and is licensed under the Apache 2.0 license.
You can find an up-to-date source and full description here: https://github.com/nestordemeure/stochastorch
"""
import torch

#----------------------------------------------------------------------------------------
# BUILDING BLOCKS

def _computeError(x, y, result):
    """
    Computes the error introduced during a floating point addition (x+y=result) using the TwoSum error-free transformation.
    In infinite precision (associative maths) this function should return 0.

    WARNING: 
    - the order of the operations *matters*, do not change this operation in a way that would alter the order of operations
    - requires rounding to nearest (the default on modern processors) and assumes that floating points follow the IEEE-754 norm 
      (but, it has been tested with alternative types such as bfloat16)
    """
    # NOTE: computing this quantity via a cast to higher precision would be faster for low precisions
    y2 = result - x
    x2 = result - y2
    error_y = y - y2
    error_x = x - x2
    return error_x + error_y

def _misroundResult(result, error):
    """
    Given the result of a floating point operation and the numerical error introduced during that operation
    returns the floating point number on the other side of the interval containing the analytical result of the operation.

    NOTE: the output of this function will be of the type of result, the type of error does not matter.
    """
    finfo = torch.finfo(result.dtype)
    float_max = torch.tensor([finfo.max], dtype=result.dtype, device=result.device)
    float_min = torch.tensor([finfo.min], dtype=result.dtype, device=result.device)
    direction = torch.where(error > 0, float_max, float_min)
    return torch.nextafter(result, direction)

def _pseudorandomBool(result, alternative_result, error, is_biased=True):
    """
    Takes  the result of a floating point operation, 
    the floating point number on the other side of the interval containing the analytical result of the operation
    and the numerical error introduced during that operation
    returns a randomly generated boolean.

    If is_biased is True, the random number generator is biased according to the relative error of the operation
    else, it will round up 50% of the time and down the other 50%.
    """
    if is_biased:
        ulp = (alternative_result - result).abs()
        random_float = torch.rand(size=ulp.shape, device=ulp.device, dtype=ulp.dtype)
        result = random_float * ulp > error.abs()
    else:
        # NOTE: we do not deal with the error==0 case as it is too uncommon to bias the results significantly
        result = torch.rand(size=result.shape, dtype=result.dtype, device=result.device) < 0.5
    return result

#----------------------------------------------------------------------------------------
# OPERATIONS

def add_highprecision(x, y_high_precision, is_biased=True):
    """
    Returns the sum of two tensors x and y_high_precision pseudorandomly rounded up or down to the nearest representable floating-point number.
    y_high_precision is assumed to be in a floating-point precision strictly higher than x.

    If is_biased is True, the random number generator is biased according to the relative error of the addition
    else, it will round up 50% of the time and down the other 50%.
    """
    # insures the input types are properly sized
    dtype_low_precision = x.dtype
    dtype_high_precision = y_high_precision.dtype
    assert(torch.finfo(dtype_low_precision).bits < torch.finfo(dtype_high_precision).bits)
    # performs the addition
    result_high_precision = x.to(dtype_high_precision) + y_high_precision
    result = result_high_precision.to(dtype_low_precision)
    # computes the numerical error
    result_rounded = result.to(dtype_high_precision)
    error = result_high_precision - result_rounded
    # picks the result to be returned
    alternativeResult = _misroundResult(result, error)
    useResult = _pseudorandomBool(result, alternativeResult, error, is_biased)
    return torch.where(useResult, result, alternativeResult)

def add(x, y, is_biased=True):
    """
    Returns the sum of two tensors x and y pseudorandomly rounded up or down to the nearest representable floating-point number.

    This function will delegate to `add_highprecision` if y is higher precision than x.
    It will then return a result of the sane precision as x.

    If is_biased is True, the random number generator is biased according to the relative error of the addition
    else, it will round up 50% of the time and down the other 50%.
    """
    # use a specialized function if y is higher precision than x
    if (torch.finfo(y.dtype).bits > torch.finfo(x.dtype).bits):
        return add_highprecision(x, y, is_biased)
    # otherwise insures the input types are coherent
    assert(x.dtype == y.dtype)
    # does the addition
    result = x + y
    error = _computeError(x, y, result)
    # picks the result to be returned
    alternativeResult = _misroundResult(result, error)
    useResult = _pseudorandomBool(result, alternativeResult, error, is_biased)
    return torch.where(useResult, result, alternativeResult)

def addcdiv(x, tensor1, tensor2, value=1, is_biased=True):
    """
    Computes x + (tensor1/tensor2)*value pseudorandomly rounding the addition up or down to the nearest representable floating-point number.
    In 16-bits or less, this operation is *significantly* more precise than doing the operations separetely.

    If is_biased is True, the random number generator is biased according to the relative error of the addition
    else, it will round up 50% of the time and down the other 50%.
    """
    # insures the input types are coherent
    assert(tensor1.dtype == tensor2.dtype)
    assert(x.dtype == tensor1.dtype)
    # does the addcdiv
    # we NEED the inplace version as it uses higher precision internally
    result = x.clone().detach()
    result.addcdiv_(tensor1, tensor2, value=value)
    # computes the numerical error
    # NOTE we might want to compute y in high precision *then* compute result for improved performance
    y = torch.zeros_like(x, dtype=x.dtype, device=x.device)
    y.addcdiv_(tensor1, tensor2, value=value)
    error = _computeError(x, y, result)
    # picks the result to be returned
    alternativeResult = _misroundResult(result, error)
    useResult = _pseudorandomBool(result, alternativeResult, error, is_biased)
    return torch.where(useResult, result, alternativeResult)
