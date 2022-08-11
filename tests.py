import torch
import stochastorch

# problem description
size = 10
dtype = torch.bfloat16

# addition
x = torch.rand(size, dtype=dtype)
y = torch.rand(size, dtype=dtype)
result = x + y

# check the error computation
error = stochastorch._computeError(x, y, result)
print(f"x + y: {result}")
print(f"error: {error}")

# take a look at the alternative result
alternativeResult = stochastorch._misroundResult(result, error)
print(f"alternative: {alternativeResult})")
print(f"difference: {result - alternativeResult}")

# runs the stochastic addition
result_sto = stochastorch.add(x, y, is_biased=False)
print(f"stochastic addition: {result_sto}")
difference = result - result_sto
print(f"difference: {difference}")

# runs the biased stochastic addition
result_bia = stochastorch.add(x, y)
print(f"biased stochastic addition: {result_bia}")
difference = result - result_bia
print(f"difference: {difference}")

# runs a multiprecision stochastic addition
y_high_precision = y.to(torch.float64)
result_sto = stochastorch.add(x, y_high_precision)
print(f"stochastic high-precision addition: {result_sto}")
difference = result - result_sto
print(f"difference: {difference}")

# runs addcdiv
result_det = torch.addcdiv(x, x, y)
result_sto = stochastorch.addcdiv(x, x, y)
print(f"deterministic addcdiv: {result_det}")
print(f"stochastic addcdiv: {result_sto}")
difference = result_det - result_sto
print(f"difference: {difference}")
