import torch
from stochastorch import StochasticAdder, computeError, computeError_highPrecision

# problem description
size = 10
dtype = torch.bfloat16

# addition
x = torch.rand(size, dtype=dtype)
y = torch.rand(size, dtype=dtype)
result = x + y

# tests the error computation
error = computeError(x, y, result)
error_high = computeError_highPrecision(x, y, result)
print(f"x + y: {result}")
print(f"error: {error}")
print(f"error high precision: {error_high}")
assert(torch.all(error == error_high))

# take a look at the alternative result
adder = StochasticAdder(dtype)
alternativeResult = adder._misroundedAddition(result, error)
print(f"alternative: {alternativeResult})")
print(f"difference: {result - alternativeResult}")

# runs the stochastic addition
result_sto = adder.add(x,y)
print(f"stochastic addition: {result_sto}")
difference = result - result_sto
print(f"difference: {difference}")

# runs the biased stochastic addition
result_bia = adder.add_biased(x,y)
print(f"biased stochastic addition: {result_bia}")
difference = result - result_bia
print(f"difference: {difference}")
