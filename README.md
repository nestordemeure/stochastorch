# StochasTorch: a Pytorch implementation of stochastic addition

This repository contains a Pytorch software-based implementation of [stochastic rounding](https://nhigham.com/2020/07/07/what-is-stochastic-rounding/) addition.

When encoding the weights of a neural network in low precision (such as `bfloat16`), one runs into stagnation problems: updates end up being too small relative to the numbers the precision of the encoding.
This leads to weights becoming stuck and the model's accuracy being significantly reduced.

Stochastic arithmetic lets you perform the addition in such a way that the weights have a non-zero probability of being modified anyway.
This avoids the stagnation problem (see [figure 4 of "Revisiting BFloat16 Training"](https://arxiv.org/abs/2010.06192)) without increasing the memory usage (as might happen if one were using a [compensated summation](https://github.com/nestordemeure/pairArithmetic) to solve the problem).

The downside is that software-based stochastic arithmetic is significantly slower than a normal floating-point addition.
It is thus viable for the weight update but would not be appropriate in a hot loop.

## Usage

This repository gives you a `StochasticAdder` type. Once you initialize it with the floating point precision you will be using (this code should be valid for all floating-point types used in Pytorch) and, optionally, a seed (to ensure reproducibility), you can use its `add` or `add_biased` function to perform stochastic additions on Pytorch tensors:

```python
import torch
from stochastorch import StochasticAdder

# struct that encapsulates the addition logic
dtype = torch.bfloat16
adder = StochasticAdder(dtype)

# numbers that will be added
size = 10
x = torch.rand(size, dtype=dtype)
y = torch.rand(size, dtype=dtype)

# deterministic addition
result_det = x + y
print(f"deterministic addition: {result_det}")

# stochastic addition
result_sto = adder.add(x,y)
print(f"stochastic addition: {result_sto}")
difference = result_det - result_sto
print(f"difference: {difference}")

# biased stochastic addition
result_bia = adder.add_biased(x,y)
print(f"biased stochastic addition: {result_bia}")
difference = result_det - result_bia
print(f"difference: {difference}")
```

`add_biased` flips the rounding mode with a probability proportional to the ratio of the error and the distance between the result and the alternative result.
This makes the addition associative *on average*.

`add` has a 50% probability of flipping the rounding mode.
This is faster but, will change the weights more often.

## Implementation details

We use `TwoSum` to measure the numerical error done by an addition, our tests show that it behaves as needed on `bfloat16` (some edge cases might be invalid, leading to an inexact computation of the numerical error but, it is reliable enough for our purpose).
This and the `nextafter` function let us emulate various rounding modes in software (this is inspired by [Verrou's backend](https://github.com/edf-hpc/verrou)).

The random number generation is done using a hashing function ([Dietzfelbinger's multiply shift hash function](https://arxiv.org/abs/1504.06804)) following [the ideas of Salmons](http://www.thesalmons.org/john/random123/papers/random123sc11.pdf) as a way to avoid paying the price of a random number generator call while ensuring that our additions are deterministic (given the same seed and numbers, we will always round in the same direction).

## Potential improvements:

- One could reduce the memory usage of the operations by using more in-place operations,
- one could improve the performance of this code by implementing it as a C++/CUDA kernel.

Do not hesitate to submit an issue or a pull request if you need added functionalities for your needs!

## Crediting this work

Please use this reference if you use Stochastorch within a published work:

```bibtex
@misc{StochasTorch,
  author = {Nestor, Demeure},
  title = {StochasTorch: a Pytorch implementation of stochastic addition},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nestordemeure/stochastorch}}
}
```

You will find a JAX implementation called Jochastic [here](https://github.com/nestordemeure/jochastic).
