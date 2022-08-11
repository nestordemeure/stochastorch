# StochasTorch: stochastically rounded operations between Pytorch tensors.

This repository contains a Pytorch software-based implementation of some [stochastically rounded operations](https://nhigham.com/2020/07/07/what-is-stochastic-rounding/).

When encoding the weights of a neural network in low precision (such as `bfloat16`), one runs into stagnation problems: updates end up being too small relative to the numbers the precision of the encoding.
This leads to weights becoming stuck and the model's accuracy being significantly reduced.

Stochastic arithmetic lets you perform the addition in such a way that the weights have a non-zero probability of being modified anyway.
This avoids the stagnation problem (see [figure 4 of "Revisiting BFloat16 Training"](https://arxiv.org/abs/2010.06192)) without increasing the memory usage (as might happen if one were using a [compensated summation](https://github.com/nestordemeure/pairArithmetic) to solve the problem).

The downside is that software-based stochastic arithmetic is significantly slower than a normal floating-point addition.
It is thus viable for things like the weight update but would not be appropriate in a hot loop.

## Usage

This repository introduces the `add` (`x+y`), `add_highprecision` and `addcdiv` (`x + epsilon*t1/t2`) operations.
They act similarly to their PyTorch counterparts but round the result up or down randomly:

```python
import torch
import stochastorch

# problem definition
size = 10
dtype = torch.bfloat16
x = torch.rand(size, dtype=dtype)
y = torch.rand(size, dtype=dtype)

# deterministic addition
result_det = x + y
print(f"deterministic addition: {result_det}")

# stochastic addition
result_sto = stochastorch.add(x,y)
print(f"stochastic addition: {result_sto}")
difference = result_det - result_sto
print(f"difference: {difference}")

# stochastic addcdiv 
# result = x + epsilon*t1/t2
t1 = torch.rand(size, dtype=dtype)
t2 = torch.rand(size, dtype=dtype)
epsilon = -0.1
result_det = torch.addcdiv(x, t1, t2, value=epsilon)
print(f"deterministic addcdiv: {result_det}")

# stochastic addcdiv
result_sto = stochastorch.addcdiv(x, t1, t2, value=epsilon)
print(f"stochastic addcdiv: {result_bia}")
difference = result_det - result_sto
print(f"difference: {difference}")
```

Both functions take an optional `is_biased` boolean parameter.
If is_biased is True, the random number generator is biased according to the relative error of the operation
else, it will round up half of the time on average.

When using low precision (16 bits floating-point arithmetic or less), we *strongly* recommend using the `stochastorch.addcdiv` function when possible as it is significantly more accurate (note that Pytorch also [increase the precision locally to 32 bits](https://github.com/pytorch/pytorch/blob/12382f0a38f8199bc74aee701465e847f368e6de/aten/src/ATen/native/cuda/PointwiseOpsKernel.cu?fbclid=IwAR0SdS6mVAGN0TB_TAdKt0WVWWjxiBkmP6Inj9lYH8oB68wjsbQzinlH-xY#L92) when computing `addcdiv`).

Otherwise, it is often beneficial to use higher precision locally *then* cast down to 16 bits at summing / storage time.
`add` deals with it automatically when its second input is higher precision than the first.

## Implementation details

We use `TwoSum` to measure the numerical error done by an addition, our tests show that it behaves as needed on `bfloat16` (some edge cases might be invalid, leading to an inexact computation of the numerical error but, it is reliable enough for our purpose) and higher floating-point precisions.

This and the [`nextafter`](https://pytorch.org/docs/stable/generated/torch.nextafter.html) function let us emulate various rounding modes in software (this is inspired by [Verrou's backend](https://github.com/edf-hpc/verrou)).

## Potential improvements:

- one could implement more operations,
- one could reduce the memory usage of the operations by using more in-place operations,
- one could improve the performance of this code by implementing it as a C++/CUDA kernel.

Do not hesitate to submit an issue or a pull request if you need added functionalities for your needs!

## Crediting this work

Please use this reference if you use Stochastorch within a published work:

```bibtex
@misc{StochasTorch,
  author = {Nestor, Demeure},
  title = {StochasTorch: stochastically rounded operations between Pytorch tensors.},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/nestordemeure/stochastorch}}
}
```

You will find a JAX implementation called Jochastic [here](https://github.com/nestordemeure/jochastic).
