import torch

def computeError(x, y, result):
    """
    Computes the error introduced during a floating point addition using the TwoSum error-free transformation
    In infinite precision (associative maths) this function should return 0.
    WARNING: 
    - the order of the operations *matters*, do not change this operation in a way that would alter the order of operations
    - requires rounding to nearest (the default on modern processors) and assumes that floating points follow the IEEE-754 norm
    """
    y2 = result - x
    x2 = result - y2
    error_y = y - y2
    error_x = x - x2
    return error_x + error_y

def computeError_highPrecision(x, y, result):
    """
    Computes the error introduced during a floating point addition using the TwoSum error-free transformation
    In 64bits and more precision, this function returns 0.
    NOTE: 
    This function goes back to 64bits in order to compute the error. It is expensive but should only be used to validate `computeError`.
    """
    start_type = result.dtype
    if torch.finfo(start_type).bits >= 64:
        raise RuntimeError("compute_highPrecision: we have no precision higher than 64bits available.")
    end_type = torch.float64
    better_result = x.type(end_type) + y.type(end_type)
    error = (better_result - result).type(start_type)
    return error

class StochasticAdder:
    """
    Does stochastic additions between tensors of a given floating-point type.
    """
    def __init__(self, float_type, seed=None):
        """
        Set up the random number generator given the `torch.dtype` of the floats that will be summed.
        You can pass it a seed (single number), otherwise we will generate one using `torch.randint`
        """
        self.float_type = float_type
        # gets some information on the float type
        finfo = torch.finfo(float_type)
        self.bits = finfo.bits
        # needs to use tensor to insure that our number will be stored in the appropriate type
        self.float_max = torch.tensor([finfo.max], dtype=float_type)
        self.float_min = torch.tensor([finfo.min], dtype=float_type)
        # finds an integer type that is bit compatible with the float type
        if self.bits == 16:
            self.int_type = torch.int16
        elif self.bits == 32:
            self.int_type = torch.int32
        elif self.bits == 64:
            self.int_type = torch.int64 
        else:
            raise RuntimeError(f"StochasticAdder: '{float_type}' has an unsupported floating-point size ({self.bits}), please use a 16, 32 or 64 bits floating-point type.")
        self.max_int = torch.iinfo(self.int_type).max
        # generates a seed
        # making sure it is positive and odd
        if seed is None:
            self.seed = torch.randint(low=0, high=self.max_int, size=(1,), dtype=self.int_type) | 1
        else:
            seed = torch.tensor([seed], dtype=self.int_type)
            self.seed = torch.abs(seed) | 1
        # the shift that will be used to convert an integer type into a single bit
        self.shift = self.bits - 1

    def _pseudorandom_bool(self, x, y):
        """
        Takes two floating point inputs and returns a boolean generated pseudorandomly (deterministically) from those values
        uses Dietzfelbinger's multiply shift hash function
        see `High Speed Hashing for Integers and Strings` (https://arxiv.org/abs/1504.06804)
        """
        # xor inputs
        inputsHash = torch.bitwise_xor(x.view(self.int_type), y.view(self.int_type))
        # generates a hash and extracts a single bit
        # due to using signed integers, we will get a tensor of 0 and -1
        result = (self.seed * inputsHash) >> self.shift
        # returns result as a boolean
        return result.bool()

    def _pseudorandom_float(self, x, y):
        """
        Takes two floating point inputs and return a float in [0;1] generated pseudorandomly (deterministically) from those values
        uses a simplification of Dietzfelbinger's multiply shift hash function
        """
        # xor inputs
        inputsHash = torch.bitwise_xor(x.view(self.int_type), y.view(self.int_type))
        # generates a hash
        result_raw = (self.seed * inputsHash)
        # returns result as a float in [0;maxfloat]
        result = result_raw.abs().type(self.float_type) / self.max_int
        return result

    def _pseudorandom_bool_biased(self, x, y, addition, alternative_addition, error):
        """
        Takes two floating point inputs (x,y), their sum (addition), misrounded sum (alternative_addition) and the numerical error of their sum (error)
        returns a boolean generated pseudorandomly (deterministically) from those values
        the random number generator is biased according to the relative error of the addition.
        """
        random_float = self._pseudorandom_float(x,y)
        ulp = (alternative_addition - addition).abs()
        result = random_float * ulp > error.abs()
        return result

    def _misroundedAddition(self, x, y, result, error):
        """
        Given x + y = result (where result has been rounded to the closest representable number in the floating-point precision used)
        and error the numerical error of that addition
        returns result2 which is the closest floating point number on the opposite side of (x+y)
        """
        direction = torch.where(error > 0, self.float_max, self.float_min)
        return torch.nextafter(result, direction)

    def add(self, x, y):
        """
        Returns the sum of two tensors x and y pseudorandomly rounded to the nearest representable floating-point number up or down.
        Will round up 50% of the time and down the other 50%.
        """
        # insures the input types are correct
        self.check_type(x)
        self.check_type(y)
        # does the addition
        result = x + y
        error = computeError(x, y, result)
        alternativeResult = self._misroundedAddition(x, y, result, error)
        # picks the result to be returned
        useResult = self._pseudorandom_bool(x, y)
        return torch.where(useResult, result, alternativeResult)
    
    def add_biased(self, x, y):
        """
        Returns the sum of two tensors x and y pseudorandomly rounded to the nearest representable floating-point number up or down.
        The random number generator is biased according to the relative error of the addition.
        NOTE: this function is has better numerical properties than `add` but it is slower.
        """
        # insures the input types are correct
        self.check_type(x)
        self.check_type(y)
        # does the addition
        result = x + y
        error = computeError(x, y, result)
        alternativeResult = self._misroundedAddition(x, y, result, error)
        # picks the result to be returned
        useResult = self._pseudorandom_bool_biased(x, y, result, alternativeResult, error)
        return torch.where(useResult, result, alternativeResult)

    def check_type(self, data):
        """
        Errors-out if the data if not of the type that was passed to the constructor of this StochasticAdder.
        """
        if data.dtype != self.float_type:
            raise RuntimeError(f"StochasticAdder: got a tensor of type '{data.dtype}', expected type '{self.float_type}'")
