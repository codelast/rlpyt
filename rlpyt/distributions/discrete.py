
import torch

from rlpyt.utils.tensor import to_onehot, from_onehot


class DiscreteMixin:

    def __init__(self, dim, dtype=torch.long, onehot_dtype=torch.float):
        self._dim = dim
        self.dtype = dtype
        self.onehot_dtype = onehot_dtype

    @property
    def dim(self):
        return self._dim

    def to_onehot(self, indexes, dtype=None):
        """
        参数里使用了 or 表达式，使得当 dtype=None 时，表达式的值为 self.onehot_dtype，当 dtype 不为 None 时，表达式的值为传入的dtype
        """

        return to_onehot(indexes, self._dim, dtype=dtype or self.onehot_dtype)

    def from_onehot(self, onehot, dtype=None):
        """
        参数里使用了 or 表达式，使得当 dtype=None 时，表达式的值为 self.dtype，当 dtype 不为 None 时，表达式的值为传入的dtype
        """

        return from_onehot(onehot, dtpye=dtype or self.dtype)
