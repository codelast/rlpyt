"""
一个用来表示"空间"的类，例如observation space，action space。
"""
import numpy as np

from rlpyt.spaces.base import Space


class IntBox(Space):
    """A box in J^n, with specificiable bound and dtype."""

    def __init__(self, low, high, shape=None, dtype="int32", null_value=None):
        """
        low and high are scalars, applied across all dimensions of shape.
        """
        assert np.isscalar(low) and np.isscalar(high)
        self.low = low
        self.high = high
        self.shape = shape if shape is not None else ()  # np.ndarray sample
        self.dtype = np.dtype(dtype)
        assert np.issubdtype(self.dtype, np.integer)  # 保证输入的数据类型必须为整型
        null_value = low if null_value is None else null_value
        assert null_value >= low and null_value < high
        self._null_value = null_value

    def sample(self):
        """
        随机选择action space内的一个index(并非得到实际的action)。
        np.random.randint()函数的size参数表示输出数据的shape，当size为()或None时，表示输出一个标量值。
        更详细的说明可参考DecorrelatingStartCollector.start_envs()函数里的注释。
        """
        return np.random.randint(low=self.low, high=self.high, size=self.shape, dtype=self.dtype)

    def null_value(self):
        """
        把self.shape这个形状的矩阵里的所有元素set成null_value定义的值。
        """
        null = np.zeros(self.shape, dtype=self.dtype)
        # _null_value肯定不为None，因为__init__的时候已经保证了这一点。在没有传入null_value参数的时候，_null_value是一个标量值
        if self._null_value is not None:
            try:
                null[:] = self._null_value  # null这个矩阵的所有元素被set成同样的标量值
            except IndexError:
                null.fill(self._null_value)
        return null

    @property
    def bounds(self):
        """
        获取space的边界。
        """
        return self.low, self.high

    @property
    def n(self):
        """
        获取space的范围。
        """
        return self.high - self.low

    def __repr__(self):
        return f"IntBox({self.low}-{self.high - 1} shape={self.shape})"
