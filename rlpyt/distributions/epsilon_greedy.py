import torch

from rlpyt.distributions.base import Distribution
from rlpyt.distributions.discrete import DiscreteMixin


class EpsilonGreedy(DiscreteMixin, Distribution):
    """Input can be shaped [T,B,Q] or [B,Q], and vector epsilon of length
    B will apply across the Batch dimension (same epsilon for all T)."""

    def __init__(self, epsilon=1, **kwargs):
        super().__init__(**kwargs)
        self._epsilon = epsilon

    def sample(self, q):
        arg_select = torch.argmax(q, dim=-1)  # 按指定的维度(dim)返回最大元素的index
        mask = torch.rand(arg_select.shape) < self._epsilon  # 得到一个bool的矩阵，标识了torch.rand生成的随机数组里的每个元素是比self._epsilon大还是小
        """
        torch.randint()返回均匀分布的[low,high]之间的整数随机值，mask.sum()得到bool矩阵中True元素的个数(假设为x)，因此得到的arg_rand是
        x个[low,high]之间的随机数。例如 print(torch.randint(0, 20, (6, ))) 的输出可能是：tensor([14,  4,  7, 17, 16,  3])
        # TODO: q 是一个 torch.nn.Module 的对象，其没有shape属性，为什么这里会没有问题？
        """
        arg_rand = torch.randint(low=0, high=q.shape[-1], size=(mask.sum(),))
        arg_select[mask] = arg_rand
        return arg_select

    @property
    def epsilon(self):
        return self._epsilon

    def set_epsilon(self, epsilon):
        self._epsilon = epsilon


class CategoricalEpsilonGreedy(EpsilonGreedy):
    """Input p to be shaped [T,B,A,P] or [B,A,P], A: number of actions,
    P: number of atoms.  Input z is domain of atom-values, shaped [P]."""

    def __init__(self, z=None, **kwargs):
        super().__init__(**kwargs)
        self.z = z

    def sample(self, p, z=None):
        q = torch.tensordot(p, z or self.z, dims=1)
        return super().sample(q)

    def set_z(self, z):
        self.z = z
