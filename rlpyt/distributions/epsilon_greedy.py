import torch

from rlpyt.distributions.base import Distribution
from rlpyt.distributions.discrete import DiscreteMixin


class EpsilonGreedy(DiscreteMixin, Distribution):
    """Input can be shaped [T,B,Q] or [B,Q], and vector epsilon of length
    B will apply across the Batch dimension (same epsilon for all T)."""

    def __init__(self, epsilon=1, **kwargs):
        super().__init__(**kwargs)
        self._epsilon = epsilon  # 应该是一个(0, 1)之间的较小的数

    def sample(self, q):
        """
        按Epsilon-Greedy算法对输入数据(矩阵)做采样。这个函数只有短短几行，但是设计很巧妙。它想做的事情是：找出输入矩阵某个维度上的最大值，然
        后按一定的机率(即epsilon)“不选取”那个值最大的index，最终得到一个代表“有少量随机性”的最大值index矩阵。

        举一个例子来说明这个函数的逻辑：假设 self._epsilon = 0.3，且输入的 q 为 3x4 的一个矩阵，即
        [[-0.2187, -0.2758,  0.4933,  1.0700],
        [ 0.2689,  3.5079,  1.5640,  1.1730],
        [-0.6858,  0.2571,  1.0396,  0.6344]]
        则：
        arg_select 值为 [3, 1, 2]，这是因为，对输入矩阵来说，第一行最大的值是 1.0700，其index为3；第二行最大的值是 3.5079，其index为0；
        第三行最大的值是 1.0396，其index为0，因此拼起来就是 [3, 0, 0]。

        mask 值为[True, False, True]，这是因为，此时 torch.rand(arg_select.shape)得到的一个随机矩阵是[0.2983, 0.4749, 0.2926]
        (由于是随机的，因此不是每次都是这个结果，这里仅拿某一次运行的结果作为例子来陈述)，这个随机矩阵的3个数，分别和 self._epsilon 比小，得
        到的结果就是 [True, False, True]。

        mask.sum() 的值为 2，因为这等同于执行 torch.sum(mask)，即计算 mask 这个 Tensor 上的所有元素的和，对元素为 bool 类型的情况，
        True为1，False为0，因此结果为2。
        q.shape[-1] 的值为 4，因为 shape 为(3, 4)，因此 shape[-1] 就是最后一个值，即 4。
        因此 arg_rand 这一句执行的语句就是：torch.rand(low=0, high=4, size=(2, ))，即在 [0, 4] 间随机取两个整数，结果为 [2, 3]

        arg_select[mask] = arg_rand 这句在执行之前，arg_select为[3, 1, 2]，mask为[True, False,  True]，arg_rand为[2, 3]，对
        mask里为True的两个位置，找到arg_select里的对应位置，替换成arg_rand里的值，就是最后的结果：[2, 1, 3]。


        :param q: 一个torch.Tensor类型的对象。
        :return: torch.Tensor类型的对象。
        """
        arg_select = torch.argmax(q, dim=-1)  # 返回指定的维度(dim)上，值最大的那个数的index。-1表示最后一个维度，即等同于dim=1的效果
        mask = torch.rand(arg_select.shape) < self._epsilon  # 得到一个bool的矩阵，标识了torch.rand生成的随机数组里的每个元素是比self._epsilon大还是小
        """
        torch.randint()返回均匀分布的[low,high]之间的整数随机值，mask.sum()得到bool矩阵中True元素的个数(假设为x)，因此得到的arg_rand是
        x个[low,high]之间的随机数。例如 print(torch.randint(0, 20, (6, ))) 的输出可能是：tensor([14,  4,  7, 17, 16,  3])
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
