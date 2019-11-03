"""
该类的作用：生成随机数种子，以及设置随机数种子。
说明：在神经网络中，参数默认是进行随机初始化的。不同的初始化参数往往会导致不同的结果，当得到比较好的结果时我们通常希望这个结果是可以复现的，在PyTorch
中，通过设置随机数种子也可以达到这个目的。随机数种子seed确定时，模型的训练结果将始终保持一致。
"""

import time

import numpy as np

from rlpyt.utils.logging.console import colorize

seed_ = None


def set_seed(seed):
    seed %= 4294967294
    global seed_
    seed_ = seed
    import random
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为[当前]GPU设置随机种子(不是所有GPU)
    print(colorize(f"using seed {seed}", "green"))


def get_seed():
    return seed_


def make_seed() -> float:
    """
    Returns a random number between [0, 10000], using timing jitter.

    This has a white noise spectrum and gives unique values for multiple
    simultaneous processes...some simpler attempts did not achieve that, but
    there's probably a better way.
    """
    d: int = 10000
    t: float = time.time()
    sub1 = int(t * d) % d
    sub2 = int(t * d ** 2) % d
    s: float = 1e-3
    s_inv: float = 1. / s
    time.sleep(s * sub2 / d)
    t2: float = time.time()
    t2 = t2 - int(t2)
    t2 = int(t2 * d * s_inv) % d
    time.sleep(s * sub1 / d)
    t3: float = time.time()
    t3 = t3 - int(t3)
    t3 = int(t3 * d * s_inv * 10) % d
    return (t3 - t2) % d
