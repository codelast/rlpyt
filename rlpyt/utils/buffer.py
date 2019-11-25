
import numpy as np
import multiprocessing as mp
import ctypes
import torch

from rlpyt.utils.collections import namedarraytuple_like
# from rlpyt.utils.misc import put


def buffer_from_example(example, leading_dims, share_memory=False):
    if example is None:
        return
    try:
        buffer_type = namedarraytuple_like(example)
    except TypeError:  # example was not a namedtuple or namedarraytuple
        return build_array(example, leading_dims, share_memory)
    return buffer_type(*(buffer_from_example(v, leading_dims, share_memory)
        for v in example))


def build_array(example, leading_dims, share_memory=False):
    a = np.asarray(example)
    if a.dtype == "object":
        raise TypeError("Buffer example value cannot cast as np.dtype==object.")
    constructor = np_mp_array if share_memory else np.zeros
    if not isinstance(leading_dims, (list, tuple)):
        leading_dims = (leading_dims,)
    return constructor(shape=leading_dims + a.shape, dtype=a.dtype)


def np_mp_array(shape, dtype):
    size = int(np.prod(shape))
    nbytes = size * np.dtype(dtype).itemsize
    mp_array = mp.RawArray(ctypes.c_char, nbytes)
    return np.frombuffer(mp_array, dtype=dtype, count=size).reshape(shape)


def torchify_buffer(buffer_):
    """
    把输入数据转换成torch.Tensor格式。

    :param buffer_: 输入数据，可能是numpy array，torch.Tensor等格式。
    :return: 转换得到的torch.Tensor。
    """
    if buffer_ is None:
        return
    if isinstance(buffer_, np.ndarray):
        return torch.from_numpy(buffer_)
    elif isinstance(buffer_, torch.Tensor):
        return buffer_
    contents = tuple(torchify_buffer(b) for b in buffer_)  # 递归调用，把 buffer_ 里的每一个元素都变成一个tensor，再组成一个tuple
    if type(buffer_) is tuple:  # tuple, namedtuple instantiate differently. 注意：namedtuple，namedarraytuple判断结果为False
        return contents
    return type(buffer_)(*contents)


def numpify_buffer(buffer_):
    """
    把输入数据转换成numpy array格式。

    :param buffer_: 输入数据，可能是torch.Tensor，numpy array等格式。
    :return: 转换得到的numpy array。
    """
    if buffer_ is None:
        return
    if isinstance(buffer_, torch.Tensor):
        return buffer_.cpu().numpy()
    elif isinstance(buffer_, np.ndarray):
        return buffer_
    contents = tuple(numpify_buffer(b) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def buffer_to(buffer_, device=None):
    if buffer_ is None:
        return
    if isinstance(buffer_, torch.Tensor):
        return buffer_.to(device)
    elif isinstance(buffer_, np.ndarray):
        raise TypeError("Cannot move numpy array to device.")
    contents = tuple(buffer_to(b, device=device) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def buffer_method(buffer_, method_name, *args, **kwargs):
    """
    把输入输入数据(buffer_)用指定函数(method_name)处理之后再组合成一定形式的数据。

    :param buffer_: 待转换的输入数据。
    :param method_name: 对buffer_或其子项调用的函数名，例如"copy"
    :param args: 为method_name提供的动态参数。
    :param kwargs: 为method_name提供的动态参数。
    :return: 转换后的数据。
    """
    if buffer_ is None:
        return
    if isinstance(buffer_, (torch.Tensor, np.ndarray)):  # buffer_的类型 是 torch.Tensor 或 np.ndarray 中的任何一个
        return getattr(buffer_, method_name)(*args, **kwargs)  # 执行 buffer_.method_name(*args, **kwargs)
    """
    当buffer_的类型不是 torch.Tensor 也不是 np.ndarray 的时候（例如 namedarraytuple），则会递归地把其每一个item都转成
    item.method_name(*args, **kwargs)的形式，再组合成一个tuple。具体转成了什么数据取决于 method_name 是什么，例如，假设 method_name
    为 "copy"，则会把每个item都进行复制，再组成一个tuple
    """
    contents = tuple(buffer_method(b, method_name, *args, **kwargs) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def buffer_func(buffer_, func, *args, **kwargs):
    if buffer_ is None:
        return
    if isinstance(buffer_, (torch.Tensor, np.ndarray)):
        return func(buffer_, *args, **kwargs)
    contents = tuple(buffer_func(b, func, *args, **kwargs) for b in buffer_)
    if type(buffer_) is tuple:
        return contents
    return type(buffer_)(*contents)


def get_leading_dims(buffer_, n_dim=1):
    if buffer_ is None:
        return
    if isinstance(buffer_, (torch.Tensor, np.ndarray)):
        return buffer_.shape[:n_dim]
    contents = tuple(get_leading_dims(b, n_dim) for b in buffer_ if b is not None)
    if not len(set(contents)) == 1:
        raise ValueError(f"Found mismatched leading dimensions: {contents}")
    return contents[0]


# def buffer_put(x, loc, y, axis=0, wrap=False):
#     if isinstance(x, (np.ndarray, torch.Tensor)):
#         put(x, loc, y, axis=axis, wrap=wrap)
#     else:
#         for vx, vy in zip(x, y):
#             buffer_put(vx, loc, vy, axis=axis, wrap=wrap)
