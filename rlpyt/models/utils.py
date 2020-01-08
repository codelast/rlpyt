
import torch


def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Returns output H, W after convolution/pooling on input H, W.
    """
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
    sh, sw = stride if isinstance(stride, tuple) else (stride,) * 2
    ph, pw = padding if isinstance(padding, tuple) else (padding,) * 2
    d = dilation
    h = (h + (2 * ph) - (d * (kh - 1)) - 1) // sh + 1
    w = (w + (2 * pw) - (d * (kw - 1)) - 1) // sw + 1
    return h, w


class ScaleGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, scale):
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return grad_output * ctx.scale, None


scale_grad = ScaleGrad.apply


def update_state_dict(model, state_dict, tau=1, strip_ddp=True):
    """
    用输入的模型参数(state_dict)，来更新目标模型(model)的参数。当τ>0的时候，使用soft update算法来更新参数，即：θ‘=θ×τ+θ’×(1−τ)，其中θ’
    是待更新的模型的参数。

    :param model: 待更新的目标模型的参数。
    :param state_dict: 用于更新目标模型的输入参数。
    :param tau: soft update算法里的τ参数。
    :param strip_ddp: 参考strip_ddp_state_dict()函数。
    """
    if strip_ddp:
        state_dict = strip_ddp_state_dict(state_dict)
    if tau == 1:
        model.load_state_dict(state_dict)
    elif tau > 0:
        update_sd = {k: tau * state_dict[k] + (1 - tau) * v
            for k, v in model.state_dict().items()}
        model.load_state_dict(update_sd)


def strip_ddp_state_dict(state_dict):
    # Workaround the fact that DistributedDataParallel prepends 'module.' to
    # every key, but the sampler models will not be wrapped in
    # DistributedDataParallel. (Solution from PyTorch forums.)
    clean_state_dict = type(state_dict)()
    for k, v in state_dict.items():
        key = k[7:] if k[:7] == "module." else k  # 去掉key里包含的"module."前缀(如果有的话)
        clean_state_dict[key] = v
    return clean_state_dict
