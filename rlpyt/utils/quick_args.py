
from inspect import getfullargspec


def save__init__args(values, underscore=False, overwrite=False, subclass_only=False):
    """
    Use in __init__() only; assign all args/kwargs to instance attributes.
    To maintain precedence of args provided to subclasses, call this in the
    subclass before super().__init__() if save__init__args() also appears in
    base class, or use overwrite=True.  With subclass_only==True, only args/kwargs
    listed in current subclass apply.
    """
    prefix = "_" if underscore else ""
    self = values['self']  # 在类的__init__()里调用locals()函数时，会把"self"这个item放进dict里面，因此values['self']取到的就是那个类对象
    args = list()
    Classes = type(self).mro()  # type()返回对象的类型，mro()函数返回该类型的方法解析顺序(MRO，Method Resolution Order)
    if subclass_only:
        Classes = Classes[:1]  # 由于在MRO列表中，子类永远在父类前面，取第一个元素即子类
    for Cls in Classes:  # class inheritances
        if '__init__' in vars(Cls):
            args += getfullargspec(Cls.__init__).args[1:]
    for arg in args:
        attr = prefix + arg
        if arg in values and (not hasattr(self, attr) or overwrite):
            setattr(self, attr, values[arg])
