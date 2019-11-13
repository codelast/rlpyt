"""
这个类很tricky，作者没有写注释，需要好好理解。
这个类的存在意义是：在具体的model类(例如AtariDqnModel)的实现中，需要一些和具体model相关的特殊参数(例如image_shape)，这些参数需要在构造
model类对象的时候(BaseAgent.initialize()函数里)提供，如果不在这里赋值的话，就要传入BaseAgent类；而BaseAgent.initialize()函数又是在
Sampler类(例如SerialSampler)中被调用的，因此，Sampler类也要提供这些参数。考虑到model类(例如AtariDqnModel)所需的特殊参数，已经完全可以从
environment spaces(即EnvSpaces这个类型的namedtuple)推断出来，并且environment spaces在Sampler类的initialize()函数里已经现成可用了，
因此直接把它传给agent类的initialize()函数，并且在里面用make_env_to_model_kwargs()函数提取出来就可以了。但为什么不直接在agent类(例如
AtariDqnAgent)的里面添加一个make_env_to_model_kwargs()函数呢？因为在里面拿不到environment spaces，所以没办法在里面构造出这些参数。另
一个办法就是在AtariDqnAgent实例化的时候，例如在example_1.py里面，直接把这些特殊参数传进去，作者没有这样做，我猜测可能的原因是，这里的特殊参
数和具体的游戏相关(例如output_size)，如果要在example_1.py中传入，则需要先用ALE接口获取特定游戏的这个参数，再构造AtariDqnAgent对象，这使得
每个构造AtariDqnAgent的地方都需要写这些逻辑，造成了代码的冗余。
"""


class AtariMixin:

    def make_env_to_model_kwargs(self, env_spaces):
        """
        为具体的model类(例如AtariDqnModel)实例化提供一些必需的特殊参数。

        :param env_spaces: 一个namedtuple(参考class Env里的EnvSpaces)，包含observation space 和 action space两个属性。这里的
        env_spaces.observation 和 env_spaces.action，都是 IntBox 类型的对象，因此 env_spaces.observation.shape 就对应 IntBox
        里的 self.shape，而 env_spaces.action.n 则对应 IntBox.n() 函数的返回值(它用@property修饰使之可以像属性一样调用)。
        :return: 一个dict，其包含创建model类(例如AtariDqnModel)对象所需的特殊参数。
        """
        return dict(image_shape=env_spaces.observation.shape,
                    output_size=env_spaces.action.n)
