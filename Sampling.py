import numpy as np

def supersample(model_func, data_X, params, num_samples=11):
    """
    对连续模型进行超采样，通过积分计算离散采样点的预测值。
    
    参数：
        model_func (callable): 连续模型函数，接受两个参数：x（采样点）和 params（模型参数）
        data_X (numpy.ndarray): 输入数据点数组，形状为 (n,)
        params: 模型参数，具体形式由 model_func 决定
        num_samples (int): 每个区间的采样点数，默认为 10
    
    返回：
        model_pred (numpy.ndarray): 每个点的积分预测值，形状为 (n,)
    """
    n = len(data_X)
    
    # 初始化每个区间的下限和上限
    lowers = np.zeros(n)
    uppers = np.zeros(n)
    
    # 第一个区间
    lowers[0] = data_X[0] - (data_X[1] - data_X[0])/2
    uppers[0] = (data_X[0] + data_X[1]) / 2
    
    # 中间区间（向量化计算）
    lowers[1:-1] = (data_X[:-2] + data_X[1:-1]) / 2
    uppers[1:-1] = (data_X[1:-1] + data_X[2:]) / 2
    
    # 最后一个区间
    lowers[-1] = (data_X[-2] + data_X[-1]) / 2
    uppers[-1] = data_X[-1] + (data_X[-1] - data_X[-2])/2
    
    # 计算每个区间的步长
    steps = (uppers - lowers) / (num_samples - 1)
    
    # 生成所有采样点，形状为 (n, num_samples)
    k = np.arange(num_samples)
    samples = lowers[:, np.newaxis] + k * steps[:, np.newaxis]
    
    # 将采样点展平为 1D 数组，计算 model_func 值
    samples_flat = samples.ravel()  # 形状为 (n * num_samples,)
    values_flat = model_func(samples_flat, params = params)  # 调用传入的模型函数
    values = values_flat.reshape(n, num_samples)  # 重塑为 (n, num_samples)
    
    # # 使用梯形法则计算每个区间的积分
    # model_pred = (0.5 * values[:, 0] + np.sum(values[:, 1:-1], axis=1) + 0.5 * values[:, -1])/ (num_samples - 1)
    
    # 定义 Simpson 系数
    coeffs = np.ones(num_samples)
    coeffs[1:-1:2] = 4  # 奇数点系数为 4
    coeffs[2:-1:2] = 2  # 偶数点系数为 2
    
    # 计算 Simpson 方法平均值（注意不是积分）
    model_pred = np.sum(values * coeffs, axis=1)/3 / (num_samples - 1)
    
    return model_pred


def supersample_decorator(num_samples=11):
    """
    装饰器：对连续模型函数进行超采样，通过 Simpson 方法计算离散采样点的预测值。
    
    参数：
        num_samples (int): 每个区间的采样点数，必须为奇数，默认为 7
    
    返回：
        decorator: 返回装饰器函数
    """
    if num_samples % 2 == 0:
        raise ValueError("num_samples 必须为奇数以使用 Simpson 方法。")

    def decorator(model_func):
        def wrapper(data_X, *args, **kwargs):
            # 如果 data_X 是标量或不提供超采样，直接调用原始函数
            if not isinstance(data_X, np.ndarray) or data_X.ndim == 0 or num_samples == 1:
                return model_func(data_X, *args, **kwargs)
            
            # 超采样逻辑
            n = len(data_X)
            
            # 初始化每个区间的下限和上限
            lowers = np.zeros(n)
            uppers = np.zeros(n)
            
            # 第一个区间
            lowers[0] = data_X[0] - (data_X[1] - data_X[0]) / 2
            uppers[0] = (data_X[0] + data_X[1]) / 2
            
            # 中间区间（向量化计算）
            lowers[1:-1] = (data_X[:-2] + data_X[1:-1]) / 2
            uppers[1:-1] = (data_X[1:-1] + data_X[2:]) / 2
            
            # 最后一个区间
            lowers[-1] = (data_X[-2] + data_X[-1]) / 2
            uppers[-1] = data_X[-1] + (data_X[-1] - data_X[-2]) / 2
            
            # 计算每个区间的步长
            steps = (uppers - lowers) / (num_samples - 1)
            
            # 生成所有采样点，形状为 (n, num_samples)
            k = np.arange(num_samples)
            samples = lowers[:, np.newaxis] + k * steps[:, np.newaxis]
            
            # 将采样点展平为 1D 数组，计算 model_func 值
            samples_flat = samples.ravel()  # 形状为 (n * num_samples,)
            values_flat = model_func(samples_flat, *args, **kwargs)
            values = values_flat.reshape(n, num_samples)  # 重塑为 (n, num_samples)
            
            # 定义 Simpson 系数
            coeffs = np.ones(num_samples)
            coeffs[1:-1:2] = 4  # 奇数点系数为 4
            coeffs[2:-1:2] = 2  # 偶数点系数为 2
            
            # 计算 Simpson 积分
            model_pred = np.sum(values * coeffs, axis=1)/3 / (num_samples - 1)
            
            return model_pred
        return wrapper
    return decorator


## 测试代码
if __name__ == '__main__':

    # 定义线性模型
    def linear_model(x, params):
        a, b = params
        return a * x + b

    # 定义指数模型
    @supersample_decorator()
    def exp_model(x, a=1, b=1, params =[]):
        if len(params) != 0:
            a, b = params
        return a * np.exp(b * x)

    # 测试数据
    data_X = np.array([0.0, 1.0, 2.0, 3.0])

    # 使用线性模型
    params_linear = [2.0, 1.0]  # a=2, b=1
    print("线性模型未采样预测值:", linear_model(data_X, params_linear))

    # 使用指数模型
    params_exp = [1.0, 0.5]  # a=1, b=0.5
    print("指数模型未采样预测值:", exp_model(data_X, b=0.5))