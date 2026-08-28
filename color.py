import matplotlib.cm
import matplotlib.colors
import numpy as np

def get_color(colormap='plasma', range=[0.1, 0.9], n=6):
    '''
    获取颜色图中的颜色并返回16进制颜色列表
    :param colormap: 颜色图名称（字符串）或colormap对象，默认为'plasma'
    :param range: 颜色范围，默认为[0.1, 0.9]
    :param n: 颜色数量，默认为6
    :return: 16进制颜色列表
    '''
    # 处理colormap参数
    if isinstance(colormap, str):
        try:
            cmap = matplotlib.cm.get_cmap(colormap)
        except ValueError:
            raise ValueError(f"无效的colormap名称: {colormap}")
    elif isinstance(colormap, matplotlib.colors.Colormap):
        cmap = colormap
    else:
        raise TypeError("colormap必须是字符串或Colormap对象")

    # 在range范围内生成n个均匀分布的点
    points = np.linspace(range[0], range[1], n)

    # 从颜色图中获取每个点对应的RGBA颜色
    colors = [cmap(p) for p in points]

    # 将RGBA颜色转换为16进制表示
    hex_colors = [matplotlib.colors.to_hex(c) for c in colors]
    return hex_colors

# 测试代码
if __name__ == '__main__':
    print(get_color(range=[0.2, 0.8]))