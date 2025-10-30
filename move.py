import math


def translate_right_of_heading(x, y, heading_rad, distance):
    """
    沿当前 heading 顺时针旋转 90° 的方向平移 distance 距离。

    参数：
      x, y           : 当前坐标
      heading_rad    : 当前朝向（弧度），0 表示 x 轴正方向，顺时针为正（或你定义的方向）
      distance       : 要平移的距离（与 x,y 单位一致）

    返回：
      (x_new, y_new) : 平移后坐标
    """
    # 顺时针旋转90°意味着向右侧偏移
    # 如果 heading_rad 是车头朝向，那么右侧偏移方向的角度 = heading_rad - π/2
    theta = heading_rad - math.pi / 2.0

    # 计算偏移向量
    dx = distance * math.cos(theta)
    dy = distance * math.sin(theta)

    x_new = x + dx
    y_new = y + dy
    return x_new, y_new


# 示例使用
if __name__ == "__main__":
    # 假设车辆当前位置
    x0 = -4.118499367
    y0 = 34.969797470
    heading_rad = 1.493347101
    dist = 0.3  # 平移 2 单位

    x1, y1 = translate_right_of_heading(x0, y0, heading_rad, dist)
    print(f"原位置: ({x0:.3f}, {y0:.3f}), heading={heading_rad}°")
    print(f"平移 {dist} 单位后: ({x1}, {y1})")
