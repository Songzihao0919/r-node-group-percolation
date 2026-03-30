import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_corrected_hypergraph_and_factor_tree():
    """生成准确匹配用户示意图的超图和因子树"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # ========== 左图：超图 ==========
    ax_left = axes[0]

    # 设置节点位置（黑色点）
    node_positions = [
        # 蓝色超边中的节点（5个节点）
        (2.0, 4.5),  # 节点1（蓝色独有）
        (3.0, 4.8),  # 节点2（蓝色独有）

        # 蓝色和红色共享的3个节点
        (2.5, 4.0),  # 共享节点1
        (3.0, 4.0),  # 共享节点2
        (3.5, 4.0),  # 共享节点3

        # 红色超边中额外的节点（总共6个节点，3个共享+3个独有）
        (3.5, 3.3),  # 红色独有节点1
        (4.0, 4.3),  # 红色独有节点2
        (4.2, 3.8),  # 红色独有节点3

        # 橙色超边中的节点（4个节点，包含部分共享节点）
        (3.0, 3.5),  # 橙色节点1
        (3.5, 3.5),  # 橙色节点2

        # 绿色超边中的节点（6个节点）
        (2.0, 2.5),  # 绿色节点1
        (2.5, 3.0),  # 绿色节点2
        (3.0, 3.0),  # 绿色节点3
        (3.5, 3.0),  # 绿色节点4
        (4.0, 2.5),  # 绿色节点5
        (2.8, 2.2),  # 绿色节点6
    ]

    # 绘制节点（黑色点）
    for pos in node_positions:
        ax_left.plot(pos[0], pos[1], 'ko', markersize=6, zorder=10)

    # 绘制超边（彩色椭圆形/多边形）

    # 1. 蓝色超边（5个节点）
    blue_nodes_indices = [0, 1, 2, 3, 4]  # 索引对应node_positions
    blue_points = [node_positions[i] for i in blue_nodes_indices]
    blue_center = np.mean(blue_points, axis=0)
    ellipse_blue = patches.Ellipse(blue_center, 2.0, 1.2,
                                   facecolor='#87CEEB', alpha=0.4, edgecolor='#87CEEB',
                                   linewidth=2, zorder=1)
    ax_left.add_patch(ellipse_blue)

    # 2. 红色超边（6个节点）
    red_nodes_indices = [2, 3, 4, 5, 6, 7]  # 共享3个节点+3个独有节点
    red_points = [node_positions[i] for i in red_nodes_indices]
    red_center = np.mean(red_points, axis=0)
    ellipse_red = patches.Ellipse(red_center, 2.5, 1.5,
                                  facecolor='#FF9999', alpha=0.4, edgecolor='#FF9999',
                                  linewidth=2, zorder=2)
    ax_left.add_patch(ellipse_red)

    # 3. 橙色超边（4个节点）
    orange_nodes_indices = [3, 4, 7, 8]  # 部分共享节点+橙色独有节点
    orange_points = [node_positions[i] for i in orange_nodes_indices]
    orange_center = np.mean(orange_points, axis=0)
    ellipse_orange = patches.Ellipse(orange_center, 1.5, 1.0,
                                     facecolor='#FFCC99', alpha=0.4, edgecolor='#FFCC99',
                                     linewidth=2, zorder=3)
    ax_left.add_patch(ellipse_orange)

    # 4. 绿色超边（6个节点）
    green_nodes_indices = [9, 10, 11, 12, 13, 14]
    green_points = [node_positions[i] for i in green_nodes_indices]
    green_center = np.mean(green_points, axis=0)

    # 使用多边形创建不规则形状
    from scipy.spatial import ConvexHull
    points_array = np.array(green_points)
    hull = ConvexHull(points_array)

    green_polygon = patches.Polygon(points_array[hull.vertices],
                                    closed=True, facecolor='#99FF99', alpha=0.4,
                                    edgecolor='#99FF99', linewidth=2, zorder=4)
    ax_left.add_patch(green_polygon)

    # 高亮显示蓝色和红色共享的3个节点
    shared_nodes_positions = [node_positions[i] for i in [2, 3, 4]]
    for pos in shared_nodes_positions:
        circle = plt.Circle(pos, 0.15, color='black', fill=False, linewidth=2, zorder=11)
        ax_left.add_patch(circle)

    ax_left.set_xlim(1, 5.5)
    ax_left.set_ylim(1.8, 5.2)
    ax_left.set_aspect('equal')
    ax_left.axis('off')

    # ========== 右图：修正的因子树 ==========
    ax_right = axes[1]

    # 设置节点位置
    # 树结构：
    #   三角形1
    #     / \
    #    5   6
    #       / \
    #      三角形2
    #      /   \
    #     4     7

    positions = {
        'triangle1': (1, 3),
        'square_blue': (2, 3.5),  # 蓝色，数字5
        'square_red': (2, 2.5),  # 红色，数字6
        'triangle2': (3, 3),
        'square_orange': (4, 3.5),  # 橙色，数字4
        'square_green': (4, 2.5)  # 绿色，数字7
    }

    # 绘制连接线（使用箭头）
    connections = [
        (positions['triangle1'], positions['square_blue']),
        (positions['triangle1'], positions['square_red']),
        (positions['square_red'], positions['triangle2']),
        (positions['triangle2'], positions['square_orange']),
        (positions['triangle2'], positions['square_green'])
    ]

    for i, ((x1, y1), (x2, y2)) in enumerate(connections):
        # 如果是到三角形2的连接，稍微弯曲
        if i == 2:  # 红色6到三角形2的连接
            # 使用三次贝塞尔曲线
            ax_right.annotate('', xy=(x2, y2), xytext=(x1, y1),
                              arrowprops=dict(arrowstyle='->', color='black',
                                              connectionstyle="arc3,rad=0.2", linewidth=1.5))
        else:
            ax_right.annotate('', xy=(x2, y2), xytext=(x1, y1),
                              arrowprops=dict(arrowstyle='->', color='black', linewidth=1.5))

    # 绘制三角形（黑色，代表3-node）
    # 三角形1
    triangle1 = plt.Polygon([
        (positions['triangle1'][0], positions['triangle1'][1] + 0.2),
        (positions['triangle1'][0] - 0.2, positions['triangle1'][1] - 0.15),
        (positions['triangle1'][0] + 0.2, positions['triangle1'][1] - 0.15)
    ], facecolor='black', edgecolor='black', zorder=5)
    ax_right.add_patch(triangle1)

    # 三角形2
    triangle2 = plt.Polygon([
        (positions['triangle2'][0], positions['triangle2'][1] + 0.2),
        (positions['triangle2'][0] - 0.2, positions['triangle2'][1] - 0.15),
        (positions['triangle2'][0] + 0.2, positions['triangle2'][1] - 0.15)
    ], facecolor='black', edgecolor='black', zorder=5)
    ax_right.add_patch(triangle2)

    # 绘制正方形（彩色，代表超边）
    # 蓝色正方形（数字5）
    square_blue = patches.Rectangle(
        (positions['square_blue'][0] - 0.25, positions['square_blue'][1] - 0.25),
        0.5, 0.5, facecolor='#87CEEB', edgecolor='black', linewidth=2, zorder=5
    )
    ax_right.add_patch(square_blue)
    ax_right.text(positions['square_blue'][0], positions['square_blue'][1],
                  '5', ha='center', va='center', fontsize=14, fontweight='bold', zorder=6)

    # 红色正方形（数字6）
    square_red = patches.Rectangle(
        (positions['square_red'][0] - 0.25, positions['square_red'][1] - 0.25),
        0.5, 0.5, facecolor='#FF9999', edgecolor='black', linewidth=2, zorder=5
    )
    ax_right.add_patch(square_red)
    ax_right.text(positions['square_red'][0], positions['square_red'][1],
                  '6', ha='center', va='center', fontsize=14, fontweight='bold', zorder=6)

    # 橙色正方形（数字4）
    square_orange = patches.Rectangle(
        (positions['square_orange'][0] - 0.25, positions['square_orange'][1] - 0.25),
        0.5, 0.5, facecolor='#FFCC99', edgecolor='black', linewidth=2, zorder=5
    )
    ax_right.add_patch(square_orange)
    ax_right.text(positions['square_orange'][0], positions['square_orange'][1],
                  '4', ha='center', va='center', fontsize=14, fontweight='bold', zorder=6)

    # 绿色正方形（数字7）
    square_green = patches.Rectangle(
        (positions['square_green'][0] - 0.25, positions['square_green'][1] - 0.25),
        0.5, 0.5, facecolor='#99FF99', edgecolor='black', linewidth=2, zorder=5
    )
    ax_right.add_patch(square_green)
    ax_right.text(positions['square_green'][0], positions['square_green'][1],
                  '7', ha='center', va='center', fontsize=14, fontweight='bold', zorder=6)

    ax_right.set_xlim(0.5, 4.5)
    ax_right.set_ylim(2, 4)
    ax_right.set_aspect('equal')
    ax_right.axis('off')

    plt.tight_layout()
    plt.show()

    return fig


def draw_alternative_style():
    """另一种更接近手绘风格的绘制方式"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 左图：超图
    ax_left = axes[0]

    # 更简单的布局，专注于展示超边重叠
    # 创建三个重叠的超边

    # 蓝色超边（椭圆）
    ellipse_blue = patches.Ellipse((3, 4.2), 2.2, 1.1,
                                   facecolor='#1E90FF', alpha=0.4, edgecolor='#1E90FF',
                                   linewidth=2, linestyle='--', zorder=1)
    ax_left.add_patch(ellipse_blue)

    # 红色超边（椭圆）
    ellipse_red = patches.Ellipse((3.5, 3.8), 2.5, 1.5,
                                  facecolor='#FF9999', alpha=0.4, edgecolor='#FF9999',
                                  linewidth=2, zorder=2)
    ax_left.add_patch(ellipse_red)

    # 橙色超边（小椭圆）
    ellipse_orange = patches.Ellipse((3.1, 3.63), 2, 1.3,
                                     facecolor='#FF9900', alpha=0.4, edgecolor='#FF9900',
                                     linewidth=1.5, zorder=3)
    ax_left.add_patch(ellipse_orange)

    # 绿色超边（不规则形状）
    green_polygon = patches.Polygon(
        [(2, 3), (3, 3.8), (4, 3.5), (4.5, 2.8), (3.5, 2), (2.5, 2)],
        closed=True, facecolor='#99FF99', alpha=0.4, edgecolor='#99FF99',
        linewidth=1.5, zorder=4
    )
    ax_left.add_patch(green_polygon)

    # 添加一些节点
    # 在重叠区域添加3个节点
    for x, y in [(3.7, 3.3), (3.2, 3.4), (3.6, 3.5), (3, 4.1), (3.9, 4.2), (3.8, 4.4)]:
        ax_left.plot(x, y, 'ko', markersize=6, zorder=10)
        # 用白边突出显示
        circle = plt.Circle((x, y), 0.1, color='white', fill=False, linewidth=1.5, zorder=9)
        ax_left.add_patch(circle)

    #ax_left.plot(3, 4.1, 'ko', markersize=6, zorder=10)
    # 蓝色独有节点
    ax_left.plot(2.2, 4.5, 'ko', markersize=6, zorder=10)
    ax_left.plot(2.5, 4.6, 'ko', markersize=6, zorder=10)

    # 红色独有节点
    #ax_left.plot(3.9, 4.2, 'ko', markersize=6, zorder=10)
    #ax_left.plot(3.8, 4.4, 'ko', markersize=6, zorder=10)
    #ax_left.plot(3.8, 3.2, 'ko', markersize=6, zorder=10)

    # 绿色节点
    for x, y in [(2.5, 2.5), (3.0, 2.6), (3.5, 2.5), (3.2, 2.4)]:
        ax_left.plot(x, y, 'ko', markersize=6, zorder=10)

    ax_left.set_xlim(1.5, 5.5)
    ax_left.set_ylim(1.5, 5)
    ax_left.set_aspect('equal')
    ax_left.axis('off')

    # 右图：因子树
    ax_right = axes[1]

    # 绘制树结构
    # 使用曲线连接
    import matplotlib.patches as mpatches

    # 节点位置
    triangle1_pos = (1, 3)
    blue5_pos = (2.2, 3.5)
    red6_pos = (2.2, 2.5)
    triangle2_pos = (3.4, 3)
    orange4_pos = (4.6, 3.5)
    green7_pos = (4.6, 2.5)

    # 绘制连接曲线
    # 三角形1 -> 蓝色5
    ax_right.annotate('', xy=blue5_pos, xytext=triangle1_pos,
                      arrowprops=dict(arrowstyle='->', color='black',
                                      connectionstyle="arc3,rad=0", linewidth=1.5))

    # 三角形1 -> 红色6
    ax_right.annotate('', xy=red6_pos, xytext=triangle1_pos,
                      arrowprops=dict(arrowstyle='->', color='black',
                                      connectionstyle="arc3,rad=0", linewidth=1.5))

    # 红色6 -> 三角形2（带弧度的曲线）
    ax_right.annotate('', xy=triangle2_pos, xytext=red6_pos,
                      arrowprops=dict(arrowstyle='->', color='black',
                                      connectionstyle="arc3,rad=0", linewidth=1.5))

    # 三角形2 -> 橙色4
    ax_right.annotate('', xy=orange4_pos, xytext=triangle2_pos,
                      arrowprops=dict(arrowstyle='->', color='black',
                                      connectionstyle="arc3,rad=0", linewidth=1.5))

    # 三角形2 -> 绿色7
    ax_right.annotate('', xy=green7_pos, xytext=triangle2_pos,
                      arrowprops=dict(arrowstyle='->', color='black',
                                      connectionstyle="arc3,rad=0", linewidth=1.5))

    # 绘制三角形（黑色实心）
    triangle1 = plt.Polygon([
        (triangle1_pos[0], triangle1_pos[1] + 0.2),
        (triangle1_pos[0] - 0.2, triangle1_pos[1] - 0.15),
        (triangle1_pos[0] + 0.2, triangle1_pos[1] - 0.15)
    ], facecolor='black', edgecolor='black', linewidth=1.5, zorder=5)
    ax_right.add_patch(triangle1)

    triangle2 = plt.Polygon([
        (triangle2_pos[0], triangle2_pos[1] + 0.2),
        (triangle2_pos[0] - 0.2, triangle2_pos[1] - 0.15),
        (triangle2_pos[0] + 0.2, triangle2_pos[1] - 0.15)
    ], facecolor='black', edgecolor='black', linewidth=1.5, zorder=5)
    ax_right.add_patch(triangle2)

    # 绘制正方形
    colors = ['#1E90FF', '#FF9999', '#FF9900', '#99FF99']
    positions = [blue5_pos, red6_pos, orange4_pos, green7_pos]
    numbers = ['5', '6', '4', '7']

    for i, (pos, color, num) in enumerate(zip(positions, colors, numbers)):
        square = patches.Rectangle((pos[0] - 0.25, pos[1] - 0.25), 0.5, 0.5,
                                   facecolor=color, edgecolor='black', linewidth=1.5, zorder=5)
        ax_right.add_patch(square)
        ax_right.text(pos[0], pos[1], num, ha='center', va='center',
                      fontsize=14, fontweight='bold', zorder=6)

    ax_right.set_xlim(0.5, 5.5)
    ax_right.set_ylim(2, 4)
    ax_right.set_aspect('equal')
    ax_right.axis('off')

    plt.tight_layout()
    plt.show()

    return fig


# 生成图像
if __name__ == "__main__":
    print("生成修正后的超图和因子树...")
    fig1 = draw_corrected_hypergraph_and_factor_tree()

    print("\n生成另一种风格（更接近手绘）...")
    fig2 = draw_alternative_style()