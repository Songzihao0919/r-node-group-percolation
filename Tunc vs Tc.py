import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
import warnings
import matplotlib

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']  # 多种字体备选
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def compute_T_ratio(m, r, a_mean, a2_mean):
    """
    计算 T_unc/T_c 的比值 (公式56)

    参数:
    m: 超边大小
    r: 集团阶数
    a_mean: ⟨a⟩, 活跃度均值
    a2_mean: ⟨a²⟩, 活跃度二阶矩

    返回:
    ratio: T_unc/T_c
    """
    # 计算常数C(m,r)
    C = (factorial(r) * factorial(m - r)) / (2 * (factorial(m) - factorial(r) * factorial(m - r)))

    # 计算分子
    numerator_term = m * a_mean
    sqrt_term = np.sqrt((m ** 2 - 4 * m) * a_mean ** 2 + 4 * m * a2_mean)
    numerator = C * a_mean * (numerator_term + sqrt_term)

    # 计算分母
    denominator = a2_mean + (m - 1) * a_mean ** 2

    # 计算比值
    ratio = numerator / denominator

    return ratio, C


def powerlaw_distribution_moments(gamma=2.5, a_min=0.01, a_max=1.0, num_samples=1000000):
    """
    计算截断幂律分布的矩
    PDF: p(a) ∝ a^{-γ}, a∈[a_min, a_max]
    """
    if gamma == 1.0:
        # 特殊处理γ=1
        gamma = 1.0001

    # 生成幂律分布的随机样本
    np.random.seed(42)  # 固定随机种子，确保结果可重复
    u = np.random.uniform(0, 1, num_samples)

    if gamma != 1.0:
        a = (u * (a_max ** (1 - gamma) - a_min ** (1 - gamma)) + a_min ** (1 - gamma)) ** (1 / (1 - gamma))
    else:
        # γ=1 时的特殊情况
        a = a_min * (a_max / a_min) ** u

    a_mean = np.mean(a)
    a2_mean = np.mean(a ** 2)

    return a_mean, a2_mean


def plot_T_ratio_heatmap_new_range(gamma=2.5, figsize=(12, 9)):
    """
    绘制T_unc/T_c 随m和r变化的热力图
    新的范围：m从2到7，r从1到6
    """
    # 幂律分布的矩
    a_mean_pl, a2_mean_pl = powerlaw_distribution_moments(gamma=gamma)

    # 创建m和r的网格 - 新范围
    m_range = np.arange(2, 8)  # m从2到7
    r_range = np.arange(1, 7)  # r从1到6

    # 计算矩阵大小
    m_size = len(m_range)
    r_size = len(r_range)

    # 初始化结果矩阵
    Z = np.full((r_size, m_size), np.nan, dtype=float)

    # 计算每个格子的值
    for i, m in enumerate(m_range):
        for j, r in enumerate(r_range):
            if r < m:  # 确保r<m
                ratio, _ = compute_T_ratio(m, r, a_mean_pl, a2_mean_pl)
                Z[j, i] = ratio

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 使用pcolormesh绘制热力图
    # 创建边界坐标
    m_edges = np.linspace(1.5, 7.5, m_size + 1)  # m边界
    r_edges = np.linspace(0.5, 6.5, r_size + 1)  # r边界

    M_edges, R_edges = np.meshgrid(m_edges, r_edges)

    im = ax.pcolormesh(M_edges, R_edges, Z,
                       cmap='viridis', vmin=0, vmax=np.nanmax(Z) * 1.1,
                       edgecolors='black', linewidth=0.5, shading='flat')

    # 设置坐标轴标签
    ax.set_xlabel('$m$', fontsize=16)
    ax.set_ylabel('$r$', fontsize=16)

    # 设置标题
    # ax.set_title(f'$T_{{{{\\mathrm{{unc}}}}}}/T_c$ 热力图 (幂律分布, $\\gamma={gamma}$)',
    #             fontsize=18, pad=20)

    # 设置刻度
    ax.set_xticks(m_range)
    ax.set_yticks(r_range)
    ax.set_xlim(1.5, 7.5)
    ax.set_ylim(0.5, 6.5)

    # 添加数值标注
    for i, m in enumerate(m_range):
        for j, r in enumerate(r_range):
            if r < m:  # 只显示有效的组合
                value = Z[j, i]
                if not np.isnan(value):
                    # 格子的中心位置
                    x_center = m
                    y_center = r

                    # 根据数值大小选择文本颜色
                    text_color = 'white' if value > np.nanmax(Z) / 2 else 'black'

                    # 格式化数值显示
                    if value < 0.001:
                        value_str = f'{value:.2e}'
                    elif value < 0.01:
                        value_str = f'{value:.4f}'
                    else:
                        value_str = f'{value:.4f}'

                    # 在格子中心添加文本
                    ax.text(x_center, y_center, value_str,
                            ha='center', va='center',
                            color=text_color,
                            fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle="circle,pad=0.3",
                                      facecolor="none", edgecolor="none", alpha=0.7))

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('$\\frac{T_{\\mathrm{unc}}}{T_c}$', fontsize=18, rotation=0, labelpad=25)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(f'T_ratio_heatmap_m2-7_r1-6_gamma{gamma}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return Z, a_mean_pl, a2_mean_pl, m_range, r_range


def plot_T_ratio_vs_r_fixed_m(gamma=2.5, m_values=[6, 7], figsize=(10, 6)):
    """
    绘制固定m时，T_unc/T_c随r变化的曲线
    """
    # 幂律分布的矩
    a_mean_pl, a2_mean_pl = powerlaw_distribution_moments(gamma=gamma)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    # 颜色和标记
    colors = ['b', 'r']
    markers = ['o', 's']

    for idx, m in enumerate(m_values):
        # 计算r的范围：从1到m-1
        r_values = np.arange(1, m)
        T_ratios = []

        for r in r_values:
            ratio, _ = compute_T_ratio(m, r, a_mean_pl, a2_mean_pl)
            T_ratios.append(ratio)

        # 绘制曲线
        ax.plot(r_values, T_ratios,
                color=colors[idx],
                marker=markers[idx],
                linewidth=2,
                markersize=8,
                label=f'$m={m}$')

        # 添加数据点标签
        for r, ratio in zip(r_values, T_ratios):
            ax.text(r, ratio, f'{ratio:.4f}',
                    ha='center', va='bottom',
                    fontsize=16, fontweight='bold')

    # 设置坐标轴标签
    ax.set_xlabel('$r$', fontsize=16)
    ax.set_ylabel('$\\frac{T_{\\mathrm{unc}}}{T_c}$', fontsize=18, rotation=0, labelpad=20)

    # 设置标题
    #ax.set_title(f'固定$m$时$T_{{\\mathrm{{unc}}}}/T_c$随$r$的变化 (幂律分布, $\\gamma={gamma}$)',
    #             fontsize=16, pad=20)

    # 添加网格
    ax.grid(True, alpha=0.3)

    # 添加图例
    ax.legend(fontsize=12)

    # 设置x轴刻度
    all_r_values = []
    for m in m_values:
        all_r_values.extend(range(1, m))
    unique_r = sorted(set(all_r_values))
    ax.set_xticks(unique_r)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    m_str = '_'.join([str(m) for m in m_values])
    plt.savefig(f'T_ratio_vs_r_m{m_str}_gamma{gamma}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return


def print_heatmap_values(Z, m_range, r_range):
    """打印热力图数值，用于验证"""
    print("热力图数值矩阵:")
    print("    m:", end="")
    for m in m_range:
        print(f"{m:10d}", end="")
    print()

    for j, r in enumerate(r_range):
        print(f"r={r}:", end="")
        for i, m in enumerate(m_range):
            if r < m:
                print(f"{Z[j, i]:10.6f}", end="")
            else:
                print("          ", end="")
        print()


def analyze_special_cases(gamma=2.5):
    """分析特殊案例，特别是r=1的情况"""
    a_mean_pl, a2_mean_pl = powerlaw_distribution_moments(gamma=gamma)

    print("\n特殊案例分析 (r=1):")
    print(f"{'m':>3} {'C(m,1)':>10} {'T_unc/T_c (r=1)':>15}")
    print("-" * 35)

    for m in range(2, 8):  # m=2到7
        if m > 1:  # r=1, 需要m>1
            ratio, C = compute_T_ratio(m, 1, a_mean_pl, a2_mean_pl)
            print(f"{m:3d} {C:10.6f} {ratio:15.6f}")

    print("\n趋势分析:")
    print("1. 当r=1时，随着m增大，C(m,1)减小")
    print("2. 当r=1时，随着m增大，T_unc/T_c的变化趋势是...")

    # 计算r=1时随m变化的趋势
    m_values = np.arange(2, 8)
    ratios_r1 = []
    for m in m_values:
        ratio, _ = compute_T_ratio(m, 1, a_mean_pl, a2_mean_pl)
        ratios_r1.append(ratio)

    # 计算变化率
    if len(ratios_r1) > 1:
        changes = np.diff(ratios_r1) / ratios_r1[:-1] * 100
        print("\nr=1时T_unc/T_c随m的变化率 (%):")
        for i, (m1, m2, change) in enumerate(zip(m_values[:-1], m_values[1:], changes)):
            print(f"  m={m1}→{m2}: {change:6.2f}%")


def main_new_range():
    """主函数：生成新范围的热力图"""
    print("=" * 60)
    print("T_unc/T_c 热力图生成 (幂律分布, γ=2.5)")
    print("新范围: m=2-7, r=1-6")
    print("=" * 60)

    # 设置幂律指数
    gamma = 2.5

    # 生成热力图
    Z, a_mean_pl, a2_mean_pl, m_range, r_range = plot_T_ratio_heatmap_new_range(gamma=gamma, figsize=(14, 10))

    # 打印热力图数值
    print("\n热力图数值矩阵:")
    print_heatmap_values(Z, m_range, r_range)

    # 分析特殊案例
    analyze_special_cases(gamma=gamma)

    # 分析对角线趋势
    print("\n对角线分析 (r = m-1):")
    print(f"{'m':>3} {'r':>3} {'T_unc/T_c':>12}")
    print("-" * 25)
    for m in m_range:
        r = m - 1
        if r >= 1:  # 确保r有效
            ratio, _ = compute_T_ratio(m, r, a_mean_pl, a2_mean_pl)
            print(f"{m:3d} {r:3d} {ratio:12.6f}")

    # 新增：绘制固定m时T_unc/T_c随r变化的曲线
    print("\n" + "=" * 60)
    print("生成固定m时T_unc/T_c随r变化的曲线图...")
    plot_T_ratio_vs_r_fixed_m(gamma=gamma, m_values=[6, 7], figsize=(12, 8))

    print("\n" + "=" * 60)
    print(f"热力图已生成并保存为: T_ratio_heatmap_m2-7_r1-6_gamma{gamma}.png")
    print(f"曲线图已生成并保存为: T_ratio_vs_r_m6_7_gamma{gamma}.png")
    print("=" * 60)

    return {
        'Z': Z,
        'm_range': m_range,
        'r_range': r_range,
        'a_mean': a_mean_pl,
        'a2_mean': a2_mean_pl,
        'gamma': gamma
    }


if __name__ == "__main__":
    results = main_new_range()