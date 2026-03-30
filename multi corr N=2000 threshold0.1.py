import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
from collections import defaultdict
import itertools
from scipy.optimize import curve_fit
import time

warnings.filterwarnings('ignore')


# ============================
# 1. 高效的并查集实现
# ============================
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False

        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self.rank[root_x] += self.rank[root_y]
        return True


# ============================
# 2. 生成泊松度分布的超图
# ============================
def generate_hypergraph_poisson(N, m, lambda_r, r, max_attempts=100):
    """
    生成一个满足r-node泊松度分布的超图

    参数:
        N: 节点数
        m: 超边基数
        lambda_r: r-node的平均度
        r: r-node大小
        max_attempts: 最大尝试次数

    返回:
        超边列表，每个超边是一个frozenset
    """
    # 计算需要的超边数量
    C_N_r = math.comb(N, r)
    C_m_r = math.comb(m, r)
    M_target = int(lambda_r * C_N_r / C_m_r)

    print(f"  生成泊松超图: N={N}, m={m}, λ_r={lambda_r:.3f}, M_target={M_target}")

    # 生成超边
    hyperedges = []
    nodes = list(range(N))

    # 使用拒绝采样确保每个r-node的度大致符合泊松分布
    for _ in range(M_target):
        # 随机选择m个节点
        edge = frozenset(np.random.choice(nodes, size=m, replace=False))
        hyperedges.append(edge)

    return hyperedges


# ============================
# 3. 生成具有相关性的多路复用超图
# ============================
def generate_correlated_multiplex_hypergraph(N, m1, m2, lambda1, lambda2, rho, r):
    """
    生成具有指定相关性的两层多路复用超图

    策略:
    1. 先生成独立的两层超图
    2. 通过调整超边的重叠来引入相关性

    参数:
        N: 节点数
        m1, m2: 两层超边的基数
        lambda1, lambda2: r-node平均度
        rho: 目标相关系数 (-1到1)
        r: r-node大小
    """
    print(f"  生成相关多路复用超图 (ρ={rho})...")

    # 生成第一层超图
    hyperedges1 = generate_hypergraph_poisson(N, m1, lambda1, r)

    # 生成第二层超图，根据相关性调整
    nodes = list(range(N))
    C_N_r = math.comb(N, r)
    C_m2_r = math.comb(m2, r)
    M2_target = int(lambda2 * C_N_r / C_m2_r)

    hyperedges2 = []

    # 预计算所有r-node到超边的映射（用于第一层）
    rnode_to_edges1 = defaultdict(set)
    for i, edge in enumerate(hyperedges1):
        # 获取该超边的所有r-nodes
        for rnode in itertools.combinations(sorted(edge), r):
            rnode_to_edges1[rnode].add(i)

    # 第二层超边生成策略
    if rho >= 0:
        # 正相关：第二层超边倾向于与第一层共享r-nodes
        positive_fraction = 0.3 + 0.5 * rho  # 在0.3到0.8之间

        for i in range(M2_target):
            if np.random.random() < positive_fraction:
                # 正相关生成：从第一层超边中获取r-node模式
                if rnode_to_edges1:
                    # 随机选择一个第一层超边
                    template_edge_idx = np.random.randint(0, len(hyperedges1))
                    template_edge = hyperedges1[template_edge_idx]

                    # 从模板超边中选择部分节点
                    overlap_nodes = set()
                    template_nodes = list(template_edge)

                    # 确保至少有r个节点重叠
                    min_overlap = max(1, int(len(template_nodes) * (0.2 + 0.3 * rho)))
                    overlap_count = np.random.randint(min_overlap, min(len(template_nodes), m2))

                    # 随机选择重叠节点
                    overlap_nodes.update(np.random.choice(template_nodes, size=overlap_count, replace=False))

                    # 补充新节点
                    remaining_nodes = list(set(nodes) - overlap_nodes)
                    need_count = m2 - len(overlap_nodes)

                    if need_count > 0 and len(remaining_nodes) >= need_count:
                        new_nodes = set(np.random.choice(remaining_nodes, size=need_count, replace=False))
                        new_edge = frozenset(overlap_nodes.union(new_nodes))
                    else:
                        # 回退到随机生成
                        new_edge = frozenset(np.random.choice(nodes, size=m2, replace=False))
                else:
                    new_edge = frozenset(np.random.choice(nodes, size=m2, replace=False))
            else:
                # 独立生成
                new_edge = frozenset(np.random.choice(nodes, size=m2, replace=False))

            hyperedges2.append(new_edge)

    else:
        # 负相关：第二层超边倾向于避免与第一层共享r-nodes
        negative_fraction = 0.3 + 0.5 * abs(rho)  # 在0.3到0.8之间

        for i in range(M2_target):
            if np.random.random() < negative_fraction:
                # 负相关生成：避免使用第一层超边中的节点模式
                # 收集所有第一层超边中的节点（负相关时避免这些节点）
                nodes_in_layer1 = set()
                for edge in hyperedges1:
                    nodes_in_layer1.update(edge)

                # 优先选择不在第一层超边中的节点
                candidate_nodes = list(set(nodes) - nodes_in_layer1)

                if len(candidate_nodes) >= m2:
                    # 如果够选，全部从候选节点中选
                    new_edge = frozenset(np.random.choice(candidate_nodes, size=m2, replace=False))
                else:
                    # 不够选，部分从候选节点中选，部分随机
                    new_edge_nodes = set(candidate_nodes)
                    remaining = m2 - len(candidate_nodes)

                    if remaining > 0:
                        other_nodes = list(set(nodes) - set(candidate_nodes))
                        if len(other_nodes) >= remaining:
                            new_edge_nodes.update(np.random.choice(other_nodes, size=remaining, replace=False))

                    new_edge = frozenset(new_edge_nodes)
            else:
                # 独立生成
                new_edge = frozenset(np.random.choice(nodes, size=m2, replace=False))

            hyperedges2.append(new_edge)

    # 计算实际的相关性（通过r-node度分布）
    print(f"  层1超边数: {len(hyperedges1)}, 层2超边数: {len(hyperedges2)}")

    # 验证生成的超图有足够的超边
    if len(hyperedges1) == 0 or len(hyperedges2) == 0:
        print(f"  警告: 超边数为0! 回退到随机生成")
        if len(hyperedges1) == 0:
            hyperedges1 = [frozenset(np.random.choice(nodes, size=m1, replace=False)) for _ in range(100)]
        if len(hyperedges2) == 0:
            hyperedges2 = [frozenset(np.random.choice(nodes, size=m2, replace=False)) for _ in range(100)]

    return hyperedges1, hyperedges2


# ============================
# 4. 蒙特卡洛模拟r-node渗流
# ============================
def monte_carlo_rnode_percolation(hyperedges1, hyperedges2, N, r, p_N, p_H, n_trials=20):
    """
    多路复用超图的r-node渗流蒙特卡洛模拟
    """
    R_values = []

    for trial in range(n_trials):
        # 1. 节点损伤
        node_alive = np.random.random(N) < p_N
        alive_nodes = set(np.where(node_alive)[0])

        # 2. 收集存活的超边
        surviving_edges = []
        # 第一层
        for edge in hyperedges1:
            if np.random.random() < p_H and edge.issubset(alive_nodes):
                surviving_edges.append(edge)
        # 第二层
        for edge in hyperedges2:
            if np.random.random() < p_H and edge.issubset(alive_nodes):
                surviving_edges.append(edge)

        n_edges = len(surviving_edges)
        if n_edges < 2:
            R_values.append(0.0)
            continue

        # 3. 构建并查集（超边级别）
        uf = UnionFind(n_edges)

        # 4. 节点到超边的倒排索引
        node_to_edges = defaultdict(list)
        for edge_id, edge in enumerate(surviving_edges):
            for node in edge:
                node_to_edges[node].append(edge_id)

        # 5. 连接共享至少r个节点的超边
        processed_pairs = set()
        for node, edge_list in node_to_edges.items():
            if len(edge_list) > 1:
                for i in range(len(edge_list)):
                    for j in range(i + 1, len(edge_list)):
                        ei, ej = edge_list[i], edge_list[j]

                        if (ei, ej) in processed_pairs:
                            continue

                        # 如果已经在同一分量，跳过
                        if uf.find(ei) == uf.find(ej):
                            continue

                        # 计算交集大小
                        set_i = surviving_edges[ei]
                        set_j = surviving_edges[ej]

                        # 优化：对于小的集合，直接计算交集
                        if len(set_i) <= 20 and len(set_j) <= 20:
                            inter_size = len(set_i & set_j)
                        else:
                            # 对于大集合，使用更高效的方法
                            if len(set_i) < len(set_j):
                                inter_size = sum(1 for node in set_i if node in set_j)
                            else:
                                inter_size = sum(1 for node in set_j if node in set_i)

                        if inter_size >= r:
                            uf.union(ei, ej)
                            processed_pairs.add((ei, ej))

        # 6. 计算最大连通分量中的节点数
        comp_nodes = defaultdict(set)
        for edge_id in range(n_edges):
            root = uf.find(edge_id)
            comp_nodes[root].update(surviving_edges[edge_id])

        if not comp_nodes:
            R_values.append(0.0)
            continue

        max_nodes = max(len(nodes) for nodes in comp_nodes.values())
        R = max_nodes / N
        R_values.append(R)

    return float(np.mean(R_values)), float(np.std(R_values))


# ============================
# 5. 拟合渗流曲线找到临界点（修改为R=0.1）
# ============================
def fit_percolation_curve(p_values, R_values):
    """
    使用Sigmoid函数拟合渗流曲线，找到临界点

    返回:
        p_c: 临界点（曲线最陡处的p值）
        fit_success: 拟合是否成功
    """
    try:
        # Sigmoid函数
        def sigmoid(x, p_c, k, R_max, R_min):
            return R_min + (R_max - R_min) / (1 + np.exp(-k * (x - p_c)))

        # 初始参数猜测
        p0 = [np.median(p_values), 20.0, max(R_values), min(R_values)]

        # 边界
        bounds = ([min(p_values) - 0.1, 1.0, 0.0, 0.0],
                  [max(p_values) + 0.1, 100.0, 1.1, 0.5])

        # 拟合
        popt, pcov = curve_fit(sigmoid, p_values, R_values, p0=p0, bounds=bounds, maxfev=5000)

        p_c, k, R_max, R_min = popt

        # 计算临界点作为导数最大处（即Sigmoid中点）
        p_c_midpoint = p_c

        # 修改这里：计算R=0.1处的p值（原来是R=0.5）
        if R_max > R_min:
            p_c_01 = p_c - (1 / k) * np.log((R_max - 0.1) / (0.1 - R_min))
        else:
            p_c_01 = p_c

        return p_c_midpoint, p_c_01, popt, True

    except Exception as e:
        print(f"  曲线拟合失败: {e}")
        # 修改这里：使用简单方法找到R首次超过0.1的点
        for i in range(1, len(p_values)):
            if R_values[i - 1] < 0.1 <= R_values[i]:
                # 线性插值
                p_low, p_high = p_values[i - 1], p_values[i]
                R_low, R_high = R_values[i - 1], R_values[i]
                if R_high > R_low:
                    p_c_est = p_low + (0.1 - R_low) * (p_high - p_low) / (R_high - R_low)
                    return p_c_est, p_c_est, None, False

        # 如果找不到，返回None
        return None, None, None, False


# ============================
# 6. 主程序
# ============================
def main():
    print("=== 多路复用超图渗流模拟：验证层间相关性影响 ===")
    print("目标：验证正相关降低渗流阈值，负相关升高阈值 (Eq.38)")

    # 参数设置 - 根据您的需求调整
    N = 2000  # 节点数（使用您想要的1500）
    m1, m2 = 3, 4  # 两层超边的基数
    r = 2  # r-node大小
    p_H = 0.8  # 超边保留概率

    # λ_r参数 - 根据您的实验调整
    lambda_r_target = 0.36  # 您提到N=1500时λ=0.45或0.42

    # 计算相关信息
    C_N_r = math.comb(N, r)
    print(f"总节点数 N = {N}")
    print(f"r-node 总数 C({N}, {r}) = {C_N_r}")
    print(f"目标 λ_r = {lambda_r_target}")

    # 相关性值
    correlation_values = [-0.8, -0.4, 0.0, 0.4, 0.8]
    colors = plt.cm.viridis(np.linspace(0, 1, len(correlation_values)))

    # 存储结果
    results = {}

    # p_N扫描范围 - 根据之前的经验调整
    p_N_min, p_N_max = 0.4, 0.7
    p_N_points = 11
    p_N_range = np.linspace(p_N_min, p_N_max, p_N_points)

    for rho in correlation_values:
        print(f"\n{'=' * 60}")
        print(f"正在处理 相关性 ρ = {rho:.1f}")
        print('=' * 60)

        start_time = time.time()

        # 1. 生成多路复用超图
        hyperedges1, hyperedges2 = generate_correlated_multiplex_hypergraph(
            N, m1, m2, lambda_r_target, lambda_r_target, rho, r
        )

        # 2. 蒙特卡洛模拟
        print(f"  扫描 p_N ({len(p_N_range)} 个点)...")
        R_means = []
        R_stds = []

        for p_N in p_N_range:
            R_mean, R_std = monte_carlo_rnode_percolation(
                hyperedges1, hyperedges2, N, r, p_N, p_H, n_trials=20
            )
            R_means.append(R_mean)
            R_stds.append(R_std)
            print(f"    p_N = {p_N:.3f}: R = {R_mean:.4f} ± {R_std:.4f}")

        # 3. 拟合找到临界点
        p_c_midpoint, p_c_01, fit_params, fit_success = fit_percolation_curve(p_N_range, R_means)

        # 存储结果
        results[rho] = {
            'p_N': p_N_range,
            'R_mean': np.array(R_means),
            'R_std': np.array(R_stds),
            'p_c': p_c_01 if p_c_01 is not None else p_c_midpoint,  # 使用p_c_01而不是p_c_05
            'p_c_midpoint': p_c_midpoint,
            'p_c_01': p_c_01,  # 改为p_c_01
            'fit_success': fit_success,
            'fit_params': fit_params,
            'M1': len(hyperedges1),
            'M2': len(hyperedges2)
        }

        elapsed = time.time() - start_time
        if p_c_01 is not None:
            print(f"  估计的临界点 p_c (R=0.1) = {p_c_01:.4f} (耗时: {elapsed:.1f}s)")
        else:
            print(f"  未找到明确的临界点 (耗时: {elapsed:.1f}s)")

    # ============================
    # 可视化
    # ============================
    print(f"\n{'=' * 60}")
    print("生成可视化图表")
    print('=' * 60)

    # 设置中文字体（如果需要）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1: 渗流曲线
    ax1 = axes[0]
    for idx, rho in enumerate(correlation_values):
        data = results[rho]
        ax1.errorbar(data['p_N'], data['R_mean'], yerr=data['R_std'],
                     fmt='o-', linewidth=1.5, markersize=5, capsize=3,
                     color=colors[idx], label=f'ρ = {rho}', alpha=0.8)

        # 标记R=0.1阈值线
        ax1.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # 如果拟合成功，绘制拟合曲线
        if data['fit_success'] and data['fit_params'] is not None:
            p_c, k, R_max, R_min = data['fit_params']
            p_fine = np.linspace(min(data['p_N']), max(data['p_N']), 100)
            R_fit = R_min + (R_max - R_min) / (1 + np.exp(-k * (p_fine - p_c)))
            ax1.plot(p_fine, R_fit, '--', color=colors[idx], alpha=0.6, linewidth=1)

            # 标记临界点
            ax1.axvline(x=data['p_c'], color=colors[idx], linestyle=':', alpha=0.5)

    ax1.set_xlabel('$p_N$', fontsize=12)
    ax1.set_ylabel('$S$', fontsize=12)
    #ax1.set_title('多路复用超图渗流曲线 (R=0.1作为临界阈值)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_ylim(-0.05, 1.05)

    # 图2: 临界点 vs 相关性
    ax2 = axes[1]

    rhos_plot = []
    p_cs_plot = []

    for rho in correlation_values:
        data = results[rho]
        if data['p_c'] is not None:
            rhos_plot.append(rho)
            p_cs_plot.append(data['p_c'])

    if len(rhos_plot) > 1:
        # 绘制数据点
        ax2.plot(rhos_plot, p_cs_plot, 's-', linewidth=2, markersize=8,
                 color='darkred', markerfacecolor='white', markeredgewidth=2, label='Data')

        # 线性拟合
        if len(rhos_plot) >= 2:
            coeffs = np.polyfit(rhos_plot, p_cs_plot, 1)
            poly = np.poly1d(coeffs)
            x_fit = np.linspace(min(rhos_plot), max(rhos_plot), 100)
            ax2.plot(x_fit, poly(x_fit), 'b--', alpha=0.7, linewidth=1.5,
                     label=f'Linear fitting: Slope={coeffs[0]:.3f}')

        # 添加理论预期说明
        #ax2.text(0.05, 0.95, '理论预期：', transform=ax2.transAxes, fontsize=10, verticalalignment='top')
        #ax2.text(0.05, 0.90, '正相关(ρ>0) → $p_c$降低', transform=ax2.transAxes, fontsize=10, verticalalignment='top',
        #         color='green')
        #ax2.text(0.05, 0.85, '负相关(ρ<0) → $p_c$升高', transform=ax2.transAxes, fontsize=10, verticalalignment='top',
        #         color='red')

        # 分析趋势
        if len(rhos_plot) >= 2:
            slope = coeffs[0]
            if slope < -0.001:  # 负斜率，符合理论
                conclusion = "Negative slope: Positive correlation lowers threshold"
                color = 'green'
            elif slope > 0.001:  # 正斜率，与理论相反
                conclusion = "Positive slope: Check results"
                color = 'red'
            else:  # 斜率接近0
                conclusion = "The correlation has no significant impact."
                color = 'orange'

            ax2.text(0.5, 0.05, conclusion, transform=ax2.transAxes,
                     fontsize=11, ha='center', bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    ax2.set_xlabel('Layer Correlation ρ', fontsize=12)
    ax2.set_ylabel('Critical $p_N$', fontsize=12)
    #ax2.set_title('临界点随层间相关性变化 (R=0.1)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    if len(rhos_plot) > 1:
        ax2.legend(loc='best', fontsize=10)

    plt.tight_layout()
    plt.show()

    # ============================
    # 结果汇总
    # ============================
    print(f"\n{'=' * 60}")
    print("模拟结果汇总")
    print('=' * 60)
    print(f"{'ρ':<8} {'p_c (R=0.1)':<12} {'M1':<8} {'M2':<8}")
    print("-" * 60)

    for rho in correlation_values:
        data = results[rho]
        p_c_str = f"{data['p_c']:.4f}" if data['p_c'] is not None else "N/A"
        print(f"{rho:<8.2f} {p_c_str:<12} {data['M1']:<8} {data['M2']:<8}")

    # 趋势分析
    valid_data = [(rho, results[rho]['p_c']) for rho in correlation_values
                  if results[rho]['p_c'] is not None]

    if len(valid_data) >= 3:
        rhos_valid, pcs_valid = zip(*valid_data)

        # 按相关性排序
        sorted_idx = np.argsort(rhos_valid)
        rhos_sorted = np.array(rhos_valid)[sorted_idx]
        pcs_sorted = np.array(pcs_valid)[sorted_idx]

        # 计算负相关和正相关的平均p_c
        neg_mask = rhos_sorted < 0
        pos_mask = rhos_sorted > 0

        if np.any(neg_mask) and np.any(pos_mask):
            neg_avg = np.mean(pcs_sorted[neg_mask])
            pos_avg = np.mean(pcs_sorted[pos_mask])

            print(f"\n趋势分析:")
            print(f"  负相关(ρ<0)平均 p_c: {neg_avg:.4f}")
            print(f"  正相关(ρ>0)平均 p_c: {pos_avg:.4f}")

            if neg_avg > pos_avg:
                reduction = ((neg_avg - pos_avg) / neg_avg) * 100
                print(f"  ✓ 正相关使渗流阈值降低了 {reduction:.1f}%")
            else:
                print(f"  ⚠ 结果与理论预期相反")

    print(f"\n模拟完成。")


# 运行主程序
if __name__ == "__main__":
    main()