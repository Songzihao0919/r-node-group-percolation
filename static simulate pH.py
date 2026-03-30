import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import math
import warnings
from collections import defaultdict, deque

warnings.filterwarnings('ignore')


# ============================
# 1. 生成随机超图模型的函数（优化版）
# ============================
def generate_random_hypergraph_ER_optimized(N, M, m_fixed, r):
    """
    生成随机超图，优化内存使用。
    不再存储所有可能的r-node组合，只记录出现在超边中的组合。
    """
    hyperedges = []

    for _ in range(M):
        # 随机选择m_fixed个不同的节点形成一条超边
        nodes_in_edge = np.random.choice(N, size=m_fixed, replace=False)
        edge_set = frozenset(nodes_in_edge)
        hyperedges.append(edge_set)

    return hyperedges


def calculate_r_node_co_degree_correct(hyperedges, N, r):
    """
    正确计算r-node共度分布，考虑所有可能的r-node组合，包括共度为0的组合。
    """
    # 使用字典记录每个r-node组合的共度
    kr_dict = defaultdict(int)

    # 遍历每条超边，更新r-node组合的计数
    for edge_set in hyperedges:
        if len(edge_set) >= r:
            # 生成当前超边中所有大小为r的子集
            for r_node in combinations(edge_set, r):
                r_node_key = tuple(sorted(r_node))  # 排序以确保元组唯一性
                kr_dict[r_node_key] += 1

    # 计算总r-node组合数
    total_r_nodes = math.comb(N, r)

    # 计算分布统计量
    if not kr_dict:
        # 如果没有r-node组合，返回退化分布
        return None, 0.0, 0.0, total_r_nodes

    # 计算每个共度值出现的次数
    max_kr = max(kr_dict.values())
    kr_counts = [0] * (max_kr + 1)

    for kr in kr_dict.values():
        kr_counts[kr] += 1

    # 计算共度为0的组合数
    kr_counts[0] = total_r_nodes - sum(kr_counts)

    # 转换为概率分布
    P_kr_array = np.array(kr_counts, dtype=float) / total_r_nodes

    # 计算均值和二阶矩
    k_vals = np.arange(len(P_kr_array))
    mean_kr = np.sum(k_vals * P_kr_array)
    mean_kr_sq_minus_kr = np.sum(k_vals * (k_vals - 1) * P_kr_array)

    return kr_dict, mean_kr, mean_kr_sq_minus_kr, total_r_nodes


# ============================
# 2. 计算理论临界阈值的函数 (Eq.6) - 固定p_N求p_H
# ============================
def theoretical_critical_pH_from_moments(mean_kr, mean_kr_sq_minus_kr, m_fixed, r, p_N_fixed):
    """
    直接从矩计算理论临界超边保留概率 p_H_c。
    """
    # 计算C(m, r) = 组合数 C(m, r)
    C_m_r = math.comb(m_fixed, r)

    # 公式中 <C(m,r)(C(m,r)-1)> / <C(m,r)> 项
    # 由于m固定，该项简化为 (C_m_r * (C_m_r - 1)) / C_m_r = C_m_r - 1
    m_term = C_m_r - 1

    # 计算 <k_r(k_r-1)>/<k_r>
    if mean_kr == 0:
        return None

    ratio = mean_kr_sq_minus_kr / mean_kr
    if ratio <= 0:
        return None

    # 公式(6)左侧等于1时临界，因此：
    # (p_N_fixed)**r * p_H_c * m_term * ratio = 1
    denominator = (p_N_fixed ** r) * m_term * ratio

    if denominator <= 0:
        return None

    p_H_c_theory = 1.0 / denominator

    # 确保结果在合理范围内
    if p_H_c_theory > 1.0:
        return 1.0
    elif p_H_c_theory < 0.0:
        return 0.0

    return p_H_c_theory


# ============================
# 3. 并查集(Union-Find)数据结构
# ============================
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n

    def find(self, x):
        # 路径压缩
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return

        # 按大小合并
        if self.size[root_x] < self.size[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x
        self.size[root_x] += self.size[root_y]
        self.count -= 1

    def get_component_sizes(self):
        # 返回每个连通分量的大小
        roots = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            roots[root] = roots.get(root, 0) + 1
        return list(roots.values())


# ============================
# 4. 使用正确连接规则的蒙特卡洛模拟（固定p_N，扫描p_H）
# ============================
def monte_carlo_percolation_fixed_pN_corrected(hyperedges, N, r, p_N_fixed, p_H, n_trials=10):
    """
    修正版蒙特卡洛模拟，使用正确的连接规则：
    两条超边共享至少r个节点则相连。
    固定p_N，扫描p_H。
    """
    M = len(hyperedges)
    R_trials = []

    # 预计算总r-node组合数（用于归一化R）
    total_r_nodes = math.comb(N, r)

    for trial in range(n_trials):
        # 步骤1: 随机损伤
        # 生成存活的节点（固定p_N）
        node_survives = np.random.rand(N) < p_N_fixed
        alive_nodes = set(np.where(node_survives)[0])

        # 随机选择保留的超边（根据当前p_H），并检查其中的节点是否都存活
        surviving_hyperedges = []
        surviving_hyperedge_nodes = []

        for idx, edge_set in enumerate(hyperedges):
            if np.random.rand() < p_H and edge_set.issubset(alive_nodes):
                surviving_hyperedges.append(idx)
                surviving_hyperedge_nodes.append(edge_set)

        num_surviving = len(surviving_hyperedges)

        # 如果没有存活的超边，R=0
        if num_surviving == 0:
            R_trials.append(0.0)
            continue

        # 步骤2: 在存活超边之间建立连接（如果共享至少r个节点）
        # 构建超边之间的连接图
        edge_graph = {i: set() for i in range(num_surviving)}

        # 为每个节点构建倒排索引：节点->包含它的超边列表
        node_to_edges = defaultdict(list)
        for edge_idx, edge_set in enumerate(surviving_hyperedge_nodes):
            for node in edge_set:
                node_to_edges[node].append(edge_idx)

        # 对于每个节点，连接共享它的所有超边
        for node, edge_list in node_to_edges.items():
            if len(edge_list) > 1:
                for i in range(len(edge_list)):
                    for j in range(i + 1, len(edge_list)):
                        edge_i = edge_list[i]
                        edge_j = edge_list[j]

                        # 检查这两条超边是否共享至少r个节点
                        if edge_j not in edge_graph[edge_i]:
                            intersection = surviving_hyperedge_nodes[edge_i].intersection(
                                surviving_hyperedge_nodes[edge_j])
                            if len(intersection) >= r:
                                edge_graph[edge_i].add(edge_j)
                                edge_graph[edge_j].add(edge_i)

        # 步骤3: 找到超边图中的最大连通分量
        visited = [False] * num_surviving
        max_component_size = 0
        max_component_nodes = set()

        for start in range(num_surviving):
            if not visited[start]:
                # BFS遍历
                queue = deque([start])
                component_edges = []
                component_nodes = set()

                while queue:
                    edge_idx = queue.popleft()
                    if not visited[edge_idx]:
                        visited[edge_idx] = True
                        component_edges.append(edge_idx)
                        component_nodes.update(surviving_hyperedge_nodes[edge_idx])

                        for neighbor in edge_graph[edge_idx]:
                            if not visited[neighbor]:
                                queue.append(neighbor)

                # 更新最大连通分量
                if len(component_edges) > max_component_size:
                    max_component_size = len(component_edges)
                    max_component_nodes = component_nodes

        # 步骤4: 计算R
        # R = 最大连通分量中的节点形成的所有r-node组合数 / 总r-node组合数
        num_nodes_in_giant = len(max_component_nodes)
        if num_nodes_in_giant < r:
            R = 0.0
        else:
            r_combinations_in_giant = math.comb(num_nodes_in_giant, r)
        R = r_combinations_in_giant / total_r_nodes
        #R = num_nodes_in_giant / N

        R_trials.append(R)

    R_mean = np.mean(R_trials)
    R_std = np.std(R_trials)
    return R_mean, R_std


# ============================
# 5. 主程序：执行参数扫描与验证（修正版）
# ============================
def main():
    # 设置模拟参数
    N = 10000  # 节点数
    m_fixed = 5  # 超边固定基数
    r = 2  # r-node的r值
    p_N_fixed = 0.9  # 固定的节点保留概率

    # 设置目标平均r-node共度λ_r
    target_lambda_r = 0.45

    # 根据目标λ_r计算M
    C_N_r = math.comb(N, r)
    C_m_r = math.comb(m_fixed, r)
    M = int(target_lambda_r * C_N_r / C_m_r)

    print("=== Simulation Parameters ===")
    print(f"Number of nodes N = {N:,}")
    print(f"Number of hyperedges M = {M:,} (calculated from λ_r={target_lambda_r})")
    print(f"Hyperedge fixed size m = {m_fixed}")
    print(f"r for r-node group = {r}")
    print(f"Fixed node retention probability p_N = {p_N_fixed}")
    print(f"Note: Using corrected connection rule: two hyperedges are connected if they share at least r nodes.")

    # 计算并显示实际λ_r
    actual_lambda_r = M * C_m_r / C_N_r
    print(f"Average r-node co-degree λ_r = M * C({m_fixed},{r}) / C({N},{r})")
    print(f"  = {M:,} * {C_m_r} / {C_N_r:,} = {actual_lambda_r:.6f}")

    print("\nGenerating random hypergraph...")

    # 生成随机超图（优化版）
    hyperedges = generate_random_hypergraph_ER_optimized(N, M, m_fixed, r)

    # 计算r-node共度统计量（修正版）
    print("Calculating r-node co-degree statistics (corrected method)...")
    result = calculate_r_node_co_degree_correct(hyperedges, N, r)

    if result[0] is None:
        print("Error: No r-node groups found in the hypergraph.")
        return

    kr_dict, mean_kr, mean_kr_sq_minus_kr, total_r_nodes = result

    print(f"  Active r-node groups: {len(kr_dict):,} (out of {total_r_nodes:,} possible)")
    print(f"  Mean co-degree <k_r> = {mean_kr:.6f}")
    print(f"  <k_r(k_r-1)> = {mean_kr_sq_minus_kr:.6f}")

    # 计算理论临界点 p_H_c
    print("\nCalculating theoretical critical threshold...")
    p_H_c_theory = theoretical_critical_pH_from_moments(mean_kr, mean_kr_sq_minus_kr, m_fixed, r, p_N_fixed)

    if p_H_c_theory is None:
        print("Failed to compute theoretical critical point.")
        return

    print(f"Theoretical critical p_H_c = {p_H_c_theory:.6f}")

    # 根据理论值设置p_H扫描范围
    #if p_H_c_theory > 0.9 or p_H_c_theory < 0.1:
    #    p_H_min, p_H_max = 0.1, 0.9
    #else:
        # 以理论值为中心，扩展±0.3的范围
    p_H_min =0 #max(0.05, p_H_c_theory - 0.3)
    p_H_max =1 #min(0.95, p_H_c_theory + 0.3)

    # 设置扫描点：每隔0.05取一个点
    n_points = 11 #int((p_N_max - p_N_min) / 0.05) + 1
    p_H_values = np.linspace(p_H_min, p_H_max, n_points)

    print(f"\nScanning p_H from {p_H_min:.3f} to {p_H_max:.3f} ({n_points} points)...")
    print("This will take some time due to the large network size...")

    R_means = []
    R_stds = []

    # 适当减少试验次数以加快计算
    n_trials = 8

    for i, p_H in enumerate(p_H_values):
        print(f"  Progress: {i + 1}/{len(p_H_values)} - p_H = {p_H:.3f}...", end="")
        R_mean, R_std = monte_carlo_percolation_fixed_pN_corrected(hyperedges, N, r, p_N_fixed, p_H, n_trials=n_trials)
        R_means.append(R_mean)
        R_stds.append(R_std)
        print(f" R = {R_mean:.6f} ± {R_std:.6f}")

    # 从模拟数据中估计临界点 - 使用更合理的阈值0.1
    threshold_R = 0.1
    sim_critical_idx = None
    for i, R in enumerate(R_means):
        if R > threshold_R:
            sim_critical_idx = i
            break

    if sim_critical_idx is not None and sim_critical_idx > 0:
        # 线性插值以获得更精确的估计
        p_low = p_H_values[sim_critical_idx - 1]
        p_high = p_H_values[sim_critical_idx]
        R_low = R_means[sim_critical_idx - 1]
        R_high = R_means[sim_critical_idx]

        if abs(R_high - R_low) > 1e-12:
            p_H_c_sim = p_low + (threshold_R - R_low) * (p_high - p_low) / (R_high - R_low)
        else:
            p_H_c_sim = p_high

        print(f"\nSimulated critical p_H_c (R > {threshold_R}) ≈ {p_H_c_sim:.6f}")

        if p_H_c_theory is not None:
            error_percent = abs(p_H_c_sim - p_H_c_theory) / p_H_c_theory * 100
            print(f"Relative error to theory: {error_percent:.2f}%")
    else:
        print(f"\nNo clear percolation transition observed in the scanned p_H range (R < {threshold_R}).")
        print("Try adjusting the p_H range or increasing network density.")
        p_H_c_sim = None

    # ============================
    # 6. 可视化结果
    # ============================
    plt.figure(figsize=(12, 7))

    # 绘制R vs p_H曲线
    plt.errorbar(p_H_values, R_means, yerr=R_stds, fmt='o-', capsize=5,
                 linewidth=2, markersize=6,
                 label=f'Monte Carlo simulation (N={N:,}, M={M:,})', color='royalblue')

    # 标记理论临界点
    if p_H_c_theory is not None:
        plt.axvline(x=p_H_c_theory, color='red', linestyle='--', linewidth=2,
                    label=f'Theoretical: $p_{{H_c}}$ = {p_H_c_theory:.4f}')

    # 标记模拟估计的临界点
    if p_H_c_sim is not None:
        plt.axvline(x=p_H_c_sim, color='green', linestyle='-.', linewidth=2,
                    label=f'Simulated: $p_{{H_c}}$ = {p_H_c_sim:.4f}')

    plt.xlabel('$p_H$', fontsize=14) #Hyperedge Retention Probability
    plt.ylabel('$R$', fontsize=14) #r-node Group Giant Component
    #plt.title(f'r-node Group Percolation on Static Hypergraph (r={r}, m={m_fixed}, fixed $p_N$={p_N_fixed})',
    #          fontsize=15)
    # 主标题
    #main_title = f'r-node Group Percolation (Fixed $p_N$={p_N_fixed})'
    #plt.title(main_title, fontsize=16, pad=20)

    # 参数信息
    param_text = f'$r$={r}, $m$={m_fixed}, fixed $p_N$={p_N_fixed}'
    plt.text(0.6, 0.97, param_text, transform=plt.gca().transAxes,
             fontsize=12, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.grid(True, alpha=0.3)#, linestyle='--'
    plt.legend(fontsize=12, loc='upper left')
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)
    #plt.xlim(min(p_H_values) - 0.02, max(p_H_values) + 0.02)
    plt.tight_layout()
    # 显示理论公式
    eq_text = r'Theoretical condition: $(p_N)^r p_H \frac{\langle \binom{m}{r}(\binom{m}{r}-1) \rangle}{\langle \binom{m}{r} \rangle} \frac{\langle k_r(k_r-1) \rangle}{\langle k_r \rangle} = 1$'
    plt.figtext(0.28, 0.78, eq_text, ha='center', fontsize=14, style='italic')

    #plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

    # 打印详细计算结果
    print("\n=== Detailed Calculation Results ===")
    C_m_r = math.comb(m_fixed, r)
    m_term = C_m_r - 1
    ratio = mean_kr_sq_minus_kr / mean_kr
    denominator = (p_N_fixed ** r) * m_term * ratio

    print(f"  C({m_fixed},{r}) = {C_m_r}")
    print(f"  m_term = C({m_fixed},{r}) - 1 = {m_term}")
    print(f"  (p_N)^r = ({p_N_fixed})^{r} = {p_N_fixed ** r:.6f}")
    print(f"  <k_r> = {mean_kr:.6f}")
    print(f"  <k_r(k_r-1)> = {mean_kr_sq_minus_kr:.6f}")
    print(f"  <k_r(k_r-1)> / <k_r> = {ratio:.6f}")
    print(f"  Denominator = (p_N)^r × m_term × (<k_r(k_r-1)>/<k_r>) = {denominator:.6f}")
    print(f"  p_H_c = 1 / {denominator:.6f} = {1 / denominator:.6f}")


# 运行主程序
if __name__ == "__main__":
    print("Corrected r-node percolation simulation (fixed p_N, scan p_H)")
    print("=" * 60)
    print("Key corrections in this version:")
    print("  1. Correct connection rule: Two hyperedges are connected if they share at least r nodes")
    print("  2. Correct r-node co-degree calculation including zero-degree combinations")
    print("  3. More reasonable percolation threshold: R > 0.1 (instead of 0.005)")
    print("  4. Optimized implementation for large networks (N=10000)")
    print("=" * 60)

    main()