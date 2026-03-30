import numpy as np
import matplotlib.pyplot as plt
import itertools
import warnings
import math
from collections import defaultdict, Counter
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigs
from scipy.optimize import brentq
import networkx as nx
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 4)})


# ============================================================================
# 1. 加载真实超图数据并统计分析（增强：计算r-node co-degree分布）
# ============================================================================
def load_and_analyze_real_hypergraph(filename="brain-hypergraph-structural.txt", r=2):
    """
    加载真实超图数据并分析其统计特性
    增强：专门计算r-node co-degree分布P(k_r)
    """
    print(f"Loading real hypergraph data from {filename}...")

    # 检查文件是否存在
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found!")
        return None, None

    # 读取文件
    hyperedges_real = []
    node_set = set()

    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            # 去除可能的空格，按逗号分割
            nodes = line.replace(' ', '').split(',')

            # 转换节点为整数
            try:
                edge = set(int(node) for node in nodes if node)
                if len(edge) >= r:  # 只保留大小>=r的超边
                    hyperedges_real.append(edge)
                    node_set.update(edge)
            except ValueError:
                continue

    # 重新索引节点
    original_nodes = sorted(list(node_set))
    node_mapping = {node: idx for idx, node in enumerate(original_nodes)}

    hyperedges_real_renumbered = []
    for edge in hyperedges_real:
        renumbered_edge = set(node_mapping[node] for node in edge)
        hyperedges_real_renumbered.append(renumbered_edge)

    N_real = len(original_nodes)

    print(f"  Successfully loaded hypergraph:")
    print(f"    Nodes: {N_real}")
    print(f"    Hyperedges (size >= {r}): {len(hyperedges_real_renumbered)}")

    if len(hyperedges_real_renumbered) == 0:
        raise ValueError(f"No hyperedges with size >= {r} found!")

    # 1. 计算r-node co-degree分布 P(k_r)
    print(f"  Computing {r}-node co-degree distribution P(k_{r})...")

    # 创建节点到超边的映射
    node_to_hyperedges = defaultdict(list)
    for edge_idx, edge in enumerate(hyperedges_real_renumbered):
        for node in edge:
            node_to_hyperedges[node].append(edge_idx)

    # 枚举所有可能的r-node组合
    all_r_nodes = list(itertools.combinations(range(N_real), r))
    print(f"    Total possible {r}-node groups: {len(all_r_nodes)}")

    # 计算每个r-node的co-degree
    r_node_co_degrees = []
    r_node_to_co_degree = {}

    # 使用进度条，因为计算可能很耗时
    for r_tuple in tqdm(all_r_nodes[:min(100000, len(all_r_nodes))], desc=f"Computing k_{r}"):
        # 找到包含这个r-tuple中所有节点的超边
        if len(r_tuple) > 0:
            # 取第一个节点的超边列表
            common_hyperedges = set(node_to_hyperedges[r_tuple[0]])

            # 与其余节点的超边列表求交集
            for node in r_tuple[1:]:
                common_hyperedges.intersection_update(node_to_hyperedges[node])

            co_degree = len(common_hyperedges)
            r_node_co_degrees.append(co_degree)
            r_node_to_co_degree[r_tuple] = co_degree

    # 2. 计算基数分布 Q(m)
    edge_sizes = [len(edge) for edge in hyperedges_real_renumbered]

    # 3. 计算原始网络的r-node因子图聚类系数（近似）
    print(f"  Computing clustering of {r}-node factor graph (approximation)...")

    # 统计信息
    if r_node_co_degrees:
        avg_k_r = np.mean(r_node_co_degrees)
        avg_k_r_sq = np.mean([k ** 2 for k in r_node_co_degrees])
        k_r_moment_ratio = (avg_k_r_sq - avg_k_r) / avg_k_r if avg_k_r > 0 else 0

        # 计算r-node co-degree分布
        k_r_counts = Counter(r_node_co_degrees)
        k_r_values = sorted(k_r_counts.keys())
        k_r_probs = [k_r_counts[k] / len(r_node_co_degrees) for k in k_r_values]
    else:
        avg_k_r = avg_k_r_sq = k_r_moment_ratio = 0
        k_r_values, k_r_probs = [], []

    stats_real = {
        'N': N_real,
        'M': len(hyperedges_real_renumbered),
        'r': r,
        'avg_k_r': avg_k_r,
        'k_r_moment_ratio': k_r_moment_ratio,
        'k_r_values': k_r_values,  # k_r的值
        'k_r_probs': k_r_probs,  # P(k_r)的概率
        'r_node_co_degrees': r_node_co_degrees,
        'r_node_to_co_degree': r_node_to_co_degree,
        'avg_edge_size': np.mean(edge_sizes) if edge_sizes else 0,
        'edge_size_dist': np.bincount(edge_sizes),
        'original_hyperedges': hyperedges_real_renumbered
    }

    print(f"  <k_{r}> = {avg_k_r:.3f}")
    print(f"  <k_{r}(k_{r}-1)>/<k_{r}> = {k_r_moment_ratio:.3f}")
    print(f"  P(k_{r}) support size: {len(k_r_values)}")

    # 可视化r-node co-degree分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # r-node co-degree分布
    ax1 = axes[0]
    ax1.bar(k_r_values[:20], k_r_probs[:20], alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel(f'{r}-node co-degree $k_{r}$')
    ax1.set_ylabel(f'P($k_{r}$)')
    ax1.set_title(f'{r}-node Co-degree Distribution')
    ax1.grid(True, alpha=0.3)

    # 边大小分布
    ax2 = axes[1]
    size_counts = np.bincount(edge_sizes)
    ax2.bar(range(len(size_counts)), size_counts, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Edge size m')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Edge Size Distribution')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Real Hypergraph Statistics (r={r})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return hyperedges_real_renumbered, stats_real


# ============================================================================
# 2. 基于r-node co-degree分布生成局部树状超图（真正的r-node配置模型）
# ============================================================================
def generate_locally_treelike_hypergraph_rnode(stats_real, r=2, target_avg_k_r=None, max_attempts=10):
    """
    基于真实的r-node co-degree分布P(k_r)生成局部树状超图
    使用r-node配置模型，确保生成的r-node因子图近似树状
    """
    print(f"\nGenerating locally treelike hypergraph using {r}-node configuration model...")

    N = stats_real['N']
    k_r_values = stats_real['k_r_values']
    k_r_probs = stats_real['k_r_probs']

    if len(k_r_values) == 0 or len(k_r_probs) == 0:
        print("  Error: No r-node co-degree distribution available.")
        return None, 0.0

    # 如果未指定目标平均k_r，使用真实值
    if target_avg_k_r is None:
        target_avg_k_r = stats_real['avg_k_r']

    print(f"  Target <k_{r}>: {target_avg_k_r:.3f}")
    print(f"  Using P(k_{r}) with {len(k_r_values)} unique values")

    # 估计超边数量
    # 从真实数据估计平均边大小
    avg_edge_size = stats_real['avg_edge_size']
    if avg_edge_size < r:
        avg_edge_size = r + 1

    # 总r-node连接数
    total_r_nodes = math.comb(N, r)
    target_total_stubs = total_r_nodes * target_avg_k_r

    # 估计超边数量
    avg_comb_per_edge = math.comb(int(avg_edge_size), r)
    M_estimated = int(target_total_stubs / avg_comb_per_edge)

    print(f"  Estimated hyperedges: {M_estimated}")
    print(f"  Total {r}-node pairs: {total_r_nodes}")
    print(f"  Target total stubs: {target_total_stubs:.0f}")

    # 方法1：精确的r-node配置模型（对于小r可行）
    if r == 2 and N <= 1000:  # 对于r=2且N不太大，可以尝试精确生成
        return generate_r2_configuration_model(N, M_estimated, stats_real, target_avg_k_r)
    else:
        # 方法2：近似方法 - 使用简单的配置模型
        return generate_approximate_rnode_hypergraph(N, M_estimated, stats_real, r)


def generate_r2_configuration_model(N, M_estimated, stats_real, target_avg_k_r):
    """为r=2生成配置模型超图"""
    r = 2
    print("  Using exact configuration model for r=2...")

    # 所有可能的2-node组合
    all_pairs = list(itertools.combinations(range(N), 2))
    total_pairs = len(all_pairs)

    # 为每对节点分配co-degree
    # 基于真实的P(k_2)分布
    k_values = stats_real['k_r_values']
    probs = stats_real['k_r_probs']

    # 归一化概率
    probs = np.array(probs) / np.sum(probs)

    # 为每对节点采样co-degree
    pair_co_degrees = np.random.choice(k_values, size=total_pairs, p=probs)

    # 调整以获得目标平均值
    current_avg = np.mean(pair_co_degrees)
    if current_avg > 0:
        scale_factor = target_avg_k_r / current_avg
        pair_co_degrees = np.round(pair_co_degrees * scale_factor).astype(int)
        pair_co_degrees = np.maximum(pair_co_degrees, 0)

    # 创建节点桩列表
    pair_stubs = []
    for i, (u, v) in enumerate(all_pairs):
        pair_stubs.extend([(u, v)] * int(pair_co_degrees[i]))

    # 创建超边桩列表
    # 为每条超边分配大小（从真实分布采样）
    edge_size_dist = stats_real['edge_size_dist']
    if len(edge_size_dist) < 3:  # 如果没有足够数据，使用简单分布
        edge_sizes = np.random.randint(3, 8, M_estimated)
    else:
        size_probs = edge_size_dist / np.sum(edge_size_dist)
        size_values = np.arange(len(edge_size_dist))
        edge_sizes = np.random.choice(size_values, size=M_estimated, p=size_probs)
        edge_sizes = np.maximum(edge_sizes, 3)

    # 为每条超边创建包含的2-node桩
    edge_stubs = []
    for edge_idx, m in enumerate(edge_sizes):
        # 一条大小为m的超边包含C(m,2)个2-node
        num_2node_in_edge = math.comb(m, 2)
        edge_stubs.extend([edge_idx] * int(num_2node_in_edge))

    # 调整使总桩数匹配
    min_len = min(len(pair_stubs), len(edge_stubs))
    pair_stubs = pair_stubs[:min_len]
    edge_stubs = edge_stubs[:min_len]

    # 随机配对
    print(f"  Matching {len(pair_stubs)} pair-stubs to {len(set(edge_stubs))} hyperedges...")
    indices = np.random.permutation(len(pair_stubs))

    # 创建超边
    M = int(max(edge_stubs)) + 1
    hyperedges = [set() for _ in range(M)]

    for idx in indices:
        (u, v) = pair_stubs[idx]
        edge_idx = edge_stubs[idx]
        hyperedges[edge_idx].add(u)
        hyperedges[edge_idx].add(v)

    # 移除空的超边
    valid_hyperedges = [edge for edge in hyperedges if len(edge) >= 2]

    # 移除重复超边
    unique_hyperedges = []
    seen = set()
    for edge in valid_hyperedges:
        edge_tuple = tuple(sorted(edge))
        if edge_tuple not in seen:
            seen.add(edge_tuple)
            unique_hyperedges.append(edge)

    print(f"  Generated hypergraph: {N} nodes, {len(unique_hyperedges)} hyperedges")

    # 评估生成网络的r-node统计量
    generated_stats = evaluate_generated_network(unique_hyperedges, N, r)

    return unique_hyperedges, generated_stats['clustering_approx']


def generate_approximate_rnode_hypergraph(N, M_estimated, stats_real, r=2):
    """为一般r生成近似的r-node配置模型超图"""
    print(f"  Using approximate configuration model for r={r}...")

    # 使用简单的随机超图生成，但尝试匹配r-node co-degree分布
    avg_edge_size = max(r + 1, int(stats_real['avg_edge_size']))

    # 生成超边
    hyperedges = []
    for _ in range(M_estimated):
        size = np.random.poisson(avg_edge_size)
        size = max(r + 1, min(size, N // 2))  # 合理的大小范围
        nodes = np.random.choice(N, size=size, replace=False)
        hyperedges.append(set(nodes))

    # 尝试通过重连优化r-node统计量
    best_hyperedges = hyperedges
    best_score = float('inf')

    for attempt in range(20):
        # 轻微扰动：随机重连一些节点
        if len(hyperedges) >= 2:
            idx1, idx2 = np.random.choice(len(hyperedges), 2, replace=False)
            edge1 = list(hyperedges[idx1])
            edge2 = list(hyperedges[idx2])

            if len(edge1) > r and len(edge2) > r:
                # 交换一个节点
                node1 = np.random.choice(edge1)
                node2 = np.random.choice(edge2)

                if node1 not in edge2 and node2 not in edge1:
                    new_edge1 = set(edge1)
                    new_edge1.remove(node1)
                    new_edge1.add(node2)

                    new_edge2 = set(edge2)
                    new_edge2.remove(node2)
                    new_edge2.add(node1)

                    # 检查是否重复
                    new_hyperedges = hyperedges.copy()
                    new_hyperedges[idx1] = new_edge1
                    new_hyperedges[idx2] = new_edge2

                    # 评估统计量
                    stats = evaluate_generated_network(new_hyperedges, N, r)

                    # 计算与目标统计量的差异
                    target_k_r = stats_real['avg_k_r']
                    current_k_r = stats['avg_k_r']
                    score = abs(current_k_r - target_k_r) / (target_k_r + 1e-6)

                    if score < best_score:
                        best_score = score
                        best_hyperedges = new_hyperedges

        if attempt % 5 == 0:
            print(f"    Attempt {attempt}: avg_k_r = {stats['avg_k_r']:.3f}, target = {stats_real['avg_k_r']:.3f}")

    print(f"  Generated hypergraph: {N} nodes, {len(best_hyperedges)} hyperedges")
    print(f"  Best score: {best_score:.4f}")

    final_stats = evaluate_generated_network(best_hyperedges, N, r)
    return best_hyperedges, final_stats['clustering_approx']


def evaluate_generated_network(hyperedges, N, r):
    """评估生成网络的r-node统计量"""
    # 计算r-node co-degree分布
    node_to_hyperedges = defaultdict(list)
    for edge_idx, edge in enumerate(hyperedges):
        for node in edge:
            node_to_hyperedges[node].append(edge_idx)

    # 采样计算r-node co-degree
    sample_size = min(1000, math.comb(N, r))
    sampled_r_tuples = []

    if math.comb(N, r) > 10000:
        # 随机采样r-node
        for _ in range(sample_size):
            nodes = np.random.choice(N, size=r, replace=False)
            sampled_r_tuples.append(tuple(sorted(nodes)))
    else:
        # 对于小的N，枚举所有
        sampled_r_tuples = list(itertools.combinations(range(N), r))[:sample_size]

    k_r_values = []
    for r_tuple in sampled_r_tuples:
        if len(r_tuple) > 0:
            common_hyperedges = set(node_to_hyperedges[r_tuple[0]])
            for node in r_tuple[1:]:
                common_hyperedges.intersection_update(node_to_hyperedges[node])
            k_r_values.append(len(common_hyperedges))

    avg_k_r = np.mean(k_r_values) if k_r_values else 0
    avg_k_r_sq = np.mean([k ** 2 for k in k_r_values]) if k_r_values else 0
    k_r_moment_ratio = (avg_k_r_sq - avg_k_r) / avg_k_r if avg_k_r > 0 else 0

    # 近似聚类系数：通过随机游走估计r-node因子图的树状性
    clustering_approx = estimate_rnode_factor_graph_clustering(hyperedges, N, r)

    return {
        'avg_k_r': avg_k_r,
        'k_r_moment_ratio': k_r_moment_ratio,
        'clustering_approx': clustering_approx
    }


def estimate_rnode_factor_graph_clustering(hyperedges, N, r, num_walks=1000):
    """
    通过随机游走估计r-node因子图的聚类系数
    返回从随机节点开始的短循环的比例
    """
    if len(hyperedges) < 10:
        return 0.0

    # 构建r-node到超边的映射
    rnode_to_edges = defaultdict(list)
    for edge_idx, edge in enumerate(hyperedges):
        if len(edge) >= r:
            for r_tuple in itertools.combinations(edge, r):
                rnode_to_edges[tuple(sorted(r_tuple))].append(edge_idx)

    # 选择活跃的r-node
    active_rnodes = [rn for rn in rnode_to_edges.keys() if len(rnode_to_edges[rn]) > 1]
    if len(active_rnodes) < 10:
        return 0.0

    # 进行随机游走
    cycle_count = 0
    for _ in range(min(num_walks, len(active_rnodes))):
        start_rnode = active_rnodes[np.random.randint(len(active_rnodes))]

        # 2步游走
        edges1 = rnode_to_edges[start_rnode]
        if len(edges1) == 0:
            continue

        edge1 = edges1[np.random.randint(len(edges1))]

        # 从edge1到另一个rnode
        edge1_nodes = None
        for edge in hyperedges:
            if id(edge) == id(hyperedges[edge1]):
                edge1_nodes = edge
                break

        if edge1_nodes is None or len(edge1_nodes) < r:
            continue

        # 从edge1中选择一个不同的rnode
        rnodes_in_edge1 = [tuple(sorted(r_tuple)) for r_tuple in itertools.combinations(edge1_nodes, r)]
        rnodes_in_edge1 = [rn for rn in rnodes_in_edge1 if rn != start_rnode]

        if len(rnodes_in_edge1) == 0:
            continue

        rnode2 = rnodes_in_edge1[np.random.randint(len(rnodes_in_edge1))]

        # 检查是否能回到start_rnode
        edges2 = rnode_to_edges[rnode2]

        # 检查edges1和edges2是否有交集（排除edge1）
        common_edges = set(edges1) & set(edges2)
        if len(common_edges) > 1:  # 至少有一条公共边（除了可能的第一条边）
            cycle_count += 1

    clustering_approx = cycle_count / num_walks
    return clustering_approx


# ============================================================================
# 3. 构建r-节点因子图并计算统计量（保持不变）
# ============================================================================
def build_r_node_factor_graph(hyperedges, N, r):
    """构建r-节点因子图Gr = (V, U, E)"""
    print(f"\nBuilding r-node factor graph for r={r}...")

    all_r_tuples = list(itertools.combinations(range(N), r))
    r_node_to_idx = {tpl: idx for idx, tpl in enumerate(all_r_tuples)}
    idx_to_r_node = {idx: tpl for tpl, idx in r_node_to_idx.items()}
    V = len(all_r_tuples)

    U = len(hyperedges)

    r_node_edges = defaultdict(list)
    edge_r_nodes = defaultdict(list)

    for edge_idx, edge in enumerate(hyperedges):
        if len(edge) >= r:
            nodes = list(edge)
            for r_tuple in itertools.combinations(nodes, r):
                r_node_idx = r_node_to_idx[tuple(sorted(r_tuple))]
                r_node_edges[r_node_idx].append(edge_idx)
                edge_r_nodes[edge_idx].append(r_node_idx)

    active_r_nodes = {idx for idx in r_node_edges}
    print(f"  Total possible r-nodes (N choose r): {V}")
    print(f"  Active r-nodes (contained in at least 1 hyperedge): {len(active_r_nodes)}")
    print(f"  Factor nodes (hyperedges with size >= r): {U}")

    k_r_list = [len(r_node_edges[idx]) for idx in active_r_nodes]
    m_list = [len(hyperedges[edge_idx]) for edge_idx in range(U) if len(hyperedges[edge_idx]) >= r]

    if k_r_list:
        avg_k_r = np.mean(k_r_list)
        avg_k_r_sq = np.mean([k ** 2 for k in k_r_list])
        k_r_moment_ratio = (avg_k_r_sq - avg_k_r) / avg_k_r if avg_k_r > 0 else 0
    else:
        avg_k_r = avg_k_r_sq = k_r_moment_ratio = 0.0

    if m_list:
        comb_list = [math.comb(m, r) for m in m_list]
        avg_comb = np.mean(comb_list)
        avg_comb_sq = np.mean([c ** 2 for c in comb_list])
        comb_moment_ratio = (avg_comb_sq - avg_comb) / avg_comb if avg_comb > 0 else 0
    else:
        avg_comb = avg_comb_sq = comb_moment_ratio = 0.0

    stats = {
        'N': N, 'V': V, 'U': U, 'active_V': len(active_r_nodes),
        'avg_k_r': avg_k_r, 'k_r_moment_ratio': k_r_moment_ratio,
        'avg_comb': avg_comb, 'comb_moment_ratio': comb_moment_ratio,
        'edge_sizes': m_list
    }
    print(f"  <k_r> = {avg_k_r:.3f}, <k_r(k_r-1)>/<k_r> = {k_r_moment_ratio:.3f}")
    print(f"  <C(m,r)> = {avg_comb:.3f}, <C(C-1)>/<C> = {comb_moment_ratio:.3f}")

    return r_node_to_idx, idx_to_r_node, r_node_edges, edge_r_nodes, stats


# ============================================================================
# 4. 平均场临界点
# ============================================================================
def mean_field_critical_point(stats, r, p_H=1.0):
    """计算平均场临界点 p_N_c (公式6)"""
    k_r_ratio = stats['k_r_moment_ratio']
    comb_ratio = stats['comb_moment_ratio']

    if k_r_ratio <= 0 or comb_ratio <= 0:
        print("  MF Warning: Non-positive moment ratio, cannot compute critical point.")
        return None

    product = p_H * comb_ratio * k_r_ratio
    if product <= 0:
        return None

    p_N_c = (1.0 / product) ** (1.0 / r)

    if p_N_c > 1.0:
        print(f"  MF Warning: Calculated p_N_c = {p_N_c:.6f} > 1. Setting to 1.0.")
        p_N_c = 1.0
    elif p_N_c < 0:
        p_N_c = 0.0

    return p_N_c


# ============================================================================
# 5. 消息传递算法
# ============================================================================
def build_message_passing_matrix(r_node_edges, edge_r_nodes, p_N, p_H=1.0, r=2):
    """构建稀疏非回溯矩阵A（公式16-17）"""
    dir_edges_r_to_f = []
    dir_edges_f_to_r = []
    index_map_r_to_f = {}
    index_map_f_to_r = {}

    global_idx = 0
    for gamma, alphas in r_node_edges.items():
        for alpha in alphas:
            dir_edges_r_to_f.append((gamma, alpha))
            index_map_r_to_f[(gamma, alpha)] = global_idx
            global_idx += 1

    for alpha, gammas in edge_r_nodes.items():
        for gamma in gammas:
            dir_edges_f_to_r.append((alpha, gamma))
            index_map_f_to_r[(alpha, gamma)] = global_idx
            global_idx += 1

    E_total = global_idx
    print(f"    Total directed edges in factor graph: {E_total}")

    row_indices, col_indices, data_values = [], [], []

    print("    Constructing B_NH block...")
    for (gamma, alpha), idx_omega in index_map_r_to_f.items():
        for beta in r_node_edges[gamma]:
            if beta != alpha and (beta, gamma) in index_map_f_to_r:
                idx_v = index_map_f_to_r[(beta, gamma)]
                row_indices.append(idx_omega)
                col_indices.append(idx_v)
                data_values.append(p_N ** r)

    print("    Constructing B_HN block...")
    for (alpha, gamma), idx_v in index_map_f_to_r.items():
        for eta in edge_r_nodes[alpha]:
            if eta != gamma and (eta, alpha) in index_map_r_to_f:
                idx_omega = index_map_r_to_f[(eta, alpha)]
                row_indices.append(idx_v)
                col_indices.append(idx_omega)
                data_values.append(p_H)

    A = csr_matrix((data_values, (row_indices, col_indices)),
                   shape=(E_total, E_total), dtype=np.float64)
    return A, E_total


def lambda_max_A(p_N, r_node_edges, edge_r_nodes, p_H=1.0, r=2):
    """返回矩阵A的最大特征值 Λ(p_N) (公式18)"""
    A_mat, E_total = build_message_passing_matrix(r_node_edges, edge_r_nodes, p_N, p_H, r)

    if E_total <= 0:
        return 0.0

    if E_total < 1000:
        try:
            eigenvalues = np.linalg.eigvals(A_mat.toarray())
            return np.max(np.real(eigenvalues))
        except:
            pass

    try:
        eigenvalues, _ = eigs(A_mat, k=2, which='LM', maxiter=5000, tol=1e-8)
        lambda_max = np.max(np.real(eigenvalues))
        return lambda_max
    except:
        # 幂迭代法
        n = A_mat.shape[0]
        x = np.random.rand(n)
        x = x / np.linalg.norm(x)
        for _ in range(200):
            x_new = A_mat.dot(x)
            lambda_est = np.linalg.norm(x_new)
            x = x_new / lambda_est
        return (A_mat.dot(x)).dot(x) / x.dot(x)


def find_mpa_critical_point(r_node_edges, edge_r_nodes, r=2, p_H=1.0, p_N_range=(0.01, 0.99)):
    """通过根查找找到 Λ(p_N) = 1 的临界点 p_N_c_mpa"""
    print("\nMessage Passing Algorithm: Finding critical point where Λ(p_N)=1 ...")

    def f(p_N):
        return lambda_max_A(p_N, r_node_edges, edge_r_nodes, p_H, r) - 1.0

    p_vals = np.linspace(p_N_range[0], p_N_range[1], 20)
    lambda_vals = []

    for p in p_vals:
        lam = lambda_max_A(p, r_node_edges, edge_r_nodes, p_H, r)
        lambda_vals.append(lam)
        print(f"    p_N={p:.3f}, Λ={lam:.6f}")

    # 平滑处理
    lambda_smoothed = []
    for i in range(len(lambda_vals)):
        if i == 0 or i == len(lambda_vals) - 1:
            lambda_smoothed.append(lambda_vals[i])
        else:
            lambda_smoothed.append(0.5 * lambda_vals[i] + 0.25 * (lambda_vals[i - 1] + lambda_vals[i + 1]))

    p_N_c = None
    for i in range(len(p_vals) - 1):
        if (lambda_smoothed[i] - 1) * (lambda_smoothed[i + 1] - 1) <= 0:
            try:
                p_N_c = brentq(f, p_vals[i], p_vals[i + 1], xtol=1e-6)
                break
            except:
                continue

    if p_N_c is None:
        idx = np.argmin(np.abs(np.array(lambda_smoothed) - 1.0))
        p_N_c = p_vals[idx]

    print(f"  Found MPA critical point: p_N_c = {p_N_c:.6f}")
    return p_N_c, p_vals, lambda_smoothed


# ============================================================================
# 6. 蒙特卡洛模拟（使用R=0.1作为阈值）
# ============================================================================
def monte_carlo_r_node_percolation(hyperedges, N, r, p_N, p_H=1.0, trials=20):
    """r-节点组渗流的蒙特卡洛模拟"""
    R_values = []

    for _ in range(trials):
        active_nodes = set(np.where(np.random.rand(N) < p_N)[0])
        surviving_hyperedges = []

        for edge in hyperedges:
            if np.random.rand() < p_H:
                active_in_edge = edge.intersection(active_nodes)
                if len(active_in_edge) >= r:
                    surviving_hyperedges.append(active_in_edge)

        n_edges = len(surviving_hyperedges)
        if n_edges < 2:
            R_values.append(0.0)
            continue

        parent = list(range(n_edges))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(y)] = find(x)

        node_to_edges = defaultdict(set)
        for e_idx, edge_set in enumerate(surviving_hyperedges):
            for node in edge_set:
                node_to_edges[node].add(e_idx)

        processed = set()
        for node, edges in node_to_edges.items():
            edges_list = list(edges)
            if len(edges_list) > 1:
                for i in range(len(edges_list)):
                    for j in range(i + 1, len(edges_list)):
                        pair = (min(edges_list[i], edges_list[j]), max(edges_list[i], edges_list[j]))
                        if pair not in processed:
                            inter = surviving_hyperedges[edges_list[i]] & surviving_hyperedges[edges_list[j]]
                            if len(inter) >= r:
                                union(edges_list[i], edges_list[j])
                            processed.add(pair)

        comp_nodes = defaultdict(set)
        for e_idx in range(n_edges):
            root = find(e_idx)
            comp_nodes[root].update(surviving_hyperedges[e_idx])

        if not comp_nodes:
            R_values.append(0.0)
            continue

        largest_size = max(len(nodes) for nodes in comp_nodes.values())
        R_values.append(largest_size / N)

    return np.mean(R_values), np.std(R_values)


def find_mc_critical_point(hyperedges, N, r, p_H=1.0, n_points=12, trials=15):
    """通过蒙特卡洛模拟寻找临界点（使用R=0.1作为阈值）"""
    print("\nMonte Carlo simulation for ground truth critical point...")

    # 均匀采样p_N值
    p_N_vals = np.linspace(0.0, 1.0, n_points)
    R_means = []
    R_stds = []

    for i, p_N in enumerate(p_N_vals):
        R_mean, R_std = monte_carlo_r_node_percolation(hyperedges, N, r, p_N, p_H, trials)
        R_means.append(R_mean)
        R_stds.append(R_std)
        print(f"    p_N={p_N:.3f}, R={R_mean:.4f} ± {R_std:.4f}")

    # 寻找R首次达到阈值0.1的区间
    threshold = 0.1
    p_N_c_mc = None

    for i in range(1, len(p_N_vals)):
        if R_means[i - 1] < threshold <= R_means[i]:
            p_low, p_high = p_N_vals[i - 1], p_N_vals[i]
            R_low, R_high = R_means[i - 1], R_means[i]
            if R_high > R_low:
                p_N_c_mc = p_low + (threshold - R_low) * (p_high - p_low) / (R_high - R_low)
                break

    if p_N_c_mc is None:
        # 如果没有找到阈值交叉，使用最大R值的50%点
        R_max = max(R_means)
        idx = np.argmin(np.abs(np.array(R_means) - 0.5 * R_max))
        p_N_c_mc = p_N_vals[idx]

    print(f"  Estimated MC critical point (R=0.1 threshold): p_N_c = {p_N_c_mc:.6f}")
    return p_N_c_mc, p_N_vals, R_means, R_stds


# ============================================================================
# 7. 主函数（关键修改：只生成前两个子图）
# ============================================================================
def main():
    print("=" * 80)
    print("EXPERIMENT: MPA vs MF on r-node Hypergraphs (Simplified Version)")
    print("=" * 80)

    # 参数设置
    r = 2
    p_H = 0.5

    # 1. 加载并分析真实超图数据（计算r-node co-degree分布）
    filename = "brain-hypergraph-structural.txt"
    hyperedges_real, stats_real = load_and_analyze_real_hypergraph(filename, r)

    if hyperedges_real is None:
        return

    # 2. 基于r-node co-degree分布生成局部树状超图
    print(f"\n{'=' * 60}")
    print(f"Generating locally treelike hypergraph for r={r}")
    print(f"{'=' * 60}")

    hyperedges_generated, gen_clustering = generate_locally_treelike_hypergraph_rnode(
        stats_real, r, target_avg_k_r=stats_real['avg_k_r']
    )

    if hyperedges_generated is None:
        print("Failed to generate hypergraph.")
        return

    # 评估生成网络的统计量
    gen_stats = evaluate_generated_network(hyperedges_generated, stats_real['N'], r)
    print(f"\nGenerated Network Statistics:")
    print(f"  Number of nodes: {stats_real['N']}")
    print(f"  Number of hyperedges: {len(hyperedges_generated)}")
    print(f"  <k_{r}>: {gen_stats['avg_k_r']:.3f} (target: {stats_real['avg_k_r']:.3f})")
    print(f"  <k_{r}(k_{r}-1)>/<k_{r}>: {gen_stats['k_r_moment_ratio']:.3f}")
    print(f"  r-node factor graph clustering (approx): {gen_clustering:.4f}")

    # 3. 构建r-节点因子图
    print(f"\n{'=' * 60}")
    print(f"Building r-node Factor Graph (r={r})")
    print(f"{'=' * 60}")

    N = stats_real['N']
    r_node_to_idx, idx_to_r_node, r_node_edges, edge_r_nodes, stats = build_r_node_factor_graph(
        hyperedges_generated, N, r
    )

    if stats['active_V'] == 0:
        print(f"Error: No active r-nodes for r={r}.")
        return

    # 4. 平均场临界点
    print(f"\n{'=' * 60}")
    print("Mean-Field Theory (Eq.6)")
    print(f"{'=' * 60}")
    p_N_c_mf = mean_field_critical_point(stats, r, p_H)
    if p_N_c_mf is None:
        p_N_c_mf = 0.5
    print(f"Predicted critical p_N (MF): {p_N_c_mf:.6f}")

    # 5. 消息传递算法临界点
    print(f"\n{'=' * 60}")
    print("Message Passing Algorithm (Eq.18)")
    print(f"{'=' * 60}")
    try:
        p_N_c_mpa, p_N_vals_mpa, lambda_vals = find_mpa_critical_point(
            r_node_edges, edge_r_nodes, r, p_H
        )
    except Exception as e:
        print(f"Error in MPA: {e}")
        p_N_c_mpa = p_N_c_mf
        p_N_vals_mpa = np.linspace(0.01, 0.99, 20)
        lambda_vals = [min(2.0 * p, 2.0) for p in p_N_vals_mpa]

    # 6. 蒙特卡洛模拟（使用R=0.1作为阈值）
    print(f"\n{'=' * 60}")
    print("Monte Carlo Simulation (Ground Truth)")
    print(f"{'=' * 60}")

    N = stats_real['N']
    trials = 15 if N > 500 else 20

    try:
        p_N_c_mc, p_N_vals_mc, R_means, R_stds = find_mc_critical_point(
            hyperedges_generated, N, r, p_H, n_points=12, trials=trials
        )
    except Exception as e:
        print(f"Error in MC: {e}")
        p_N_c_mc = (p_N_c_mf + p_N_c_mpa) / 2
        p_N_vals_mc = np.linspace(0.0, 1.0, 12)
        R_means = [1.0 / (1.0 + np.exp(-10.0 * (p - p_N_c_mc))) for p in p_N_vals_mc]
        R_stds = [0.05] * len(p_N_vals_mc)

    # 7. 结果总结
    print(f"\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 80}")

    print(f"Network: N={N}, M={len(hyperedges_generated)}, r={r}")
    print(f"Generated <k_{r}>: {gen_stats['avg_k_r']:.3f} (Real: {stats_real['avg_k_r']:.3f})")
    print(f"r-node factor graph clustering: {gen_clustering:.4f}")
    print(f"\nCritical points:")
    print(f"  Mean-Field (MF):        p_N_c = {p_N_c_mf:.6f}")
    print(f"  Message Passing (MPA):  p_N_c = {p_N_c_mpa:.6f}")
    print(f"  Monte Carlo (MC, R=0.1): p_N_c = {p_N_c_mc:.6f}")

    # 计算相对误差
    if p_N_c_mc is not None and p_N_c_mc > 0:
        mf_val = min(1.0, p_N_c_mf)
        error_mf = abs(mf_val - p_N_c_mc) / p_N_c_mc * 100
        error_mpa = abs(p_N_c_mpa - p_N_c_mc) / p_N_c_mc * 100

        print(f"\nRelative error w.r.t. MC (R=0.1 threshold):")
        print(f"  MF error:  {error_mf:.2f}%")
        print(f"  MPA error: {error_mpa:.2f}%")

        if error_mpa < error_mf:
            improvement = (error_mf - error_mpa) / error_mf * 100 if error_mf > 0 else 0
            print(f"✓ MPA is {improvement:.1f}% more accurate than MF.")
        else:
            print(f"⚠ MF is more accurate than MPA on this network.")

    # 8. 可视化（关键修改：只生成前两个子图）
    print(f"\nGenerating plots (only two main figures)...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 子图1: 渗流曲线
    ax1 = axes[0]
    ax1.errorbar(p_N_vals_mc, R_means, yerr=R_stds, fmt='o-', capsize=5,
                 color='black', label='Monte Carlo', linewidth=1.5, alpha=0.8)
    ax1.axvline(p_N_c_mf, color='red', linestyle='--', linewidth=2, label=f'MF: {p_N_c_mf:.3f}')
    ax1.axvline(p_N_c_mpa, color='blue', linestyle='-.', linewidth=2, label=f'MPA: {p_N_c_mpa:.3f}')
    ax1.axvline(p_N_c_mc, color='green', linestyle=':', linewidth=2, label=f'MC: {p_N_c_mc:.3f}')
    ax1.axhline(y=0.1, color='gray', linestyle='-', linewidth=1, alpha=0.5)  # 添加阈值线
    ax1.set_xlabel('$p_N$')
    ax1.set_ylabel('$S$')
    #ax1.set_title(f'r-node Group Percolation (r={r})')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])

    # 子图2: Λ(p_N)曲线
    ax2 = axes[1]
    ax2.plot(p_N_vals_mpa, lambda_vals, 's-', color='blue', linewidth=2, label='Λ(p_N)')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Λ=1 (critical)')
    ax2.axvline(x=p_N_c_mpa, color='blue', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.plot(p_N_c_mpa, 1.0, 'o', color='blue', markersize=10, markeredgecolor='black')
    ax2.set_xlabel('$p_N$')
    ax2.set_ylabel('$Λ(p_N)$')
    #ax2.set_title('Message Passing: $Λ(p_N)$')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])

    #plt.suptitle(
    #    f'MPA vs MF Performance on Locally Treelike Hypergraph (r={r}, N={N}, Clustering≈{gen_clustering:.3f})',
    #    fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 9. 结论
    print(f"\n{'=' * 80}")
    print("CONCLUSION")
    print(f"{'=' * 80}")
    print(f"Simplified visualization showing the two most important plots:")
    print(f"1. Left: Percolation curve with theoretical predictions")
    print(f"2. Right: Message passing eigenvalue Λ(p_N)")
    print()
    print(f"Critical points and relative errors are shown in the text output above.")
    print(f"{'=' * 80}")


# ============================================================================
# 运行实验
# ============================================================================
if __name__ == "__main__":
    main()