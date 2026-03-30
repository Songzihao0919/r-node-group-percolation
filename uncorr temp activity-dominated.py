import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb, factorial
from collections import defaultdict
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class ActivityDominatedHOAVD:
    """
    活动主导机制HOAVD渗流模拟器
    仅包含活动主导机制，用于验证非相关超图的r点时间渗流阈值T_unc
    """

    def __init__(self, N=5000, m=4, r=2, dist_type='uniform', gamma=2.5):
        self.N = N
        self.m = m
        self.r = r
        self.threshold_R = 0.10  # R的临界值为0.1

        print(f"Initializing network: N={N}, m={m}, r={r}")

        # 节点属性生成
        if dist_type == 'uniform':
            self.a = np.random.uniform(0.01, 1.0, N)
            self.b = np.random.uniform(0.01, 1.0, N)
        elif dist_type == 'powerlaw':
            self.a = 0.6 + 0.4 * np.random.power(gamma, N)
            self.b = np.random.exponential(0.005, N)

        print(f"  Activity: mean={self.a.mean():.4f}, std={self.a.std():.4f}")
        print(f"  Vulnerability: mean={self.b.mean():.4f}, std={self.b.std():.4f}")

        # 预计算理论值
        self._precompute_theoretical_values()

    def _precompute_theoretical_values(self, num_samples=500000):
        """预计算理论临界值"""
        print("Computing theoretical values...")

        # 计算理论T_unc (公式47)
        mean_a = np.mean(self.a)
        mean_a2 = np.mean(self.a ** 2)

        numerator_T = (factorial(self.r) * factorial(self.m - self.r) ** 2 *
                       self.N ** (self.r - 1) * mean_a)
        denominator_T = (factorial(self.m - 1) *
                         (factorial(self.m) - factorial(self.r) * factorial(self.m - self.r)) *
                         (mean_a2 + (self.m - 1) * mean_a ** 2))
        self.T_unc_theory = numerator_T / denominator_T if denominator_T > 0 else float('inf')
        print(f"  Theoretical T_unc = {self.T_unc_theory:.4f}")

    # ========== 活动主导机制 ==========
    def simulate_activity_dominated(self, T_max=None, dt=0.2):
        """
        活动主导机制模拟
        返回R = S^r
        """
        print(f"\nSimulating activity-dominated regime...")

        if T_max is None:
            T_max = max(1.5 * self.T_unc_theory, 20.0)

        print(f"  T_max = {T_max:.1f}, dt = {dt:.3f}, threshold_R = {self.threshold_R}")

        # 初始化
        t = 0
        times = []
        R_values = []

        # 数据结构
        hyperedge_connections = defaultdict(set)
        hyperedges_list = []
        node_to_hyperedges = defaultdict(set)

        # 记录间隔
        record_interval = max(1, int(0.5 / dt))

        pbar = tqdm(total=int(T_max / dt), desc="Time evolution")

        while t < T_max:
            # 超边形成
            new_hyperedges = []
            for i in range(self.N):
                if np.random.random() < self.a[i] * dt:
                    other_nodes = np.arange(self.N)
                    other_nodes = np.delete(other_nodes, i)
                    selected = np.random.choice(other_nodes, size=self.m - 1, replace=False)
                    hyperedge = tuple(sorted([i] + list(selected)))
                    new_hyperedges.append(hyperedge)

            # 添加新超边并更新连接
            for hyperedge in new_hyperedges:
                edge_idx = len(hyperedges_list)
                hyperedges_list.append(hyperedge)

                # 更新节点到超边的映射
                for node in hyperedge:
                    node_to_hyperedges[node].add(edge_idx)

                # 查找连接
                candidate_edges = set()
                for node in hyperedge:
                    candidate_edges.update(node_to_hyperedges[node])

                candidate_edges.discard(edge_idx)

                # 边界检查，防止访问已删除的超边
                for other_idx in candidate_edges:
                    # 检查索引是否有效
                    if other_idx < len(hyperedges_list):
                        common_nodes = len(set(hyperedge) & set(hyperedges_list[other_idx]))
                        if common_nodes >= self.r:
                            hyperedge_connections[edge_idx].add(other_idx)
                            hyperedge_connections[other_idx].add(edge_idx)

            # 记录
            if int(t / dt) % record_interval == 0:
                times.append(t)
                R = self._calculate_R_from_hyperedges(hyperedges_list, hyperedge_connections)
                R_values.append(R)

            t += dt
            pbar.update(1)
            if len(R_values) > 0:
                pbar.set_postfix({'R': f'{R_values[-1]:.4f}'})

        pbar.close()

        # 最终记录
        if not times or abs(times[-1] - T_max) > 1e-9:
            times.append(T_max)
            R = self._calculate_R_from_hyperedges(hyperedges_list, hyperedge_connections)
            R_values.append(R)

        # 寻找临界时间
        T_c_measured = None
        for i in range(1, len(times)):
            if R_values[i] > self.threshold_R:
                T_c_measured = times[i - 1] + (times[i] - times[i - 1]) * (self.threshold_R - R_values[i - 1]) / (
                        R_values[i] - R_values[i - 1])
                break

        return times, R_values, T_c_measured

    def _calculate_R_from_hyperedges(self, hyperedges, connections):
        """计算R = (节点数/N)^r"""
        if len(hyperedges) < 2:
            return 0.0

        # 找到最大连通分量
        visited = [False] * len(hyperedges)
        max_component_nodes = set()

        for i in range(len(hyperedges)):
            if not visited[i] and i in connections:
                # BFS遍历
                component_edges = []
                queue = [i]
                visited[i] = True

                while queue:
                    current = queue.pop()
                    component_edges.append(current)

                    for neighbor in connections[current]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)

                # 计算节点并集
                component_nodes = set()
                for edge_idx in component_edges:
                    component_nodes.update(hyperedges[edge_idx])

                if len(component_nodes) > len(max_component_nodes):
                    max_component_nodes = component_nodes

        if len(max_component_nodes) == 0:
            return 0.0

        # 计算S和R
        S = len(max_component_nodes) / self.N
        R = S ** self.r

        return R


# ========== 可视化函数 ==========
def plot_activity_dominated_results(times, R_values, T_unc_theory, T_c_measured, N, m, r):
    """绘制活动主导机制的结果 - 单张线性坐标图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(times, R_values, 'b-', linewidth=2, label='$S(T)$')

    if T_c_measured:
        ax.axvline(x=T_c_measured, color='green', linestyle=':',
                   linewidth=2, label=f'Simulated: $T_{{unc}}$ = {T_c_measured:.2f}')

    ax.axvline(x=T_unc_theory, color='red', linestyle='--',
               linewidth=2, label=f'Theoretical: $T_{{unc}}$ = {T_unc_theory:.2f}')
    ax.axhline(y=0.10, color='black', linestyle=':', alpha=0.7, label='$S$ = 0.1')

    ax.set_xlabel('Time $T$', fontsize=12)
    ax.set_ylabel('S', fontsize=12)#$R = S^{r}$ (2-node giant component size)
    #ax.set_title(f'Activity-Dominated Regime\n$N$={N}, $m$={m}, $r$={r}', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.01, max(R_values) * 1.1 if len(R_values) > 0 else 0.2)

    plt.tight_layout()
    return fig


# ========== 主实验函数 ==========
def run_activity_dominated_experiment(N=2000, m=4, r=2, dist_type='uniform'):
    """运行实验：非相关超图的活动主导机制验证"""
    print("=" * 80)
    print(f"EXPERIMENT: ACTIVITY-DOMINATED REGIME")
    print(f"N={N}, m={m}, r={r}, distribution={dist_type}")
    print("=" * 80)

    # 初始化模拟器
    simulator = ActivityDominatedHOAVD(N=N, m=m, r=r, dist_type=dist_type)

    # 活动主导机制模拟
    print("\n" + "=" * 60)
    print("SIMULATING ACTIVITY-DOMINATED REGIME")
    print("=" * 60)

    start_time = time.time()
    times, R_values, T_c_measured = simulator.simulate_activity_dominated(
        T_max=None,
        dt=0.2
    )
    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.1f} seconds")

    if T_c_measured:
        error_T = abs(simulator.T_unc_theory - T_c_measured) / simulator.T_unc_theory * 100
        print(f"Theoretical T_unc: {simulator.T_unc_theory:.4f}")
        print(f"Measured T_c: {T_c_measured:.4f}")
        print(f"Relative error: {error_T:.1f}%")
    else:
        print("Warning: No percolation transition detected")

    # 绘制结果
    fig = plot_activity_dominated_results(times, R_values, simulator.T_unc_theory,
                                          T_c_measured, N, m, r)
    fig.savefig(f'activity_dominated_N{N}_m{m}_r{r}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 总结
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    print(f"\nNetwork parameters:")
    print(f"  N={N}, m={m}, r={r}")
    print(f"  Activity distribution: mean={simulator.a.mean():.4f}, std={simulator.a.std():.4f}")
    print(f"  Vulnerability distribution: mean={simulator.b.mean():.4f}, std={simulator.b.std():.4f}")

    print(f"\nTheoretical prediction:")
    print(f"  T_unc (activity-dominated) = {simulator.T_unc_theory:.4f}")

    print(f"\nExperimental result:")
    if T_c_measured:
        print(f"  T_c (activity-dominated) = {T_c_measured:.4f}, error = {error_T:.1f}%")
    else:
        print(f"  T_c (activity-dominated): Not detected")

    return {
        'times': times,
        'R_values': R_values,
        'T_c_measured': T_c_measured,
        'T_unc_theory': simulator.T_unc_theory
    }


# ========== 主函数 ==========
def main():
    """主函数"""
    import sys

    np.random.seed(42)

    N = 10000
    m = 4
    r = 2
    dist_type = 'powerlaw'

    print("HOAVD Model: r-node Percolation on Temporal Hypergraphs")
    print(f"Using R = S^{r} as percolation measure, threshold R = 0.10")
    print(f"Network size: N={N}")
    print(f"Testing only the activity-dominated regime")

    results = run_activity_dominated_experiment(
        N=N,
        m=m,
        r=r,
        dist_type=dist_type
    )


if __name__ == "__main__":
    main()