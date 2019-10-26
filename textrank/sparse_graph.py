#! /usr/bin/env python3
from collections import defaultdict
from typing import List
import operator

VERBOSE = False


def verbose(*msg):
    if VERBOSE:
        print(*msg)


class SparseGraph:
    def __init__(self):
        self.nodes = set()
        self.link_out = defaultdict(set)
        self.link_in = defaultdict(set)

    def add_node(self, n):
        self.nodes.add(n)

    def link(self, from_node, to_node):
        self.add_node(from_node)
        self.add_node(to_node)
        self.link_out[from_node].add(to_node)
        self.link_in[to_node].add(from_node)

    def __repr__(self):
        return f"SparseGraph(nodes={self.nodes}, link_out={self.link_out}, link_in={self.link_in})"

    def page_rank(
        self,
        max_iteration: int = 10000,
        convergence: float = 0.0001,
        dampling_factor: float = 0.2,
    ):
        """ 使用PageRank算法计算节点在网络中的重要程度
        
        Args:
            max_iteratioin: int 最大迭代次数
            convergence: float (0 - 1) 收敛判据
            dampling_factor: float (0 - 1) 阻尼系数，（随机迁移概率）
        """
        nodes = self.nodes
        N = len(nodes)
        init_val = 1 / N
        pr = {n: init_val for n in nodes}

        verbose(f"initial pr: {pr}")

        tax = dampling_factor * init_val

        def iter_compute_new_pr(pr):
            new_pr = {}
            for v in pr.keys():
                t = 0.0
                for node in self.link_in.get(v, []):
                    t += 1.0 / len(self.link_out[node]) + pr[node]
                new_pr[v] = t * (1 - dampling_factor)
            for v in new_pr.keys():
                new_pr[v] += tax

            return new_pr

        def compute_diff(old_pr, new_pr):
            return max(abs(old_pr[node] - new_pr[node]) for node in nodes)

        # 收敛判据
        convergence: float = init_val / 10000.0
        # 两次迭代间的diff
        diff: float = 1.0
        # 循环次数
        i: int = 0
        while i < max_iteration and diff > convergence:
            _pr = iter_compute_new_pr(pr)
            diff = compute_diff(_pr, pr)
            verbose(f"{i}. {_pr} {diff * 100:.2f}%")
            pr = _pr
            i += 1

        return sorted(pr.items(), key=operator.itemgetter(1), reverse=True)


if __name__ == "__main__":
    g = SparseGraph()
    g.link(1, 3)
    g.link(1, 2)
    g.link(2, 4)
    g.link(2, 3)

    VERBOSE = True

    print(g)

    print(g.page_rank())

