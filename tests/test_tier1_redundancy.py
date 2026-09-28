"""Synthetic unit test for tier1_redundancy.py's greedy pruning logic.

No real UniProt group in the Tier 1 dataset requires more than one pass
of the greedy pick-remove loop (every real collapse is a trivial single
pair -- see tier1_redundancy.py's own __main__ output), so this
constructs a hand-specified 5-node RMSD sub-network that forces exactly
two passes, with zero ties and zero leftover isolated nodes, so the
result is fully predictable by hand.

Structure: a tight pendant pair {1, 2}, bridged by a comparatively weak
edge to a tight 3-node clique {3, 4, 5}:

    1 --0.05-- 2 --0.45-- 3
                          |  \\
                        0.10  0.12
                          |     \\
                          4--0.11--5

Pass 1: node 1's mean (0.05, its only neighbor) is the smallest of all
five nodes' means (2: 0.25, 3: 0.223, 4: 0.105, 5: 0.115), so it's
picked; its only neighbor, 2, is removed with it. The bridge edge
(2-3) disappears with node 2, leaving {3, 4, 5} as an isolated clique.

Pass 2: among {3, 4, 5}, node 4's mean (mean(0.10, 0.11) = 0.105) is
the smallest (3: 0.11, 5: 0.115), so it's picked; both its neighbors --
3 and 5, the rest of the clique -- are removed with it.

Expected result: exactly 2 representatives (1 and 4) covering all 5
nodes, with {2: 1, 3: 4, 5: 4} as constituents -- nothing left over to
fall through to the "isolated node keeps itself" branch, so this
exercises the repeated while-loop pass that no real UniProt group
currently does.
"""

import networkx as nx

from preprocessing.tier1.tier1_redundancy import reduce_component


def test_multi_step_greedy_pruning():
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            (1, 2, 0.05),
            (2, 3, 0.45),
            (3, 4, 0.10),
            (3, 5, 0.12),
            (4, 5, 0.11),
        ],
        weight="rmsd",
    )

    assignment = reduce_component(graph)

    expected = {1: 1, 2: 1, 3: 4, 4: 4, 5: 4}
    assert assignment == expected

    representatives = set(assignment.values())
    assert representatives == {1, 4}
