import itertools
from typing import Union, Optional
import numpy as np


class Graph:

    """
    Takes Adjacency matrix and a dictionary of node names in a form of
        {0: 'name_of_node_0', 1: 'name_of_node_1', etc.}
    Dictionary of names is optional. If not provided, names of nodes will be their indices in the adjacency matrix.

    "Geodesic" keyword in many methods allows to treat the graph as unweighted one.

    Implemented methods:

    1. shortest_path:
        Searching for paths from node "_from" to the node "_to".
        If there is no path: False is returned.

    2. distance:
        Returns the cost of the shortest path between 2 nodes.

    3. is_complete:
        Checks if every node is connected to every other node directly.

    4. is_connected
        Checks if every node is reachable from any other node.

    5. is_regular:
        Check if every node has the same number of adjacent edges.

    6. is_weighted:
        Checks if edges have different weights.

    7. is_directed:
        Checks if the adjacent matrix is symmetric.

    8. is_moore:
        Checks if the graph is Moore Bound.

    9. is_acyclic:
        Check if there are no cycles in the graph.

    10. is_tree:
        Checks if the graph can be called a tree (connected and acyclic).

    11. diameter:
        Returns the maximum shortest path.

    12. eccentricity:
        Returns the maximum distance from the given vertex to other vertices.

    13. radius:
        Returns the minimum eccentricity of all vertices.

    14. average_distance_of_vertex:
        Returns the average distance from the given vertex to every other vertex.

    15. average_distance_of_graph:
        Returns the mean of average distances of all vertices.

    16. stress_centrality:
        Returns the total number of shortest paths going through the given node.

    17. betweenness_centrality:
        Returns the sum of fractions of shortest paths between each pair of nodes (not containing the given one), which
        go through the given node.

    18. degree_centrality:
        Returns the degree of a node.

    19. k_hop_centrality:
        Returns the number of nodes, which are reachable through k edges from the given one. (can be normalized)

    20. k_rank_centrality:
        Returns the rank of the given node among its own k-hop neighbours sorted according to their degrees.

    21. closeness_centrality:
        Returns 1 / sum of distances from the given node to every other node.

    22. clustering_coefficient_of_vertex:
        Returns the fraction of existing edges between the neighbours of the given node to the theoretically
        maximum number of edges.

    23. clustering_coefficient_of_graph:
        Returns the mean of individual clustering coefficients of all vertices.

    24. neighbourhood_overlap:
        Returns the fraction of common neighbours to all the neighbours of the 2 given nodes.

    25. minimum_spanning_tree:
        Return the adjacency matrix of a minimum spanning tree constructed according to Prim-Jarnik or Kruskal's
        algorithm.

    26. summary:
        Returns the report with main measures of the graph (not parametrizable).
    """

    def __init__(self, matrix: np.array, node_names: dict = None):

        assert isinstance(matrix, np.ndarray), 'matrix must be a numpy array'
        assert len(matrix.shape) == 2, 'matrix must be a 2D array'
        assert matrix.shape[0] == matrix.shape[1], 'matrix must be a square'
        assert np.all(matrix >= 0), 'edges must have non-negative weights'

        self.matrix = matrix
        if not node_names:
            self.node_names = {k: str(k) for k in range(len(matrix))}
            self.node_indices = {str(k): int(k) for k in range(len(matrix))}
        else:
            assert isinstance(node_names, dict), 'node_names must be a dictionary, if provided'
            for k, v in node_names.items():
                assert 0 <= k < len(matrix), 'keys of node_names must be integers from 0 upto len(matrix) - 1'
                assert isinstance(v, str), 'node_names values must be of type str'
            assert len(node_names) == len(matrix), 'node_names are less than matrix rows'
            assert len(set(node_names.values())) == len(matrix), 'node_names values must be unique'
            self.node_names = node_names
            self.node_indices = {v: k for k, v in self.node_names.items()}

    def process_node(self, node: Union[int, str]):
        """
        Checking and unification of node input
        """

        if isinstance(node, int):
            assert 0 <= node < len(self.matrix), f'node with index {node} is not in the matrix'
            return node
        elif isinstance(node, str):
            assert node in self.node_names.values(), f'no such name {node} in node_names'
            return self.node_indices[node]
        else:
            raise ValueError(f'node must be an integer or a string, not {type(node)}')

    def shortest_paths(self, _from: Union[int, str], _to: Union[int, str],
                       provide_all: bool = False, geodesic: bool = False):
        """
        Searching for paths from node "_from" to the node "_to".
        If there is no path: False is returned.

        Params:
            _from: node to start the path from
            _to: node to end the path at
            provide_all: return all possible unique paths (not only the shortest)
            geodesic: set all weights to 1

        Returns:
            instance of class Path, which is a list of nodes with .cost attribute
        """

        _from, _to = self.process_node(_from), self.process_node(_to)

        class Path(list):

            def __init__(self, _from, _to, cost: int = 0, node_names: dict = None):
                super().__init__()
                self.append(_from)
                self.cost = cost
                self.goal = _to
                self.node_names = node_names

            def copy(self):
                path_copy = Path(self[0], self.goal, self.cost, node_names=self.node_names)
                for node in self[1:]:
                    path_copy.append(node)
                return path_copy

            def __hash__(self):
                return sum(hash(node) for node in self)

            def is_terminated(self):
                return self.goal == self[-1]

            def __str__(self):
                return ' -> '.join(self.node_names[x] for x in self) + f' (cost: {self.cost})'

            def __repr__(self):
                return str(self)

        paths = {Path(_from, _to, node_names=self.node_names)}
        while True:
            stop = True
            new_step_paths = set()
            for path in paths:
                if path.is_terminated():
                    new_step_paths.add(path)
                    continue
                for node in range(len(self.matrix)):
                    if node not in path and self.matrix[path[-1], node] > 0:
                        stop = False
                        new_path = path.copy()
                        new_path.append(node)
                        if geodesic:
                            new_path.cost += 1
                        else:
                            new_path.cost += self.matrix[path[-1], node]
                        new_step_paths.add(new_path)
            paths = new_step_paths
            if stop:
                break

        if not paths:
            return False

        if provide_all:
            return list(paths)

        min_cost = min(p.cost for p in paths)
        paths = [p for p in paths if p.cost == min_cost]
        return paths

    def distance(self, _from: Union[int, str], _to: Union[int, str], geodesic: bool = False):
        """
        Returns the cost of the shortest path.
        """

        _from, _to = self.process_node(_from), self.process_node(_to)

        shortest = self.shortest_paths(_from, _to, geodesic=geodesic)
        if not shortest:
            return False
        return shortest[0].cost

    def is_complete(self):
        """
        Checks if every node is connected to every other node directly
        """
        m = np.sum(self.matrix > 0)
        n = len(self.matrix)
        return m == n * (n - 1) / 2

    def is_connected(self, strongly: bool = True):
        """
        Checks if every node is reachable from any other node
        """

        if not strongly:
            pseudo_matrix = self.matrix + self.matrix.T
            return Graph(pseudo_matrix).is_connected()

        for start_node in range(len(self.matrix)):
            for end_node in range(len(self.matrix)):
                if start_node != end_node:
                    shortest = self.shortest_paths(_from=start_node, _to=end_node)
                    if not shortest:
                        return False
        return True

    def is_directed(self):
        """
        Checks if the adjacent matrix is symmetric.
        """
        return not np.all(self.matrix == self.matrix.T)

    def is_regular(self):
        """
        Check if every node has the same number of adjacent edges.
        """
        degree = sum(self.matrix[0] > 0)
        for row in self.matrix[1:]:
            if sum(row > 0) != degree:
                return False
        return True

    def is_weighted(self):
        """
        Checks if edges have different weights.
        """
        if len(set(self.matrix.flatten()) - {0}) == 1:
            return False
        return True

    def is_moore(self):
        """
        Checks if the graph is Moore Bound.
        """
        if not self.is_regular():
            return False
        degree = sum(self.matrix[0] > 0)
        diam = self.diameter()
        upper_bound = 1 + sum(degree * (degree - 1) ** i for i in range(diam))
        if len(self.matrix) == upper_bound:
            return True
        else:
            return False

    def is_acyclic(self):
        """
        Check if there are no cycles in the graph.
        """
        for start_node in range(len(self.matrix)):
            for end_node in range(start_node, len(self.matrix)):
                if start_node != end_node:
                    paths = self.shortest_paths(_from=start_node, _to=end_node, provide_all=True)
                    back_paths = self.shortest_paths(_from=end_node, _to=start_node, provide_all=True)
                    if paths is False or back_paths is False:
                        continue
                    for path in paths:
                        p = ''.join(str(x) for x in path)
                        for back_path in back_paths:
                            bp = ''.join(str(x) for x in back_path[::-1])
                            if p != bp:
                                return False

        return True

    def is_tree(self):
        """
        Checks if the graph can be called a tree (connected and acyclic).
        """
        return self.is_connected() & self.is_acyclic()

    def diameter(self, geodesic: bool = False):
        """
        Returns the maximum shortest path.
        """
        score = 0
        for start_node in range(len(self.matrix)):
            for end_node in range(len(self.matrix)):
                if start_node != end_node:
                    shortest_cost = self.distance(_from=start_node, _to=end_node, geodesic=geodesic)
                    score = max(shortest_cost, score)
        return score

    def eccentricity(self, node: Union[int, str] = None, geodesic: bool = False):
        """
        Returns the maximum distance from the given vertex to other vertices.
        """
        if node is None:
            result = {}
            for n in self.node_names.values():
                result.update({n: self.eccentricity(n, geodesic)})
            return result

        node = self.process_node(node)

        score = 0
        for end_node in range(len(self.matrix)):
            if node != end_node:
                shortest_cost = self.distance(_from=node, _to=end_node, geodesic=geodesic)
                score = max(shortest_cost, score)
        return score

    def radius(self, geodesic: bool = False):
        """
        Returns the minimum eccentricity of all vertices.
        """
        score = self.diameter(geodesic=geodesic)
        for node in range(len(self.matrix)):
            ecc = self.eccentricity(node, geodesic=geodesic)
            score = min(ecc, score)
        return score

    def average_distance_of_vertex(self, node: Union[int, str] = None, geodesic: bool = False):
        """
        Returns the average distance from the given vertex to every other vertex.
        """
        if node is None:
            result = {}
            for n in self.node_names.values():
                result.update({n: self.average_distance_of_vertex(n, geodesic)})
            return result

        node = self.process_node(node)
        node = self.process_node(node)
        score = 0
        for end_node in range(len(self.matrix)):
            if node != end_node:
                d = self.distance(_from=node, _to=end_node, geodesic=geodesic)
                score += d
        return score / (len(self.matrix) - 1)

    def average_distance_of_graph(self, geodesic: bool = False):
        """
        Returns the mean of average distances of all vertices.
        """
        score = 0
        for node in range(len(self.matrix)):
            d = self.average_distance_of_vertex(node, geodesic=geodesic)
            score += d
        return score / len(self.matrix)

    def stress_centrality(self, node: Union[int, str] = None, geodesic: bool = False):
        """
        Returns the total number of shortest paths going through the given node.
        """

        if node is None:
            result = {}
            for n in self.node_names.values():
                result.update({n: self.stress_centrality(n, geodesic)})
            return result

        directed = self.is_directed()
        node = self.process_node(node)
        score = 0

        for start_node in range(len(self.matrix)):
            for end_node in range(len(self.matrix)):
                if start_node != node and end_node != node and start_node != end_node:
                    shortest = self.shortest_paths(_from=start_node, _to=end_node, geodesic=geodesic)
                    containing_node = 0
                    for path in shortest:
                        if node in path:
                            containing_node += 1
                    score += containing_node

        if not directed:
            score /= 2
        return score

    def betweenness_centrality(self, node: Union[int, str] = None, geodesic: bool = False):
        """
        Returns the sum of fractions of shortest paths between each pair of nodes (not containing the given one), which
        go through the given node.
        """

        if node is None:
            result = {}
            for n in self.node_names.values():
                result.update({n: self.betweenness_centrality(n, geodesic)})
            return result

        directed = self.is_directed()
        node = self.process_node(node)
        score = 0

        for start_node in range(len(self.matrix)):
            for end_node in range(len(self.matrix)):
                if start_node != node and end_node != node and start_node != end_node:
                    shortest = self.shortest_paths(_from=start_node, _to=end_node, geodesic=geodesic)
                    total_paths = len(shortest)
                    containing_node = 0
                    for path in shortest:
                        if node in path:
                            containing_node += 1
                    score += containing_node / total_paths

        if not directed:
            score /= 2
        return score

    def degree_centrality(self, node: Union[int, str] = None, normalized: bool = False):
        """
        Returns the degree of a node.
        """
        if node is None:
            result = {}
            for n in self.node_names.values():
                result.update({n: self.degree_centrality(n, normalized)})
            return result

        directed = self.is_directed()
        node = self.process_node(node)

        if not directed:
            score = sum(self.matrix[node] > 0)
            if normalized:
                score /= (len(self.matrix) - 1)
            return score

        if directed:
            score_out = sum(self.matrix[node] > 0)
            score_in = sum(self.matrix[:, node] > 0)
            if normalized:
                score_out /= (len(self.matrix) - 1)
                score_in /= (len(self.matrix) - 1)
            return score_out + score_in

    def k_hop_centrality(self, node: Union[int, str] = None, k: int = 2, normalized: bool = False):
        """
        Returns the number of nodes, which are reachable through k edges from the given one. (can be normalized)
        """
        if node is None:
            result = {}
            for n in self.node_names.values():
                result.update({n: self.k_hop_centrality(n, k, normalized)})
            return result

        node = self.process_node(node)
        assert 1 <= k < len(self.matrix), 'k must be between 1 and len(matrix)'

        reachable_nodes = {node}
        for h in range(k):
            new_reachable_nodes = set()
            for node in reachable_nodes:
                for i, edge in enumerate(self.matrix[node]):
                    if edge > 0:
                        new_reachable_nodes.add(i)
            reachable_nodes |= new_reachable_nodes

        score = len(reachable_nodes) - 1
        if normalized:
            score /= (len(self.matrix) - 1)
        return score

    def k_rank_centrality(self, node: Union[int, str] = None, k: int = 2, resolves_ties: bool = False):
        """
        Returns the rank of the given node among its own k-hop neighbours sorted according to their degrees.
        Setting resolve_tiers to true allows to rank consequently the vertices with the same degree.
        """
        if node is None:
            result = {}
            for n in self.node_names.values():
                result.update({n: self.k_rank_centrality(n, k, resolves_ties)})
            return result

        node = self.process_node(node)
        assert 1 <= k < len(self.matrix), 'k must be between 1 and len(matrix)'

        reachable_nodes = {node}
        for h in range(k):
            new_reachable_nodes = set()
            for n in reachable_nodes:
                for i, edge in enumerate(self.matrix[n]):
                    if edge > 0:
                        new_reachable_nodes.add(i)
            reachable_nodes |= new_reachable_nodes
        reachable_nodes |= {node}

        scores = [(n, self.degree_centrality(n)) for n in reachable_nodes]
        nodes = [n[0] for n in sorted(scores, key=lambda x: x[-1], reverse=True)]
        scores = [n[1] for n in sorted(scores, key=lambda x: x[-1], reverse=True)]

        if resolves_ties:
            return nodes.index(node) + 1

        node_score = self.degree_centrality(node)
        counter = 1
        for i, score in enumerate(scores):
            if i > 0 and score == scores[i-1]:
                continue
            if node_score != score:
                counter += 1
            else:
                return counter

    def closeness_centrality(self, node: Union[int, str] = None):
        """
        Returns 1 / sum of distance from the given node to every other node.
        """

        if node is None:
            result = {}
            for n in self.node_names.values():
                result.update({n: self.closeness_centrality(n)})
            return result

        node = self.process_node(node)
        score = 0

        for end_node in range(len(self.matrix)):
            if end_node != node:
                score += float(self.distance(_from=node, _to=end_node))

        if score == 0:
            return 0

        return 1 / score

    def clustering_coefficient_of_vertex(self, node: Union[int, str] = None):
        """
        Returns the fraction of existing edges between the neighbours of the given node to the theoretically
        maximum number of edges.
        """

        if node is None:
            result = {}
            for n in self.node_names.values():
                result.update({n: self.clustering_coefficient_of_vertex(n)})
            return result

        node = self.process_node(node)
        directed = self.is_directed()
        d = self.degree_centrality(node)

        if d == 0 or d == 1:
            return 0

        neighbours = np.argwhere(self.matrix[node] > 0).flatten()
        sub_matrix = self.matrix[neighbours]
        sub_matrix = sub_matrix[:, neighbours]

        k = np.sum(sub_matrix > 0)

        if not directed:
            k /= 2
        return 2 * k / (d * (d - 1))

    def clustering_coefficient_of_graph(self):
        """
        Returns the mean of individual clustering coefficients of all vertices.
        """
        score = 0
        for node in range(len(self.matrix)):
            score += self.clustering_coefficient_of_vertex(node)
        return score / len(self.matrix)

    def neighbourhood_overlap(self, node_1: Union[int, str] = None, node_2: Union[int, str] = None):
        """
        Returns the fraction of common neighbours among all the neighbours of the 2 given nodes.
        """
        if node_1 is None or node_2 is None:
            if node_1 is None:
                node_1 = list(self.node_names.values())
            else:
                node_1 = [node_1]
            if node_2 is None:
                node_2 = list(self.node_names.values())
            else:
                node_2 = [node_2]
            result = {}
            for n1 in node_1:
                for n2 in node_2:
                    if f'{n1} - {n2}' in result.keys() or f'{n2} - {n1}' in result.keys():
                        continue
                    result.update({f'{n1} - {n2}': self.neighbourhood_overlap(n1, n2)})
            return result

        node_1, node_2 = self.process_node(node_1), self.process_node(node_2)
        neighbours_1 = set(np.argwhere(self.matrix[node_1] > 0).flatten())
        neighbours_2 = set(np.argwhere(self.matrix[node_2] > 0).flatten())
        return len(neighbours_1 & neighbours_2) / len((neighbours_1 | neighbours_2) - {node_1, node_2})

    def minimum_spanning_tree(self, algorithm: str = 'prim-jarnik', start_node: Optional[Union[int, str]] = None):
        """
        Return the adjacency matrix of a minimum spanning tree constructed according to Prim-Jarnik or Kruskal's
        algorithm.
        """
        if not start_node:
            start_node = 0
        else:
            start_node = self.process_node(start_node)

        directed = self.is_directed()

        if algorithm == 'prim-jarnik':
            nodes_to_add = [x for x in list(range(len(self.matrix))) if x != start_node]
            nodes_added = [start_node]
            mst = np.zeros_like(self.matrix)
            while nodes_to_add:
                best_edge, best_score = None, float('Inf')
                for node in nodes_added:
                    for edge in np.argwhere(self.matrix[node] > 0):
                        edge = edge[0]
                        if edge not in nodes_added and self.matrix[node, edge] <= best_score:
                            best_score = self.matrix[node, edge]
                            best_edge = (node, edge)
                mst[best_edge[0], best_edge[1]] = best_score
                if not directed:
                    mst[best_edge[1], best_edge[0]] = best_score
                nodes_added.append(best_edge[1])
                nodes_to_add = [n for n in nodes_to_add if n != best_edge[1]]

            return mst

        elif algorithm == 'kruskal':
            nodes_to_add = [x for x in list(range(len(self.matrix)))]
            nodes_added = []
            mst = np.zeros_like(self.matrix)
            while nodes_to_add:
                best_edge, best_score = None, float('Inf')
                for node_1 in range(len(self.matrix)):
                    for node_2 in range(len(self.matrix)):
                        if mst[node_1, node_2] == 0 and self.matrix[node_1, node_2] > 0:
                            if self.matrix[node_1, node_2] <= best_score:
                                trial_matrix = mst.copy()
                                trial_matrix[node_1, node_2] = self.matrix[node_1, node_2]
                                if not directed:
                                    trial_matrix[node_2, node_1] = self.matrix[node_1, node_2]
                                if Graph(trial_matrix).is_acyclic():
                                    best_score = self.matrix[node_1, node_2]
                                    best_edge = (node_1, node_2)
                mst[best_edge[0], best_edge[1]] = best_score
                if not directed:
                    mst[best_edge[1], best_edge[0]] = best_score
                nodes_added.append(best_edge[1])
                nodes_to_add = [n for n in nodes_to_add if n != best_edge[1] and n != best_edge[0]]

            return mst

        else:
            raise ValueError(f'unknown algorithm: {algorithm}. Possible: "prim-jarnik", "kruskal"')

    def summary(self):
        """
        Returns the report with main measures of the graph (not parametrizable).
        """
        report = f'Adjacency matrix:\n{self.matrix}\n\nThis graph:\n'
        report += f'\t - is {"NOT " if not self.is_connected() else ""}connected\n'
        report += f'\t - is {"NOT " if not self.is_regular() else ""}regular\n'
        report += f'\t - is {"NOT " if not self.is_directed() else ""}directed\n'
        report += f'\t - is {"NOT " if not self.is_weighted() else ""}weighted\n'
        report += f'\t - is {"NOT " if not self.is_complete() else ""}complete\n'
        report += f'\t - is {"NOT " if not self.is_acyclic() else ""}acyclic\n'
        report += f'\t - is {"NOT " if not self.is_moore() else ""}a Moore graph\n'
        report += f'\t - is {"NOT " if not self.is_tree() else ""}a Tree\n\n'
        report += 'Main Measures:\n'
        report += f'\t Diameter (geodesic) = {self.diameter(geodesic=True):3g}\n'
        report += f'\t Diameter (not geodesic) = {self.diameter(geodesic=False):3g}\n'
        report += f'\t Radius (geodesic) = {self.radius(geodesic=True):3g}\n'
        report += f'\t Radius (not geodesic) = {self.radius(geodesic=False):3g}\n'
        report += f'\t Average distance (geodesic) = {self.average_distance_of_graph(geodesic=True):3g}\n'
        report += f'\t Average distance (not geodesic) = {self.average_distance_of_graph(geodesic=False):3g}\n'
        report += f'\t Clustering coefficient = {self.clustering_coefficient_of_graph():3g}\n\n'

        report += 'Minimum Spanning Trees:\n'
        report += f'by Prim-Jarnik:\n{self.minimum_spanning_tree(algorithm='prim-jarnik')}\n\n'
        report += f'by Kruskal:\n{self.minimum_spanning_tree(algorithm='kruskal')}\n\n'

        report += f'Node Names: {self.node_names}'

        node_measures = [self.eccentricity,
                         self.average_distance_of_vertex,
                         self.degree_centrality,
                         self.k_hop_centrality,
                         self.k_rank_centrality,
                         self.closeness_centrality,
                         self.betweenness_centrality,
                         self.stress_centrality,
                         self.clustering_coefficient_of_vertex]
        node_measures_names = ['Eccentricity',
                         'Average Distance',
                         'Degree Centrality',
                         'K-hop Centrality (k=2)',
                         'K-rank Centrality (k=2)',
                         'Closeness Centrality',
                         'Betweenness Centrality',
                         'Stress Centrality',
                         'Clustering Coefficient']
        for measure, name in zip(node_measures, node_measures_names):
            report += f'\n{name}\n'
            for node in range(len(self.matrix)):
                score = measure(node)
                report += f'\tof node "{self.node_names[node]}" = {score:3g}\n'

        report += f'\nExamples of Neighbourhood Overlap:\n'
        pairs = list(itertools.product(list(range(len(self.matrix))), list(range(len(self.matrix)))))
        indices = np.random.choice(list(range(len(pairs))), 10)
        for i in indices:
            node_1, node_2 = pairs[i]
            score = self.neighbourhood_overlap(node_1, node_2)
            report += f'\tof nodes "{self.node_names[node_1]} and {self.node_names[node_2]}" = {score:3g}\n'


        return report


if __name__ == '__main__':

    matrix_1 = np.array(
        [  # A, B, C, D, E, F, G
            [0, 1, 1, 1, 0, 0, 1], # A
            [1, 0, 1, 0, 0, 0, 0], # B
            [1, 1, 0, 1, 0, 0, 0], # C
            [1, 0, 1, 0, 1, 1, 1], # D
            [0, 0, 0, 1, 0, 0, 0], # E
            [0, 0, 0, 1, 0, 0, 0], # F
            [1, 0, 0, 1, 0, 0, 0]  # G
        ]
    )

    names = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}

    g = Graph(matrix_1, names)
    print(g.summary())

    matrix_2 = np.array(
        [  # A, B, C, D, E, F, G
            [0, 10, 0, 21, 25, 15, 0],  # A
            [10, 0, 11, 8, 13, 7, 9],  # B
            [0, 11, 0, 2, 3, 5, 15],  # C
            [21, 8, 2, 0, 14, 1, 1],  # D
            [25, 13, 3, 14, 0, 0, 6],  # E
            [15, 7, 5, 1, 0, 0, 3],  # F
            [0, 9, 15, 1, 6, 3, 0]  # G
        ]
    )

    g_2 = Graph(matrix_2, names)

    for node in names.values():
        print(f'Shortest Path from A to {node}: {g_2.shortest_paths("A", node)}')
