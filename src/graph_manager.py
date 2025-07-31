import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any
import sys

import networkx as nx
from pyvis.network import Network
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.colors as mcolors


class GraphManager:
    def __init__(self, edges: List[Tuple[str, str]]):
        print("Initializing Graph Manager...")
        self.edge_weights = Counter(edges)
        self.graph = self._build_analytical_graph()
        print("Graph Manager initialized.")

    # ... (from_gml, _build_analytical_graph, etc. are unchanged) ...

    def generate_chapter_wise_report(self, chapter_data: Dict[int, List[Tuple[str, str]]], top_n: int = 5) -> str:
        """
        Analyzes each chapter individually to find the most important characters.
        'Importance' here is defined by the highest number of interactions (degree centrality).
        """
        report_lines = [f"\n--- Top {top_n} Most Important Characters by Chapter ---"]

        for chapter_idx, edges in sorted(chapter_data.items()):
            if not edges:
                continue

            report_lines.append(f"\nChapter {chapter_idx + 1}:")

            # Create a temporary graph for this chapter only
            chapter_g = nx.Graph()
            for char1, char2 in edges:
                chapter_g.add_edge(char1, char2)

            if not chapter_g.nodes:
                continue

            # Calculate degree centrality for just this chapter's graph
            degrees = nx.degree_centrality(chapter_g)
            sorted_degrees = sorted(degrees.items(), key=lambda item: item[1], reverse=True)

            for i, (char, score) in enumerate(sorted_degrees[:top_n]):
                report_lines.append(f"  {i + 1}. {char:<30} (Score: {score:.4f})")

        return "\n".join(report_lines)

    # ... (The rest of the class, including the full analysis and visualization methods, remains the same) ...

    @classmethod
    def from_gml(cls, gml_path: Path) -> 'GraphManager':
        print(f"Loading graph from {gml_path}...")
        try:
            G = nx.read_gml(str(gml_path))
            for u, v, data in G.edges(data=True):
                if 'details' in data and isinstance(data['details'], str):
                    data['details'] = json.loads(data['details'])
            all_edges = []
            for u, v, data in G.edges(data=True):
                weight = data.get('weight', 1)
                all_edges.extend([(u, v)] * weight)
            instance = cls(all_edges)
            instance.graph = G
            return instance
        except Exception as e:
            print(f"FATAL: Could not read or parse the graph file. Error: {e}")
            sys.exit(1)

    def _build_analytical_graph(self) -> nx.Graph:
        G = nx.Graph()
        for (char1, char2), weight in self.edge_weights.items():
            G.add_edge(char1, char2, weight=weight, value=weight)
        return G

    def _add_node_attributes(self):
        if not self.graph.nodes: return
        try:
            partition = community_louvain.best_partition(self.graph, weight='weight')
            num_communities = len(set(partition.values()))
            nx.set_node_attributes(self.graph, partition, 'group')
            colors = cm.get_cmap('tab20', num_communities)
            self.color_map = [mcolors.to_hex(colors(i)) for i in range(num_communities)]
        except Exception as e:
            print(f"Warning: Community detection failed. All nodes will be one color. Error: {e}")
            nx.set_node_attributes(self.graph, 0, 'group')
            self.color_map = ["#97c2fc"]

        degrees = nx.degree_centrality(self.graph)
        min_degree, max_degree = (min(degrees.values(), default=0), max(degrees.values(), default=1))

        for node in self.graph.nodes():
            self.graph.nodes[node]['color'] = self.color_map[self.graph.nodes[node]['group']]
            total_interactions = self.graph.degree(node, weight='weight')
            self.graph.nodes[node]['title'] = (
                f"<b>{node}</b><br>Community: {self.graph.nodes[node]['group']}<br>Total Interactions: {total_interactions}")
            if max_degree > min_degree:
                normalized_degree = (degrees.get(node, 0) - min_degree) / (max_degree - min_degree)
                self.graph.nodes[node]['size'] = 15 + normalized_degree * 40
            else:
                self.graph.nodes[node]['size'] = 15

    def _add_legend_nodes(self, net: Network):
        if not hasattr(self, 'color_map') or len(self.color_map) <= 1: return
        try:
            height_val = int(net.height.replace("px", "").strip())
            width_val = 1000
            legend_x = -width_val / 2 * 0.9
            legend_y = -height_val / 2 * 0.9
            for i, color in enumerate(self.color_map):
                net.add_node(f"community_{i}", label=f"Community {i}", color=color, size=10, x=legend_x,
                             y=legend_y + (i * 25), fixed=True, physics=False)
        except Exception as e:
            print(f"Warning: Could not generate visualization legend. Error: {e}")

    def generate_full_analysis_report(self, top_n: int = 10) -> str:
        if not self.graph.nodes: return "Graph is empty. No analysis can be performed."
        report_lines = ["--- Character Network Analysis Report ---\n"]
        report_lines.append(f"Total Characters (Nodes): {self.graph.number_of_nodes()}")
        report_lines.append(f"Total Unique Relationships (Edges): {self.graph.number_of_edges()}")
        report_lines.append(f"\n--- Top {top_n} Relationships by Interaction Count ---")
        for (char1, char2), weight in self.edge_weights.most_common(top_n):
            report_lines.append(f"  {weight:<5} | {char1} -- {char2}")
        for name, result in self._get_all_centralities(top_n=top_n).items():
            report_lines.append(f"\n--- Top {top_n} Characters by {name} ---")
            for char, score in result:
                report_lines.append(f"  {char:<30} | Score: {score:.4f}")
        return "\n".join(report_lines)

    def _get_all_centralities(self, top_n: int = 10) -> Dict[str, Any]:
        if not self.graph.nodes: return {}
        try:
            eigenvector = nx.eigenvector_centrality(self.graph, weight='weight', max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            eigenvector = {}
        return {
            "Degree Centrality": sorted(nx.degree_centrality(self.graph).items(), key=lambda item: item[1],
                                        reverse=True)[:top_n],
            "Betweenness Centrality": sorted(nx.betweenness_centrality(self.graph, weight='weight').items(),
                                             key=lambda item: item[1], reverse=True)[:top_n],
            "Eigenvector Centrality": sorted(eigenvector.items(), key=lambda item: item[1], reverse=True)[:top_n],
        }

    def save_interactive_visualization(self, output_path: Path):
        print(f"Generating interactive visualization... -> {output_path}")
        self._add_node_attributes()
        net = Network(height="900px", width="100%", bgcolor="#1a1a1a", font_color="white", cdn_resources='in_line')
        net.from_nx(self.graph)
        self._add_legend_nodes(net)
        net.force_atlas_2based(spring_length=150)
        net.show_buttons(filter_=['physics', 'nodes', 'edges'])
        try:
            net.save_graph(str(output_path))
            print("Visualization saved successfully.")
        except Exception as e:
            print(f"An error occurred during visualization: {e}")