"""
Web environment model - treats the web as a graph.
"""
from collections import defaultdict
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


@dataclass
class PageNode:
    """Represents a page as a node in the web graph."""
    url: str
    domain: str
    depth: int = 0
    is_job_page: bool = False
    is_relevant: bool = False
    was_blocked: bool = False
    links: List[str] = field(default_factory=list)
    discovered_at: int = 0
    visited_at: Optional[int] = None


class WebGraph:
    """
    Represents the web as a graph structure.
    
    Nodes = pages
    Edges = links between pages
    
    This is a PARTIAL graph - we only build it as we discover pages.
    The environment is:
    - Dynamic: pages may change or disappear
    - Partially observable: we only see what we crawl
    """
    
    def __init__(self):
        self.nodes: Dict[str, PageNode] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self.domains: Dict[str, Set[str]] = defaultdict(set)
        
        self.total_nodes = 0
        self.total_edges = 0
    
    def add_node(self, url: str, depth: int = 0, discovered_at: int = 0) -> PageNode:
        """Add a new page node to the graph."""
        if url in self.nodes:
            return self.nodes[url]
        
        domain = urlparse(url).netloc
        node = PageNode(url=url, domain=domain, depth=depth, discovered_at=discovered_at)
        self.nodes[url] = node
        self.domains[domain].add(url)
        self.total_nodes += 1
        
        return node
    
    def add_edge(self, from_url: str, to_url: str):
        """Add a link (edge) between two pages."""
        self.edges[from_url].add(to_url)
        self.total_edges += 1
        
        if to_url not in self.nodes:
            self.add_node(to_url)
    
    def add_page_with_links(self, url: str, links: List[str], depth: int, 
                           discovered_at: int = 0):
        """Add a page and all its outgoing links."""
        node = self.add_node(url, depth, discovered_at)
        
        for link in links:
            self.add_edge(url, link)
            if link not in self.nodes:
                self.add_node(link, depth + 1, discovered_at)
    
    def mark_visited(self, url: str, visited_at: int):
        """Mark a node as visited."""
        if url in self.nodes:
            self.nodes[url].visited_at = visited_at
    
    def mark_job_page(self, url: str, is_job: bool):
        """Mark whether a page is a job page."""
        if url in self.nodes:
            self.nodes[url].is_job_page = is_job
    
    def mark_relevant(self, url: str, is_relevant: bool):
        """Mark whether a page is relevant."""
        if url in self.nodes:
            self.nodes[url].is_relevant = is_relevant
    
    def mark_blocked(self, url: str, was_blocked: bool):
        """Mark whether a page was blocked."""
        if url in self.nodes:
            self.nodes[url].was_blocked = was_blocked
    
    def get_domain_stats(self) -> Dict[str, dict]:
        """Get statistics per domain."""
        stats = {}
        for domain, urls in self.domains.items():
            pages = [self.nodes[u] for u in urls]
            stats[domain] = {
                'total_pages': len(urls),
                'job_pages': sum(1 for p in pages if p.is_job_page),
                'relevant_pages': sum(1 for p in pages if p.is_relevant),
                'blocked_pages': sum(1 for p in pages if p.was_blocked),
                'visited_pages': sum(1 for p in pages if p.visited_at is not None)
            }
        return stats
    
    def get_graph_stats(self) -> dict:
        """Get overall graph statistics."""
        visited_count = sum(1 for n in self.nodes.values() if n.visited_at is not None)
        job_count = sum(1 for n in self.nodes.values() if n.is_job_page)
        relevant_count = sum(1 for n in self.nodes.values() if n.is_relevant)
        blocked_count = sum(1 for n in self.nodes.values() if n.was_blocked)
        
        return {
            'total_nodes': self.total_nodes,
            'total_edges': self.total_edges,
            'visited_nodes': visited_count,
            'job_pages_found': job_count,
            'relevant_pages_found': relevant_count,
            'blocked_pages': blocked_count,
            'num_domains': len(self.domains)
        }
    
    def get_all_nodes(self) -> List[PageNode]:
        """Get all nodes in the graph."""
        return list(self.nodes.values())
    
    def get_visited_nodes(self) -> List[PageNode]:
        """Get all visited nodes."""
        return [n for n in self.nodes.values() if n.visited_at is not None]
