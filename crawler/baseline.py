"""
Baseline crawler for comparison with the intelligent crawler.

Implements a simple FIFO (First-In-First-Out) crawler with no prioritization.
This serves as a baseline to demonstrate the improvement of the intelligent crawler.
"""
import logging
from collections import deque
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from crawler.fetcher import fetch_page
from crawler.extractor import extract_links

logger = logging.getLogger(__name__)


class BaselineCrawler:
    """
    Simple FIFO crawler - no prioritization, no learning.
    
    This baseline crawler:
    - Visits pages in the order they were discovered
    - Does not prioritize job-related URLs
    - Does not learn from past visits
    - Does not track domain statistics
    
    Used to compare against the intelligent crawler.
    """
    
    def __init__(self, max_pages: int = 50, max_depth: int = 10):
        self.max_pages = max_pages
        self.max_depth = max_depth
        
        self.queue: deque = deque()
        self.visited: set = set()
        
        self.stats = {
            'total_visited': 0,
            'job_pages_found': 0,
            'relevant_pages_found': 0,
            'blocked_pages': 0,
            'links_extracted': 0
        }
        
        self.job_keywords = ['job', 'jobs', 'career', 'careers', 'vacancy',
                           'position', 'positions', 'opening', 'openings', 'hiring']
        
        self.relevant_keywords = ['python', 'java', 'javascript', 'aws', 'azure',
                               'cloud', 'machine learning', 'data scientist',
                               'developer', 'engineer', 'devops', 'sql']
        
        self.depth_map: dict = {}
    
    def initialize(self, seed_urls: List[str]):
        """Initialize with seed URLs at depth 0."""
        for url in seed_urls:
            if url not in self.visited:
                self.queue.append((url, 0))
                self.depth_map[url] = 0
        logger.info(f"Baseline initialized with {len(seed_urls)} seeds")
    
    def crawl(self) -> List[Tuple[str, bool, bool, int]]:
        """
        Run the baseline crawl.
        
        Returns:
            List of (url, is_job_page, was_blocked, depth) tuples
        """
        results = []
        
        logger.info(f"Starting baseline crawl (max_pages={self.max_pages})")
        
        while self.queue and len(self.visited) < self.max_pages:
            url, depth = self.queue.popleft()
            
            if url in self.visited:
                continue
            
            self.visited.add(url)
            self.stats['total_visited'] += 1
            
            logger.info(f"[{self.stats['total_visited']}] Visiting: {url}")
            
            html, was_blocked = self._fetch_page(url)
            
            if html:
                links = extract_links(html, url)
                self.stats['links_extracted'] += len(links)
                
                for link in links:
                    if link not in self.visited and link not in dict(self.queue):
                        link_depth = depth + 1
                        if link_depth <= self.max_depth:
                            self.queue.append((link, link_depth))
                            self.depth_map[link] = link_depth
                
                is_job = self._is_job_page(url)
                is_relevant = self._is_relevant(url, html)
                
                if is_job:
                    self.stats['job_pages_found'] += 1
                    logger.info(f"  -> JOB PAGE")
                
                if is_relevant:
                    self.stats['relevant_pages_found'] += 1
                
                results.append((url, is_job, was_blocked, depth))
            else:
                self.stats['blocked_pages'] += 1
                logger.warning(f"  -> FAILED/BLOCKED")
                results.append((url, False, True, depth))
        
        logger.info(f"Baseline crawl complete. Visited {len(self.visited)} pages")
        return results
    
    def _fetch_page(self, url: str) -> Tuple[Optional[str], bool]:
        """Fetch a page and detect blocking."""
        html = fetch_page(url)
        
        was_blocked = False
        if html is None:
            was_blocked = True
        
        return html, was_blocked
    
    def _is_job_page(self, url: str) -> bool:
        """Check if URL contains job-related keywords."""
        url_lower = url.lower()
        return any(kw in url_lower for kw in self.job_keywords)
    
    def _is_relevant(self, url: str, html: str = "") -> bool:
        """Check if page is relevant to tech jobs."""
        url_lower = url.lower()
        content_lower = html.lower() if html else ""
        
        url_match = any(kw in url_lower for kw in self.relevant_keywords)
        content_match = any(kw in content_lower for kw in self.relevant_keywords)
        
        return url_match or content_match
    
    def get_stats(self) -> dict:
        """Get crawler statistics."""
        efficiency = (
            self.stats['relevant_pages_found'] / self.stats['total_visited']
            if self.stats['total_visited'] > 0 else 0.0
        )
        
        return {
            **self.stats,
            'efficiency': efficiency,
            'queue_remaining': len(self.queue),
            'visited_count': len(self.visited)
        }
    
    def print_summary(self):
        """Print crawl summary."""
        stats = self.get_stats()
        
        print("\n" + "="*50)
        print("BASELINE CRAWLER (FIFO) SUMMARY")
        print("="*50)
        print(f"Total Pages Visited:  {stats['total_visited']}")
        print(f"Job Pages Found:      {stats['job_pages_found']}")
        print(f"Relevant Pages:       {stats['relevant_pages_found']}")
        print(f"Blocked Pages:        {stats['blocked_pages']}")
        print(f"Links Extracted:      {stats['links_extracted']}")
        print(f"Efficiency:           {stats['efficiency']:.4f}")
        print("="*50)
