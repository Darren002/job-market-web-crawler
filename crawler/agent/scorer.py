"""
Adaptive scoring mechanism for the crawling agent.
Implements Q-value-like scoring with domain statistics and priority updates.
"""
from typing import Dict, List, Optional
from collections import defaultdict
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DomainStatistics:
    """Tracks statistics per domain for adaptive learning."""
    
    def __init__(self):
        self.domain_stats: Dict[str, dict] = defaultdict(lambda: {
            'total_visited': 0,
            'relevant_found': 0,
            'job_pages_found': 0,
            'total_blocks': 0,
            'avg_relevance': 0.0,
            'success_rate': 0.0
        })
        self.global_stats = {
            'total_visited': 0,
            'relevant_found': 0,
            'job_pages_found': 0
        }
    
    def record_visit(self, url: str, is_relevant: bool, is_job: bool, was_blocked: bool):
        """Record outcome of visiting a URL."""
        domain = urlparse(url).netloc
        
        self.domain_stats[domain]['total_visited'] += 1
        self.global_stats['total_visited'] += 1
        
        if is_relevant:
            self.domain_stats[domain]['relevant_found'] += 1
            self.global_stats['relevant_found'] += 1
        
        if is_job:
            self.domain_stats[domain]['job_pages_found'] += 1
            self.global_stats['job_pages_found'] += 1
        
        if was_blocked:
            self.domain_stats[domain]['total_blocks'] += 1
        
        self._update_success_rate(domain)
    
    def _update_success_rate(self, domain: str):
        """Update success rate for a domain."""
        stats = self.domain_stats[domain]
        if stats['total_visited'] > 0:
            stats['success_rate'] = stats['job_pages_found'] / stats['total_visited']
            if stats['total_visited'] > 0:
                stats['avg_relevance'] = (
                    (stats['avg_relevance'] * (stats['total_visited'] - 1) + 
                     (1.0 if stats['relevant_found'] > 0 else 0.0)) 
                    / stats['total_visited']
                )
    
    def get_domain_score(self, domain: str) -> float:
        """
        Get quality score for a domain based on historical performance.
        Higher score = domain tends to have more relevant pages.
        """
        stats = self.domain_stats[domain]
        
        if stats['total_visited'] < 3:
            return 0.5
        
        relevance_weight = 0.4
        job_weight = 0.4
        success_weight = 0.2
        
        relevance_score = min(stats['relevant_found'] / max(stats['total_visited'], 1), 1.0)
        job_score = min(stats['job_pages_found'] / max(stats['total_visited'], 1), 1.0)
        block_penalty = max(0, 1 - stats['total_blocks'] / max(stats['total_visited'], 1))
        
        return (relevance_weight * relevance_score + 
                job_weight * job_score + 
                success_weight * block_penalty)
    
    def get_stats_summary(self) -> dict:
        """Get summary of all statistics."""
        return {
            'global': self.global_stats.copy(),
            'domains': dict(self.domain_stats)
        }


class LinkScorer:
    """
    Scores links based on multiple features for priority assignment.
    Implements a simple Q-value approximation.
    """
    
    def __init__(self, domain_stats: DomainStatistics):
        self.domain_stats = domain_stats
        self.q_values: Dict[str, float] = defaultdict(lambda: 0.5)
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        
        self.job_keywords = ['job', 'jobs', 'career', 'careers', 'vacancy', 
                            'position', 'positions', 'opening', 'openings', 'hiring']
        self.tech_keywords = ['python', 'java', 'javascript', 'aws', 'azure', 
                              'cloud', 'machine-learning', 'ml', 'ai', 'data',
                              'developer', 'engineer', 'devops', 'kubernetes']
    
    def compute_priority(self, url: str, depth: int, parent_relevance: float = 0) -> tuple[float, str]:
        """
        Compute priority score for a URL.
        
        Returns:
            (priority_score, reason)
        """
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path.lower()
        
        url_score = self._compute_url_score(url, path)
        depth_penalty = self._compute_depth_penalty(depth)
        domain_score = self.domain_stats.get_domain_score(domain)
        
        q_value = self.q_values.get(url, 0.5)
        
        priority = (
            0.30 * url_score +
            0.20 * depth_penalty +
            0.20 * domain_score +
            0.15 * q_value +
            0.15 * parent_relevance
        )
        
        reason = f"url={url_score:.2f}, depth={depth_penalty:.2f}, domain={domain_score:.2f}, q={q_value:.2f}"
        
        return priority, reason
    
    def _compute_url_score(self, url: str, path: str) -> float:
        """Score based on URL keywords."""
        score = 0.5
        
        for kw in self.job_keywords:
            if kw in path:
                score += 0.15
        
        for kw in self.tech_keywords:
            if kw in path:
                score += 0.10
        
        return min(score, 1.0)
    
    def _compute_depth_penalty(self, depth: int) -> float:
        """
        Penalize deeper pages (job listings often close to root).
        But don't penalize too harshly - some jobs are in subdirectories.
        """
        if depth <= 2:
            return 1.0
        elif depth <= 4:
            return 0.8
        elif depth <= 6:
            return 0.5
        else:
            return max(0.2, 0.3 ** (depth - 6))
    
    def update_q_value(self, url: str, reward: float, next_state_max_q: float = 0):
        """
        Update Q-value using simple TD-like update.
        
        Q(s) = Q(s) + alpha * (reward + gamma * max(Q(s')) - Q(s))
        """
        current_q = self.q_values.get(url, 0.5)
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_state_max_q - current_q
        )
        self.q_values[url] = new_q
        logger.debug(f"Updated Q-value for {url}: {current_q:.3f} -> {new_q:.3f}")
    
    def get_reward(self, is_job_page: bool, is_relevant: bool, was_blocked: bool) -> float:
        """
        Compute reward signal based on page outcome.
        
        Rewards:
            +1.0 : Found a job page
            +0.5 : Relevant but not job page
            -0.2 : Blocked/failed
            +0.0 : Normal navigation
        """
        if was_blocked:
            return -0.2
        if is_job_page:
            return 1.0
        if is_relevant:
            return 0.5
        return 0.0
    
    def get_q_value(self, url: str) -> float:
        """Get current Q-value estimate for a URL."""
        return self.q_values.get(url, 0.5)
