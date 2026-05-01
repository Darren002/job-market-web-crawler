"""
Main crawling agent that integrates state, action, and scoring mechanisms.

Key difference from simple crawler:
- Agent LEARNS which domains/URLs lead to job pages
- Agent EXPLOITS learned knowledge to prioritize high-value URLs
- Agent EXPLORES new URLs to discover new patterns
- Q-values are updated after each visit and USED for selection
- Multi-Armed Bandit (MAB) for balancing exploration/exploitation
"""
import logging
import random
import math
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field

from .state import State, URLFeatures, ContentFeatures
from .scorer import LinkScorer, DomainStatistics

logger = logging.getLogger(__name__)


@dataclass
class PrioritizedState:
    """State with computed priority score for selection."""
    state: State
    priority: float
    discovered_from: str = ""
    
    def __lt__(self, other):
        return self.priority < other.priority


class MultiArmedBandit:
    """
    Multi-Armed Bandit implementation for reward estimation.
    
    Uses UCB1 (Upper Confidence Bound) for exploration-exploitation balance:
    - Exploitation: Select arm with highest estimated reward
    - Exploration: Add confidence bonus for less-explored arms
    
    Also tracks Beta distribution parameters for Bayesian inference.
    """
    
    def __init__(self, exploration_constant: float = 2.0):
        self.exploration_constant = exploration_constant
        # Arm statistics: {arm_id: {'successes': int, 'failures': int, 'pulls': int}}
        self.arms: Dict[str, Dict] = {}
    
    def register_arm(self, arm_id: str):
        """Register a new arm (URL/domain) for tracking."""
        if arm_id not in self.arms:
            self.arms[arm_id] = {
                'successes': 0,
                'failures': 0,
                'pulls': 0,
                'total_reward': 0.0
            }
    
    def update(self, arm_id: str, reward: float):
        """Update arm statistics after observing reward."""
        self.register_arm(arm_id)
        arm = self.arms[arm_id]
        arm['pulls'] += 1
        arm['total_reward'] += reward
        
        if reward >= 0.5:
            arm['successes'] += 1
        elif reward < 0:
            arm['failures'] += 1
    
    def get_ucb_value(self, arm_id: str, total_pulls: int) -> float:
        """
        Compute UCB1 value for an arm.
        
        UCB1 = mean_reward + sqrt(2 * ln(total_pulls) / arm_pulls)
        
        This balances exploitation (high mean) with exploration (high uncertainty).
        """
        if arm_id not in self.arms:
            return float('inf')  # Unexplored arms get infinite bonus
        
        arm = self.arms[arm_id]
        if arm['pulls'] == 0:
            return float('inf')
        
        mean_reward = arm['total_reward'] / arm['pulls']
        
        # UCB1 exploration bonus
        if total_pulls > 0 and arm['pulls'] > 0:
            exploration_bonus = self.exploration_constant * math.sqrt(
                math.log(total_pulls) / arm['pulls']
            )
        else:
            exploration_bonus = float('inf')
        
        return mean_reward + exploration_bonus
    
    def get_thompson_sample(self, arm_id: str) -> float:
        """
        Thompson Sampling: Sample from posterior Beta distribution.
        
        For each arm, we maintain Beta(successes+1, failures+1).
        We sample from each arm's distribution and pick the highest sample.
        """
        if arm_id not in self.arms:
            return 1.0  # Unexplored arms get high sample
        
        arm = self.arms[arm_id]
        # Beta distribution parameters
        alpha = arm['successes'] + 1
        beta_param = arm['failures'] + 1
        
        # Sample from Beta distribution
        import random
        sample = random.betavariate(alpha, beta_param)
        return sample
    
    def get_mean(self, arm_id: str) -> float:
        """Get mean reward for an arm."""
        if arm_id not in self.arms or self.arms[arm_id]['pulls'] == 0:
            return 0.5  # Default for unexplored arms
        return self.arms[arm_id]['total_reward'] / self.arms[arm_id]['pulls']
    
    def get_best_arm(self, method: str = 'ucb', total_pulls: int = 0) -> Optional[str]:
        """Get the best arm using specified method."""
        if not self.arms:
            return None
        
        if method == 'ucb':
            best_arm = max(self.arms.keys(), 
                         key=lambda a: self.get_ucb_value(a, total_pulls))
        elif method == 'thompson':
            best_arm = max(self.arms.keys(),
                         key=lambda a: self.get_thompson_sample(a))
        else:  # greedy
            best_arm = max(self.arms.keys(),
                         key=lambda a: self.get_mean(a))
        
        return best_arm


class CrawlingAgent:
    """
    INTELLIGENT crawling agent with learned behavior.
    
    Key Intelligence:
    1. Q-LEARNING: Updates Q-values based on rewards (found job = +1, blocked = -0.2)
    2. EXPLOITATION: Prioritizes URLs from high-performing domains
    3. EXPLORATION: Sometimes picks random URLs to discover new patterns
    4. MAB (Multi-Armed Bandit): UCB1/Thompson Sampling for arm selection
    5. ADAPTIVE: Learns which keywords lead to job pages
    
    The agent's POLICY is: 
        Select URL with highest (Q-value + MAB_score + domain_score)
        With epsilon-greedy exploration
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        max_pages: int = 50,
        epsilon: float = 0.15,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        mab_method: str = 'ucb',  # 'ucb', 'thompson', 'greedy'
        skills: List[str] = None
    ):
        self.max_depth = max_depth
        self.max_pages = max_pages

        # RL parameters
        self.epsilon = epsilon  # Exploration rate (15% random)
        self.alpha = learning_rate  # Learning rate
        self.gamma = discount_factor  # Discount factor
        self.mab_method = mab_method  # MAB selection method

        # Skills for keyword influence
        self.skills = skills or ['python', 'java', 'javascript', 'aws', 'azure', 'cloud', 'machine learning', 'data scientist', 'developer', 'engineer', 'devops']

        # State tracking
        self.state_space: Dict[str, State] = {}
        self.domain_stats = DomainStatistics()
        self.scorer = LinkScorer(self.domain_stats)
        
        # Q-table: URL -> Q-value (expected reward)
        self.q_table: Dict[str, float] = {}
        
        # Multi-Armed Bandit for domain-level rewards
        self.domain_bandit = MultiArmedBandit(exploration_constant=2.0)
        self.keyword_bandit = MultiArmedBandit(exploration_constant=1.5)
        
        # Frontier with priorities
        self.frontier: List[PrioritizedState] = []
        
        # Visit history for learning
        self.visit_history: List[Tuple[str, float]] = []
        
        self.visited: set = set()
        self.domain_visits: Dict[str, int] = {}  # Track visits per domain
        self.step_count = 0
        
        # Exploration vs Exploitation tracking
        self.exploration_count = 0
        self.exploitation_count = 0
        self.mab_selection_count = 0
        
        # Keywords for classification
        self.job_keywords = ['job', 'jobs', 'career', 'careers', 'vacancy',
                           'position', 'positions', 'opening', 'openings', 'hiring']
        self.relevant_keywords = ['python', 'java', 'javascript', 'aws', 'azure',
                               'cloud', 'machine learning', 'data scientist',
                               'developer', 'engineer', 'devops', 'sql']
        
        # Learning statistics
        self.total_rewards = 0.0
        self.jobs_found = 0
        self.relevant_jobs_found = 0
        self.blocks_encountered = 0
    
    def initialize(self, seed_urls: List[str]):
        """Initialize agent with seed URLs."""
        logger.info(f"Initializing INTELLIGENT agent with {len(seed_urls)} seed URLs")
        logger.info(f"  epsilon (exploration): {self.epsilon}")
        logger.info(f"  learning_rate: {self.alpha}")
        logger.info(f"  discount_factor: {self.gamma}")
        
        for url in seed_urls:
            state = State.from_url(url, depth=0, discovered_at=0)
            self.state_space[url] = state
            self.q_table[url] = 0.5  # Initial Q-value (optimistic)
            
            priority = self._compute_priority(url, 0, parent_q=0.5)
            self.frontier.append(PrioritizedState(
                state=state,
                priority=priority,
                discovered_from="seed"
            ))
        
        # Sort frontier by priority
        self.frontier.sort(reverse=True)
        logger.info(f"Agent initialized. Frontier: {len(self.frontier)} URLs")
    
    PRIORITY_DOMAINS = {
        'my.jora.com': 0.5,
        'hiredly.com': 0.5,
        'jobstreet.com.my': 0.5,
    }

    PENALIZED_DOMAINS = {
        'jobcity.my': -0.8,
        'jobstreet.com': -0.3,
    }

    def _compute_priority(self, url: str, depth: int, parent_q: float = 0.5) -> float:
        """
        Compute selection priority using LEARNED knowledge + MAB + DOMAIN BALANCING.
        
        Priority = w1*Q_value + w2*MAB_score + w3*keyword_score + w4*depth_bonus + w5*domain_bonus - w6*domain_penalty
        """
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        q_value = self.q_table.get(url, 0.5)

        # STRICT DOMAIN PRIORITISATION
        priority_domains = {
            'my.jora.com': 0.5,
            'hiredly.com': 0.5,
            'jobstreet.com.my': 0.5,
        }
        penalized_domains = {
            'jobcity.my': -0.8,
            'jobstreet.com': -0.3,
        }
        domain_bonus = 0.0
        for priority_domain, bonus in priority_domains.items():
            if priority_domain in domain:
                domain_bonus = bonus
                break

        domain_penalty = 0.0
        for bad_domain, penalty in penalized_domains.items():
            if bad_domain in domain:
                domain_penalty = abs(penalty)
                break

        # DOMAIN BALANCING - Force diversity
        domain_visits = self.domain_visits.get(domain, 0)
        max_visits_per_domain = 3  # Reduced to force diversity
        if domain_visits >= max_visits_per_domain:
            overage = domain_visits - max_visits_per_domain
            domain_penalty += min(0.8, 0.4 + overage * 0.15)  # Stronger penalty

        # MAB-based domain score (UCB1 or Thompson), with balancing
        total_pulls = sum(arm['pulls'] for arm in self.domain_bandit.arms.values())

        if self.mab_method == 'ucb':
            mab_score = self.domain_bandit.get_ucb_value(domain, total_pulls)
        elif self.mab_method == 'thompson':
            mab_score = self.domain_bandit.get_thompson_sample(domain)
        else:
            mab_score = self.domain_bandit.get_mean(domain)

        # Normalize MAB score to 0-1 range, apply domain penalty
        mab_score = max(0.0, min(1.0, mab_score - domain_penalty))

        # Keyword matching score - ENHANCED with skills parameter
        url_lower = url.lower()
        keyword_score = 0.0
        keyword_found = None

        # Primary: Check job keywords
        for kw in self.job_keywords:
            if kw in url_lower:
                keyword_score += 0.25
                keyword_found = kw
                break

        # Secondary: Check skills (passed from keyword_set)
        for kw in self.skills:
            if kw.lower() in url_lower:
                keyword_score += 0.20
                keyword_found = kw
                break

        # Tertiary: Check relevant keywords
        for kw in self.relevant_keywords:
            if kw in url_lower:
                keyword_score += 0.10
                break

        # MAB for keyword patterns
        if keyword_found:
            self.keyword_bandit.register_arm(keyword_found)
            kw_total = sum(arm['pulls'] for arm in self.keyword_bandit.arms.values())
            if self.mab_method == 'ucb':
                kw_mab = self.keyword_bandit.get_ucb_value(keyword_found, kw_total)
            elif self.mab_method == 'thompson':
                kw_mab = self.keyword_bandit.get_thompson_sample(keyword_found)
            else:
                kw_mab = self.keyword_bandit.get_mean(keyword_found)
            keyword_score += 0.15 * kw_mab

        # URL Quality Score - bonus for job-indicative URLs
        url_quality_bonus = 0.0
        job_url_indicators = ['/job/', '/jobs/', '/career/', '/vacancy', '/position/']
        for ind in job_url_indicators:
            if ind in url_lower:
                url_quality_bonus += 0.15
                break

        # Depth penalty (closer to seed = better)
        depth_score = max(0.1, 1.0 - (depth * 0.1))

        # COMBINED PRIORITY (weighted sum)
        exploration_bonus = max(0.0, 1.0 - domain_penalty * 2)
        priority = (
            0.15 * q_value +
            0.15 * mab_score +
            0.20 * min(keyword_score, 1.0) +
            0.10 * url_quality_bonus +
            0.15 * domain_bonus +
            0.10 * depth_score +
            0.15 * exploration_bonus
        )

        return priority
    
    def _select_action(self) -> Optional[PrioritizedState]:
        """
        Select next URL using epsilon-greedy + MAB policy.
        
        With probability epsilon: EXPLORE (random selection)
        With probability 1-epsilon: EXPLOIT (best known URL using MAB)
        
        This is where the agent makes an INTELLIGENT decision.
        """
        if not self.frontier:
            return None
        
        # SMARTER SELECTION: Use domain balancing + adaptive exploration
        from urllib.parse import urlparse

        # In early steps, force exploration of multiple domains
        force_explore = self.step_count <= 10 and len(self.domain_visits) < 3

        if force_explore:
            # Find domains not yet visited
            visited_domains = set(self.domain_visits.keys())
            # Look for unexplored domain in frontier
            for ps in self.frontier:
                d = urlparse(ps.state.url_features.url).netloc
                if d not in visited_domains:
                    self.exploration_count += 1
                    return ps

        # Epsilon-greedy with adaptive rate
        if random.random() < self.epsilon:
            # EXPLORATION with preference for high-value unexplored domains
            high_value_candidates = []
            for ps in self.frontier:
                d = urlparse(ps.state.url_features.url).netloc
                # Bonus for unexplored domains
                if d not in self.domain_visits:
                    high_value_candidates.append(ps)
            if high_value_candidates:
                selected = max(high_value_candidates, key=lambda x: x.priority)
            else:
                selected = random.choice(self.frontier)
            self.exploration_count += 1
        else:
            # EXPLOITATION: Select highest priority
            self.frontier.sort(reverse=True)
            selected = self.frontier[0]
            self.exploitation_count += 1

        return selected
    
    def get_next_action(self) -> Optional[State]:
        """
        Get the next state to visit using the agent's POLICY.
        
        This method embodies the agent's intelligence:
        1. Select based on learned Q-values
        2. Update statistics
        3. Return the selected state
        """
        if not self.frontier:
            return None
        
        if len(self.visited) >= self.max_pages:
            logger.info(f"Reached max pages limit ({self.max_pages})")
            return None
        
        # Remove already visited URLs
        self.frontier = [ps for ps in self.frontier 
                        if ps.state.url_features.url not in self.visited]
        
        if not self.frontier:
            return None
        
        # Select using epsilon-greedy policy
        selected = self._select_action()
        
        if selected is None:
            return None
        
        url = selected.state.url_features.url
        
        # Remove from frontier
        self.frontier = [ps for ps in self.frontier 
                        if ps.state.url_features.url != url]
        
        # Mark as visited
        self.visited.add(url)
        self.step_count += 1

        # Track domain visits for domain balancing
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        self.domain_visits[domain] = self.domain_visits.get(domain, 0) + 1

        # Update state's visit count
        selected.state.visit_count += 1
        
        return selected.state
    
    def add_candidates(self, urls: List[str], parent_state: State, relevance: float = 0):
        """
        Add discovered URLs to the frontier with LEARNED priorities.
        
        The agent assigns priorities based on:
        1. Parent's Q-value (propagates value)
        2. URL's existing Q-value
        3. Domain quality
        4. Keyword matching
        """
        parent_url = parent_state.url_features.url
        parent_q = self.q_table.get(parent_url, 0.5)
        
        for url in urls:
            if url in self.visited:
                continue

            # Check if already in frontier
            if any(ps.state.url_features.url == url for ps in self.frontier):
                continue

            # FILTER: Skip low-value URLs
            url_lower = url.lower()
            low_value_patterns = ['/blog/', '/category/', '/help/', '/faq/', '/support/', '/terms', '/privacy', '/contact']
            if any(pat in url_lower for pat in low_value_patterns):
                continue

            depth = parent_state.crawl_depth + 1
            if depth > self.max_depth:
                continue
            
            # Initialize Q-value if new
            if url not in self.q_table:
                # Initial Q-value is parent's Q * discount (optimistic propagation)
                self.q_table[url] = parent_q * self.gamma
            
            # Create state if new
            if url not in self.state_space:
                state = State.from_url(url, depth=depth, discovered_at=self.step_count)
                state.estimated_relevance = self.q_table[url]
                self.state_space[url] = state
            else:
                state = self.state_space[url]
            
            # Compute priority using learned knowledge
            priority = self._compute_priority(url, depth, parent_q)
            
            # Add to frontier
            self.frontier.append(PrioritizedState(
                state=state,
                priority=priority,
                discovered_from=parent_url
            ))
    
    def record_outcome(self, url: str, is_job_page: bool, is_relevant: bool,
                       was_blocked: bool, content_relevance: float):
        """
        FIXED: 3-stage reward function.
        
        Reward structure:
        - +1.0 → RELEVANT_JOB (AI keywords found)
        - +0.3 → IRRELEVANT_JOB (job but not relevant)
        - 0.0 → NOT_JOB (non-job page)
        - -1.0 → BLOCKED or error
        """
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        is_new_domain = domain not in self.domain_visits

        if is_job_page and is_relevant:
            # RELEVANT_JOB: AI keywords found - SUCCESS
            reward = 1.0
            if is_new_domain:
                reward += 0.2
            self.jobs_found += 1
            self.relevant_jobs_found += 1  # Count relevant jobs separately
        elif is_job_page:
            # IRRELEVANT_JOB: job but not relevant - partial reward
            reward = 0.3
            self.jobs_found += 1  # Count all job pages found
        elif was_blocked:
            # BLOCKED but not a job
            reward = -0.2
            self.blocks_encountered += 1
        else:
            # NOT_JOB: no reward
            reward = 0.0

        self.total_rewards += reward

        # Get current Q-value
        current_q = self.q_table.get(url, 0.5)

        # Get best next Q-value (for TD learning)
        next_q_values = [self.q_table.get(ps.state.url_features.url, 0.5)
                        for ps in self.frontier]
        max_next_q = max(next_q_values) if next_q_values else 0.0

        # TD Update: Q(s) = Q(s) + alpha * (reward + gamma * max(Q(s')) - Q(s))
        td_target = reward + self.gamma * max_next_q
        td_error = td_target - current_q
        new_q = current_q + self.alpha * td_error

        # Update Q-table
        self.q_table[url] = new_q

        # Update Multi-Armed Bandit for domain
        self.domain_bandit.update(domain, reward)
        
        # Update MAB for matching keywords
        url_lower = url.lower()
        for kw in self.job_keywords:
            if kw in url_lower:
                self.keyword_bandit.update(kw, reward)
                break
        for kw in self.relevant_keywords:
            if kw in url_lower:
                self.keyword_bandit.update(kw, reward)
                break
        
        # Update state
        if url in self.state_space:
            self.state_space[url].estimated_relevance = new_q
            if content_relevance >= 0:
                self.state_space[url].update_content_features(
                    "", content_relevance, is_job_page
                )
        
        # Update domain statistics
        self.domain_stats.record_visit(url, is_relevant, is_job_page, was_blocked)
        
        # Update frontier priorities (re-sort based on new Q-values + MAB)
        for ps in self.frontier:
            ps.priority = self._compute_priority(
                ps.state.url_features.url,
                ps.state.crawl_depth,
                self.q_table.get(ps.state.url_features.url, 0.5)
            )
        
        logger.info(f"LEARNED: {url}")
        logger.info(f"  Old Q={current_q:.3f}, Reward={reward:.2f}, New Q={new_q:.3f}")
        logger.info(f"  TD_error={td_error:.3f}, Running reward={self.total_rewards:.2f}")
    
    def is_job_page(self, url: str) -> bool:
        """Check if URL likely contains job listings."""
        url_lower = url.lower()
        return any(kw in url_lower for kw in self.job_keywords)
    
    def is_relevant_page(self, url: str, html_content: str = "") -> bool:
        """Check if page is relevant to tech jobs."""
        url_lower = url.lower()
        content_lower = html_content.lower() if html_content else ""
        
        url_match = any(kw in url_lower for kw in self.relevant_keywords)
        content_match = any(kw in content_lower for kw in self.relevant_keywords)
        
        return url_match or content_match
    
    def get_stats(self) -> dict:
        """Get agent statistics including learning metrics and MAB."""
        avg_q = sum(self.q_table.values()) / len(self.q_table) if self.q_table else 0.0
        best_q = max(self.q_table.values()) if self.q_table else 0.0
        
        total_selections = self.exploration_count + self.exploitation_count + self.mab_selection_count
        exploration_pct = (self.exploration_count / max(total_selections, 1)) * 100
        exploitation_pct = (self.exploitation_count / max(total_selections, 1)) * 100
        mab_pct = (self.mab_selection_count / max(total_selections, 1)) * 100
        
        return {
            'total_steps': self.step_count,
            'pages_visited': len(self.visited),
            'frontier_size': len(self.frontier),
            'states_tracked': len(self.state_space),
            'q_values_learned': len(self.q_table),
            'avg_q_value': avg_q,
            'best_q_value': best_q,
            'total_rewards': self.total_rewards,
            'jobs_found': self.jobs_found,
            'relevant_jobs_found': self.relevant_jobs_found,
            'blocks_encountered': self.blocks_encountered,
            'learning_efficiency': self.jobs_found / max(self.step_count, 1),
            'domain_stats': self.domain_stats.get_stats_summary(),
            'exploration_count': self.exploration_count,
            'exploitation_count': self.exploitation_count,
            'mab_selection_count': self.mab_selection_count,
            'exploration_pct': exploration_pct,
            'exploitation_pct': exploitation_pct,
            'mab_pct': mab_pct,
            'mab_method': self.mab_method,
            'domain_bandit_arms': len(self.domain_bandit.arms),
            'keyword_bandit_arms': len(self.keyword_bandit.arms)
        }
    
    def print_learning_summary(self):
        """Print detailed learning statistics including MAB and exploration/exploitation."""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("AGENT LEARNING SUMMARY")
        print("="*60)
        print(f"Steps executed:        {stats['total_steps']}")
        print(f"Pages visited:         {stats['pages_visited']}")
        print(f"Q-values learned:      {stats['q_values_learned']}")
        print(f"Average Q-value:       {stats['avg_q_value']:.4f}")
        print(f"Best Q-value:          {stats['best_q_value']:.4f}")
        print("-"*60)
        print(f"Jobs found:            {stats['jobs_found']} (relevant: {stats['relevant_jobs_found']})")
        print(f"Blocks encountered:    {stats['blocks_encountered']}")
        print(f"Total rewards:        {stats['total_rewards']:.2f}")
        print(f"Learning efficiency:   {stats['learning_efficiency']:.2%}")
        
        # Exploration vs Exploitation
        print("\n" + "="*60)
        print("EXPLORATION VS EXPLOITATION ANALYSIS")
        print("="*60)
        total_selections = stats['exploration_count'] + stats['exploitation_count'] + stats['mab_selection_count']
        print(f"Total selections:      {total_selections}")
        print(f"  EXPLORATION steps:   {stats['exploration_count']:4d} ({stats['exploration_pct']:.1f}%)")
        print(f"  EXPLOITATION steps:  {stats['exploitation_count']:4d} ({stats['exploitation_pct']:.1f}%)")
        print(f"  MAB selection:       {stats['mab_selection_count']:4d} ({stats['mab_pct']:.1f}%)")
        print(f"\nMAB Method:            {stats['mab_method'].upper()}")
        print(f"Domain bandit arms:    {stats['domain_bandit_arms']}")
        print(f"Keyword bandit arms:   {stats['keyword_bandit_arms']}")
        print("="*60)
        
        # MAB Domain Performance
        print("\nMAB DOMAIN REWARDS (learned by bandit):")
        for arm_id, arm in sorted(self.domain_bandit.arms.items(), 
                                   key=lambda x: x[1]['total_reward']/max(x[1]['pulls'],1), 
                                   reverse=True)[:5]:
            if arm['pulls'] > 0:
                mean = arm['total_reward'] / arm['pulls']
                ucb = self.domain_bandit.get_ucb_value(arm_id, stats['total_steps'])
                print(f"  {arm_id}: mean={mean:.3f}, UCB={ucb:.3f}, pulls={arm['pulls']}")
        
        # Top Q-values
        if self.q_table:
            print("\nTOP 5 LEARNED URLS (by Q-value):")
            sorted_q = sorted(self.q_table.items(), key=lambda x: x[1], reverse=True)
            for url, q in sorted_q[:5]:
                print(f"  Q={q:.3f}: {url[:60]}...")
        
        # Domain performance
        print("\nDOMAIN PERFORMANCE (learned):")
        domain_stats = self.domain_stats.get_stats_summary()
        for domain, stats in domain_stats.get('domains', {}).items():
            if stats['total_visited'] > 0:
                print(f"  {domain}: {stats['job_pages_found']}/{stats['total_visited']} jobs "
                      f"({stats['success_rate']:.1%})")
    
    def get_frontier_urls(self) -> List[str]:
        """Get list of URLs in frontier."""
        return [ps.state.url_features.url for ps in self.frontier]
    
    def get_best_unvisited_url(self) -> Optional[str]:
        """Get the URL with highest Q-value that hasn't been visited."""
        best_url = None
        best_q = -float('inf')
        
        for url, q in self.q_table.items():
            if url not in self.visited and q > best_q:
                best_q = q
                best_url = url
        
        return best_url
    
    def reset(self):
        """Reset agent state for new crawl session."""
        self.state_space.clear()
        self.q_table.clear()
        self.frontier.clear()
        self.visit_history.clear()
        self.visited.clear()
        self.step_count = 0
        self.total_rewards = 0.0
        self.jobs_found = 0
        self.blocks_encountered = 0
        self.exploration_count = 0
        self.exploitation_count = 0
        self.mab_selection_count = 0
        self.domain_stats = DomainStatistics()
        self.scorer = LinkScorer(self.domain_stats)
        self.domain_bandit = MultiArmedBandit(exploration_constant=2.0)
        self.keyword_bandit = MultiArmedBandit(exploration_constant=1.5)
       
