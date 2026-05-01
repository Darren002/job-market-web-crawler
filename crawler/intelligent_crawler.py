"""
Intelligent Crawler - Complete system integrating all components.

This crawler combines:
- CrawlingAgent: State representation, action selection, Q-learning
- WebGraph: Environment modeling as a graph
- DataLayer: TF-IDF vectorization and deduplication
- Evaluator: Metrics computation and CSV logging
"""
import logging
import re
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

NEGATIVE_JOB_KEYWORDS = [
    "clerk", "driver", "admin", "assistant", "telemarketing", "sales",
    "cashier", "waiter", "receptionist", "security guard", "cleaner",
    "helper", "labour", "packer", "merchandiser", "crew", "stocker",
    "cashier", "barista", "server", "attendant", "technician",
    "customer service", "production", "warehouse", "accounts", "costing"
]

def _has_negative_keyword(text: str) -> bool:
    """Check if text contains negative job keywords."""
    text_lower = text.lower()
    for neg in NEGATIVE_JOB_KEYWORDS:
        pattern = r'\b' + re.escape(neg) + r'\b'
        if re.search(pattern, text_lower):
            return True
    return False

from crawler.agent import CrawlingAgent
from crawler.env import WebGraph
from crawler.integration import DataLayer
from crawler.evaluator import Evaluator
from crawler.fetcher import fetch_page
from crawler.extractor import extract_links

logger = logging.getLogger(__name__)

# Parallel crawling support
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
_crawl_lock = threading.Lock()


class IntelligentCrawler:
    """
    Complete intelligent web crawling system.

    Integrates:
    - Agent with adaptive learning (Q-values, domain stats)
    - Environment modeling (web graph)
    - Data processing (TF-IDF, deduplication)
    - Evaluation (metrics)

    The crawl loop:
    1. Agent selects next URL (based on priority/Q-value)
    2. Fetcher retrieves page (with blocking detection)
    3. Extracts job content (title, company)
    4. Detects if page is a job page (URL + HTML signals)
    5. Agent records outcome and updates Q-values
    6. Web graph is updated
    7. New links are added to frontier with priorities
    8. Results are logged to evaluator
    """

    def __init__(
        self,
        skills: List[str] = None,
        max_pages: int = 50,
        max_depth: int = 10,
        output_dir: str = ".",
        keyword_set: str = "general",
        epsilon: float = 0.30
    ):
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.output_dir = output_dir
        self.keyword_set = keyword_set

        # Store skills for keyword influence
        self.skills = skills or self.default_skills

        # Core components - increased epsilon for better domain exploration
        self.agent = CrawlingAgent(
            max_depth=max_depth,
            max_pages=max_pages,
            epsilon=epsilon,
            skills=skills
        )
        self.web_graph = WebGraph()
        self.data_layer = DataLayer(skills=skills)
        self.evaluator = Evaluator(output_dir=output_dir)

        self.is_initialized = False

        # Learning history for adaptation demonstration
        self.q_value_history: List[Dict] = []
        self.jobs_found_over_time: List[int] = []

        # Default tech job skills
        self.default_skills = [
            "python", "java", "javascript", "typescript",
            "machine learning", "deep learning", "ai",
            "aws", "azure", "cloud", "docker", "kubernetes",
            "sql", "database", "postgresql", "mongodb",
            "react", "angular", "vue", "frontend",
            "backend", "api", "rest", "graphql",
            "git", "agile", "scrum", "devops"
        ]
    
    def initialize(self, seed_urls: List[str], training_pages: List[str] = None):
        """Initialize all components with seed URLs and training data."""
        logger.info("="*60)
        logger.info("INITIALIZING INTELLIGENT CRAWLER")
        logger.info("="*60)
        
        # Initialize data layer for TF-IDF
        if training_pages is None:
            training_pages = [
                "<html><body><h1>Software Engineer</h1><p>Python AWS Kubernetes</p></body></html>",
                "<html><body><h1>Data Scientist</h1><p>Machine learning Python SQL</p></body></html>",
                "<html><body><h1>DevOps Engineer</h1><p>Docker Kubernetes AWS Cloud</p></body></html>"
            ]
        
        self.data_layer.initialize(training_pages)
        logger.info(f"Data layer ready. Vocab size: {self.data_layer.get_stats()['vocab_size']}")
        
        # Initialize agent with seeds
        self.agent.initialize(seed_urls)
        
        # Build initial web graph
        for url in seed_urls:
            self.web_graph.add_node(url, depth=0, discovered_at=0)
        
        self.is_initialized = True
        logger.info("="*60)

    def crawl(self) -> List[Tuple[str, bool, bool, float, int]]:
        """
        Run the intelligent crawl.

        Returns:
            List of (url, is_job_page, was_blocked, relevance_score, depth) tuples
        """
        if not self.is_initialized:
            raise RuntimeError("Crawler not initialized. Call initialize() first.")

        results = []
        jobs_found_count = 0

        logger.info("="*60)
        logger.info(f"STARTING INTELLIGENT CRAWL (max_pages={self.max_pages})")
        logger.info(f"Keyword set: {self.keyword_set}")

        step = 0
        BATCH_SIZE = 3  # Fetch this many pages in parallel

        logger.info(f"Parallel fetch enabled (batch_size={BATCH_SIZE})")

        while step < self.max_pages:
            # SEQUENTIAL: select next batch of URLs (agent state not thread-safe)
            batch = []
            for _ in range(BATCH_SIZE):
                if step >= self.max_pages:
                    break
                _state = self.agent.get_next_action()
                if _state is None:
                    break
                batch.append((_state.url_features.url, _state.crawl_depth, _state))

            if not batch:
                logger.info("No more pages to crawl (frontier empty or max reached)")
                break

            logger.info(f"\n--- Fetching batch of {len(batch)} pages in parallel ---")

            # PARALLEL: fetch all URLs concurrently
            def _do_fetch(args):
                _url, _depth, _st = args
                try:
                    _html, _blocked = self._fetch_page(_url)
                    return (_url, _depth, _st, _html, _blocked, None)
                except Exception as _e:
                    return (_url, _depth, _st, None, True, _e)

            fetch_results = {}
            with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
                futures = {executor.submit(_do_fetch, item): item for item in batch}
                for future in as_completed(futures):
                    _url, _depth, _st, _html, _blocked, _err = future.result()
                    fetch_results[_url] = (_depth, _st, _html, _blocked, _err)

            # SEQUENTIAL: process results and update Q-values
            for _batch_url, _batch_depth, _batch_state in batch:
                if _batch_url not in fetch_results:
                    continue
                depth, state, html, was_blocked, _err = fetch_results[_batch_url]
                url = _batch_url
                step += 1
                logger.info(f"\n[{step}] Processing: {url}")

                if _err is not None:
                    logger.warning(f"  -> FETCH ERROR: {type(_err).__name__}: {str(_err)[:100]}")
                    self.web_graph.mark_visited(url, visited_at=step)
                    self.agent.record_outcome(url, False, False, False, -0.5)
                    self.evaluator.add_result(
                        url,
                        crawler_predicted_job=False,
                        crawler_predicted_relevant=False,
                        crawler_relevance_score=0.0,
                        was_blocked=False,
                        depth=depth,
                        html_content=None
                    )
                    results.append((url, False, False, 0.0, depth))
                    continue

                # Update web graph
                self.web_graph.mark_visited(url, visited_at=step)

                if was_blocked:
                    # Check URL for job pattern BEFORE skipping
                    url_lower = url.lower()
                    url_has_job = (
                        '/job/' in url_lower or 
                        '/job-search/' in url_lower or
                        '/position/' in url_lower or 
                        '/vacancy/' in url_lower
                    )
                
                    # Check for negative keywords in URL
                    has_negative = _has_negative_keyword(url)
                
                    if url_has_job:
                        # It's a blocked job page
                        if has_negative:
                            relevance = 0.1
                            label = "IRRELEVANT_JOB"
                            is_job = True
                        else:
                            relevance = 0.4
                            label = "IRRELEVANT_JOB"  # Can't verify relevance without content
                            is_job = True
                    else:
                        relevance = 0.0
                        label = "BLOCKED"
                        is_job = False
                
                    self.web_graph.mark_blocked(url, True)
                    self.agent.record_outcome(url, is_job, label == "RELEVANT_JOB", False , relevance)
                    self.evaluator.add_result(
                        url,
                        crawler_predicted_job=is_job,
                        crawler_predicted_relevant=label == "RELEVANT_JOB",
                        crawler_relevance_score=relevance,
                        was_blocked=True,
                        depth=depth,
                        html_content=html[:5000] if html else None
                    )
                    results.append((url, is_job, True, relevance, depth))
                
                    # Log with classification
                    if is_job:
                        logger.warning(f"  -> ⚠ BLOCKED JOB (URL): {url[:50]}... (rel={relevance:.2f})")
                    else:
                        logger.warning("  -> BLOCKED")
                    continue

                if html:
                    # Extract links
                    links = extract_links(html, url)
                    logger.info(f"  -> Found {len(links)} links")

                    # Extract job content (title, company)
                    job_title, company = self._extract_job_content(html, url)

                    # ==========================================
                    # STAGE 1: JOB DETECTION (URL-based first)
                    # ==========================================
                    # FIRST: Check URL for job signal (works even without HTML)
                    # STRICT: only /job/ = actual job page
                    url_lower = url.lower()
                    url_has_job = (
                        '/job/' in url_lower or
                        '/job-search/' in url_lower or
                        '/position/' in url_lower or
                        '/vacancy/' in url_lower
                    )
                
                    # Secondary: category/listing pages (not actual jobs)
                    url_is_listing = '-jobs' in url_lower or '/jobs-in-' in url_lower
                
                    is_blocked = self._is_blocked_page(url, html)
                    relevance = 0.0  # Default value
                
                    # Parse HTML only if available
                    if html:
                        html_is_job = self._is_job_page(url, html)
                    else:
                        html_is_job = False
                
                    # Job if URL has job OR HTML detects job
                    is_job = url_has_job or html_is_job
                
                    if not html:
                        # No HTML content - use URL-based detection
                        if url_has_job:
                            # Check URL for negative keywords
                            has_negative = _has_negative_keyword(url)
                            if has_negative:
                                relevance = 0.1  # Very low - negative keywords in URL
                                label = "IRRELEVANT_JOB"
                            else:
                                relevance = 0.5  # Moderate relevance from URL
                                label = "IRRELEVANT_JOB"
                        else:
                            relevance = 0.0
                            label = "NOT_JOB"
                    elif is_blocked:
                        # Blocked page - but if URL has job, still count as job
                        if url_has_job:
                            # Check URL for negative keywords
                            has_negative = _has_negative_keyword(url)
                            if has_negative:
                                relevance = 0.1  # Very low - negative keywords in URL
                                label = "IRRELEVANT_JOB"
                            else:
                                relevance = 0.4  # Lower confidence due to blocking
                                label = "IRRELEVANT_JOB"
                            is_job = True  # Override to true
                        else:
                            label = "BLOCKED"
                    elif not is_job:
                        label = "NOT_JOB"
                    else:
                        # ==========================================
                        # STAGE 2: RELEVANCE CLASSIFICATION
                        # ==========================================
                        relevance, is_relevant = self._compute_relevance(html, job_title)
                    
                        # ==========================================
                        # STAGE 3: FINAL CLASSIFICATION
                        # ==========================================
                        if is_relevant:
                            label = "RELEVANT_JOB"
                        else:
                            label = "IRRELEVANT_JOB"

                    # Track jobs found over time
                    if is_job:
                        jobs_found_count += 1
                    self.jobs_found_over_time.append(jobs_found_count)

                    # Record Q-value for learning history
                    current_q = self.agent.q_table.get(url, 0.5)
                    self.q_value_history.append({
                        'step': step,
                        'url': url[:50],
                        'q_value': current_q,
                        'label': label,
                        'jobs_found': jobs_found_count
                    })

                    # ==========================================
                    # RECORD OUTCOME WITH PROPER REWARDS
                    # ==========================================
                    if is_blocked and not url_has_job:
                        # Blocked AND no job URL signal = negative
                        reward = -0.3
                    elif is_blocked and url_has_job:
                        # Blocked but has job URL = partial reward
                        reward = 0.2
                    elif label == "RELEVANT_JOB":
                        reward = 1.0
                    elif label == "IRRELEVANT_JOB":
                        reward = 0.3
                    else:
                        reward = 0.0
                
                    self.agent.record_outcome(url, is_job, label == "RELEVANT_JOB", is_blocked, relevance)

                    # Update web graph
                    self.web_graph.add_page_with_links(url, links, depth, discovered_at=step)
                    self.web_graph.mark_job_page(url, is_job)
                    self.web_graph.mark_relevant(url, label == "RELEVANT_JOB")
                    if is_blocked:
                        self.web_graph.mark_blocked(url, True)

                    # ==========================================
                    # LOG WITH CLEAR CLASSIFICATION
                    # ==========================================
                    company_display = company[:30] if company else "N/A"
                
                    if is_blocked and url_has_job:
                        # Blocked but has job URL
                        logger.warning(f"  -> ⚠ BLOCKED JOB (URL): {url[:50]}... (rel={relevance:.2f})")
                    elif label == "BLOCKED":
                        logger.warning(f"  -> ⚠ BLOCKED PAGE")
                    elif label == "RELEVANT_JOB":
                        logger.info(f"  -> ✓ RELEVANT JOB: '{job_title[:35]}' @ {company_display} (rel={relevance:.2f})")
                    elif label == "IRRELEVANT_JOB":
                        logger.info(f"  -> ⚠ IRRELEVANT JOB: '{job_title[:35]}' @ {company_display} (rel={relevance:.2f})")
                    else:
                        logger.info(f"  -> ✗ NOT A JOB")

                    # Add to evaluator
                    self.evaluator.add_result(
                        url,
                        crawler_predicted_job=is_job,
                        crawler_predicted_relevant=label == "RELEVANT_JOB",
                        crawler_relevance_score=relevance,
                        was_blocked=is_blocked,
                        depth=depth,
                        html_content=html[:5000],
                        job_title=job_title,
                        company_name=company
                    )
                    results.append((url, is_job, is_blocked, relevance, depth))

                    # Add discovered links to agent's frontier
                    self.agent.add_candidates(links, state, relevance)
                else:
                    logger.warning("  -> FAILED to fetch")
                    self.evaluator.add_result(
                        url,
                        crawler_predicted_job=False,
                        crawler_predicted_relevant=False,
                        crawler_relevance_score=0.0,
                        was_blocked=True,
                        depth=depth,
                        html_content=None
                    )
                    results.append((url, False, True, 0.0, depth))

            # Finalize
        self.data_layer.processor.finalize()

        # Print adaptation demonstration
        self._print_adaptation_demonstration()

        logger.info("="*60)
        logger.info("CRAWL COMPLETE")
        logger.info("="*60)

        return results

    def _is_blocked_page(self, url: str, html: str = None) -> bool:
        """Detect if page is blocked (CAPTCHA, verification, etc.)."""
        if not html:
            return False
        
        html_lower = html.lower()
        
        # Check for ONLY clear blocked signals
        blocked_signals = [
            'captcha', 'access denied', 'cloudflare', 
            'unusual traffic', 'security check',
            'human verification', 'not a robot', 'please verify'
        ]
        for signal in blocked_signals:
            if signal in html_lower:
                return True
        
        return False

    def _is_job_page(self, url: str, html: str = None) -> bool:
        """
        STAGE 1: JOB DETECTION - STRICT
        
        ONLY returns TRUE if URL contains /job/ (individual job page)
        Everything else (listings, categories) = NOT a job
        """
        url_lower = url.lower()
        
        if not html:
            return False
        
        # STRICT: Only /job/ = individual job page
        if '/job/' in url_lower:
            return True
        
        # Also accept other specific job patterns
        other_job_patterns = ['/position/', '/vacancy/']
        for pattern in other_job_patterns:
            if pattern in url_lower:
                return True
        
        # Everything else = NOT a job
        return False

    def _compute_relevance(self, html: str, job_title: str = "", keywords: list = None) -> tuple[float, bool]:
        """
        STAGE 2: RELEVANCE CLASSIFICATION
        
        Computes relevance score based on keyword matching.
        - Requires AI keywords for true relevance
        - Tech keywords only give partial relevance
        - Applies negative filtering for irrelevant job titles
        
        Returns:
            relevance_score (0.05-1.0), is_relevant (bool)
        """
        import re
        if not html:
            return 0.05, False
        
        if keywords is None:
            keywords = self.skills
        
        # Extract ALL text from page for better matching
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Get full text content
        full_text = soup.get_text().lower() if soup else html.lower()
        
        # Add title if available
        if job_title:
            title_lower = job_title.lower()
        else:
            # Try to find title from page
            title_elem = soup.find('title') or soup.find('h1')
            title_lower = title_elem.get_text().lower() if title_elem else ""
        
        combined = full_text + ' ' + title_lower
        
        # REQUIRED AI Keywords for TRUE relevance
        ai_keywords = [
            'machine learning', 'deep learning', 'artificial intelligence',
            ' ai ', 'nlp', 'natural language', 'computer vision', 
            'data scientist', 'tensorflow', 'pytorch', 'keras', 'neural network',
            'ml engineer', 'ai engineer', 'data science', 'machine engineer'
        ]
        
        # Tech keywords - NOT relevant unless paired with AI keywords
        tech_keywords = [
            'engineer', 'developer', 'software', 'python', 'java', 'cloud',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'devops',
            'system', 'data', 'sql', 'api', 'backend', 'frontend'
        ]
        
        all_keywords = ai_keywords + tech_keywords
        
        # Count keyword matches - SEPARATE AI vs tech
        matches = 0
        matched_keywords = []
        
        # AI keywords get double weight
        ai_match_count = 0
        for kw in ai_keywords:
            if kw.strip() in combined:
                ai_match_count += 1
                matches += 2  # Double weight for AI keywords
                matched_keywords.append(kw)
        
        # Tech keywords get single weight
        tech_match_count = 0
        for kw in tech_keywords:
            if kw.strip() in combined:
                tech_match_count += 1
                matches += 1
                matched_keywords.append(kw)
        
                # Calculate relevance score - VERY STRICT
        # ONLY AI keywords = RELEVANT
        if ai_match_count >= 1:
            # AI/ML job - TRUE relevance
            relevance = min(1.0, 0.7 + matches * 0.1)
            is_relevant = True
        else:
            # NO AI keywords = NOT RELEVANT (regardless of tech keywords)
            relevance = 0.05
            is_relevant = False
        
        # NEGATIVE FILTER: penalize irrelevant job titles
        negative_titles = ['clerk', 'driver', 'admin', 'assistant', 'sales', 
                       'telemarketing', 'cashier', 'waiter', 'receptionist']
        for neg in negative_titles:
            if neg in title_lower:
                relevance *= 0.2
                is_relevant = False
                break
        
        relevance = max(0.05, min(1.0, relevance))
        
        return relevance, is_relevant

    def _extract_job_content(self, html: str, url: str = "") -> Tuple[str, str]:
        """Extract job title and company ONLY from actual job pages."""
        title = ""
        company = ""

        if not html:
            return title, company
        
        # Only extract from actual job pages
        url_lower = url.lower() if url else ""
        is_job_url = '/job/' in url_lower or '/position/' in url_lower or '/vacancy/' in url_lower
        
        if not is_job_url:
            return title, company

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            # Find title - look for specific job title patterns FIRST
            title_elem = (
                soup.find('h1', {'class': lambda x: x and 'title' in str(x).lower()}) or
                soup.find('h1', {'class': lambda x: x and 'job' in str(x).lower()}) or
                soup.find('h1', {'class': lambda x: x and 'header' in str(x).lower()}) or
                soup.find('h1')
            )
            if title_elem:
                title = title_elem.get_text(strip=True)
            
            # Find company - look for company name patterns
            company_elem = (
                soup.find('a', {'class': lambda x: x and 'employer' in str(x).lower() and 'nav' not in str(x).lower()}) or
                soup.find('span', {'class': lambda x: x and 'employer' in str(x).lower()}) or
                soup.find('div', {'class': lambda x: x and 'company' in str(x).lower() and 'nav' not in str(x).lower()}) or
                soup.find('span', {'class': lambda x: x and 'posted' in str(x).lower()}) or
                soup.find('a', {'class': lambda x: x and 'company' in str(x).lower()})
            )
            if company_elem:
                company = company_elem.get_text(strip=True)

            # Filter out noise - common non-company text
            noise_phrases = ['login', 'sign up', 'job', 'vacancy', 'search', 'browse', 'home', 'menu', 'register', 'employer', 'jobseeker', 'click', 'apply']
            if company:
                company_lower = company.lower()
                if any(phrase in company_lower for phrase in noise_phrases) or len(company) < 2:
                    company = ""

            # Filter title noise
            if title:
                title_lower = title.lower()
                # Filter page headers, not job titles
                noise_titles = ['login', 'sign in', 'sign up', 'register', 'home', 'error', '404', 'page not found', 'search', 'jobs', 'browse', 'found', 'now']
                if any(t in title_lower for t in noise_titles):
                    title = ""

        except Exception:
            pass

        # Clean up
        title = title[:200] if title else ""
        company = company[:100] if company else ""

        return title, company

    def _print_adaptation_demonstration(self):
        """Print learning behavior to show adaptation."""
        if not self.q_value_history:
            return

        print("\n" + "="*60)
        print("ADAPTATION DEMONSTRATION")
        print("="*60)

        # Show Q-value updates over time
        print("\nQ-VALUE LEARNING (first 10 steps):")
        for entry in self.q_value_history[:10]:
            label = entry.get('label', 'UNKNOWN')
            status = label[:4] if label else "---"
            print(f"  Step {entry['step']}: Q={entry['q_value']:.3f} [{status}] {entry['url'][:40]}...")

        # Show jobs found over time
        print("\nJOBS FOUND OVER TIME:")
        if self.jobs_found_over_time:
            for i in range(0, len(self.jobs_found_over_time), max(1, len(self.jobs_found_over_time)//5)):
                if i < len(self.jobs_found_over_time):
                    print(f"  Step {i+1}: {self.jobs_found_over_time[i]} jobs found")

        # Calculate improvement
        if len(self.q_value_history) >= 10:
            early_jobs = sum(1 for e in self.q_value_history[:5] if 'JOB' in e.get('label', ''))
            late_jobs = sum(1 for e in self.q_value_history[-5:] if 'JOB' in e.get('label', ''))
            if early_jobs > 0:
                improvement = ((late_jobs - early_jobs) / early_jobs) * 100
                print(f"\nImprovement (last 5 vs first 5): {improvement:+.1f}%")

    def print_keyword_matched_jobs(self):
        """Print jobs that match the selected keyword domain."""
        if not hasattr(self, 'evaluator') or not self.evaluator.results:
            return

        # Show ALL actual job pages (ground truth)
        actual_jobs = [(r.url, r.job_title, r.company_name, r.crawler_relevance_score)
                    for r in self.evaluator.results if r.ground_truth_is_job]

        print("\n" + "="*60)
        print("ACTUAL JOB PAGES FOUND")
        print("="*60)
        if actual_jobs:
            for i, (url, title, company, rel) in enumerate(actual_jobs, 1):
                title_disp = (title[:50] if title and title != "N/A" else "N/A")
                company_disp = (company[:30] if company and company != "N/A" else "N/A")
                print(f"  {i}. {title_disp}")
                if company_disp != "N/A":
                    print(f"     Company: {company_disp}")
                print(f"     URL: {url[:70]}")
                print(f"     Relevance: {rel:.3f}")
                print()
        else:
            print("  No actual job pages found.")

        # Now show keyword-matched jobs
        print("\n" + "="*60)
        print(f"KEYWORD-MATCHED JOBS ({self.keyword_set.upper()})")
        print("="*60)

        matched_jobs = []
        for result in self.evaluator.results:
            # Use ground truth OR crawler prediction
            is_job = result.ground_truth_is_job or result.crawler_predicted_job
            if is_job:
                url = result.url
                title = result.job_title or 'N/A'
                company = result.company_name or 'N/A'
                relevance = result.crawler_relevance_score

                # For 'general', show all jobs
                if self.keyword_set == 'general':
                    matched_jobs.append((url, title, company, relevance))
                elif self.keyword_set == 'ai':
                    # More inclusive AI keywords - includes tech roles
                    ai_keywords = ['ai', 'ml', 'machine learning', 'deep learning', 'artificial intelligence', 
                                  'data scientist', 'data analyst', 'data engineer', 'nlp', 'neural', 
                                  'tensorflow', 'python', 'analytics', 'software', 'engineer', 'developer',
                                  'software', 'engineer', 'developer', 'programmer', 'technical']
                    if any(kw in url.lower() or kw in title.lower() for kw in ai_keywords):
                        matched_jobs.append((url, title, company, relevance))
                elif self.keyword_set == 'cloud':
                    cloud_keywords = ['aws', 'azure', 'gcp', 'cloud', 'devops', 'docker', 'kubernetes', 'k8s', 'sre', 'serverless']
                    if any(kw in url.lower() or kw in title.lower() for kw in cloud_keywords):
                        matched_jobs.append((url, title, company, relevance))
                elif self.keyword_set == 'web':
                    web_keywords = ['javascript', 'react', 'angular', 'vue', 'frontend', 'backend', 'fullstack', 'full stack', 'http', 'api', 'node', 'django']
                    if any(kw in url.lower() or kw in title.lower() for kw in web_keywords):
                        matched_jobs.append((url, title, company, relevance))

        print(f"\nTotal job pages found: {len(matched_jobs)}")

        if matched_jobs:
            print(f"\nJobs matching {self.keyword_set.upper()} keywords:")
            for i, (url, title, company, rel) in enumerate(matched_jobs[:15], 1):
                title_short = title[:45] if title and title != "N/A" else "N/A"
                print(f"  {i}. {title_short}")
                if company and company != "N/A":
                    print(f"     Company: {company[:30]}")
                print(f"     URL: {url[:65]}")
                print(f"     Relevance: {rel:.3f}")
                print()

        if len(matched_jobs) > 15:
            print(f"\n... and {len(matched_jobs) - 15} more jobs")

        print("="*60)
    
    def _fetch_page(self, url: str) -> Tuple[Optional[str], bool]:
        """Fetch page with blocking detection."""
        html = fetch_page(url)
        
        was_blocked = (html is None)
        
        return html, was_blocked
    
    def generate_ground_truth(self, job_urls: List[str]):
        """Set ground truth URLs for evaluation."""
        self.evaluator.set_ground_truth(job_urls)
    
    def save_results(self):
        """Save all results to files in output directory."""
        import os
        
        # Save visited URLs to CSV
        csv_path = os.path.join(self.output_dir, "visited_urls.csv")
        self.evaluator.save_to_csv(csv_path)
        
        # Save evaluation report
        report_path = os.path.join(self.output_dir, "evaluation_results.txt")
        self.evaluator.save_evaluation_report(report_path)
        
        # Save ground truth
        gt_path = os.path.join(self.output_dir, "ground_truth.csv")
        self.evaluator.save_ground_truth(gt_path)
    
    def print_summary(self, baseline_results: List[Tuple] = None):
        """Print comprehensive summary."""
        # Agent learning summary
        self.agent.print_learning_summary()

        # Evaluator stats
        self.evaluator.print_summary()

        # Print keyword-matched jobs at bottom
        self.print_keyword_matched_jobs()

        # Web graph stats
        graph_stats = self.web_graph.get_graph_stats()

        print("\n" + "="*60)
        print("WEB GRAPH STATS")
        print("="*60)
        print(f"Total nodes:          {graph_stats['total_nodes']}")
        print(f"Total edges:           {graph_stats['total_edges']}")
        print(f"Domains discovered:    {graph_stats['num_domains']}")

        # Domain breakdown
        domain_stats = self.web_graph.get_domain_stats()
        if domain_stats:
            print("\nDOMAIN BREAKDOWN:")
            for domain, stats in sorted(domain_stats.items(), 
                                        key=lambda x: x[1]['job_pages'],
                                        reverse=True)[:5]:
                print(f"  {domain}: {stats['visited_pages']} visited, "
                      f"{stats['job_pages']} job pages")
    
    def get_domain_stats(self) -> dict:
        """Get per-domain statistics."""
        return self.web_graph.get_domain_stats()
    
    def get_agent_stats(self) -> dict:
        """Get agent learning statistics."""
        return self.agent.get_stats()