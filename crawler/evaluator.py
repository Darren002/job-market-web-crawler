"""
Evaluator module for the intelligent web crawler.

This module provides REALISTIC evaluation:
1. Ground Truth: Created from STRICT keyword matching (definitive job pages)
2. Predictions: The crawler's decisions (URL-based heuristics)
3. Confusion Matrix: TP / FP / FN / TN
4. Metrics: Precision, Recall, F1 computed correctly

CATEGORY SPLITTING:
- AI URLs: URLs containing AI/ML keywords (regardless of job status)
- Cloud URLs: URLs containing Cloud/DevOps keywords (regardless of job status)
- For each category: compute TP/FP/FN/TN for job page detection
"""
import csv
import logging
import os
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class PageResult:
    """Result of crawling a single page."""
    url: str
    domain: str
    depth: int = 0

    # Extracted content
    job_title: str = ""
    company_name: str = ""

    # What the crawler PREDICTED
    crawler_predicted_job: bool = False
    crawler_predicted_relevant: bool = False
    crawler_relevance_score: float = 0.0

    # Ground truth (what it ACTUALLY is)
    ground_truth_is_job: bool = False

    # Category based on URL keywords (AI, Cloud, Other)
    url_category: str = "other"  # 'ai', 'cloud', 'other'

    # Error classification for false positives
    is_false_positive: bool = False
    error_type: str = ""  # 'listing_page', 'keyword_noise', 'navigation', 'none'

    # Environment info
    was_blocked: bool = False
    html_content: Optional[str] = None


@dataclass
class ConfusionMatrix:
    """Standard confusion matrix components."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0


@dataclass
class EvaluationMetrics:
    """Comprehensive metrics for crawler evaluation."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    specificity: float
    
    # Counts
    total_pages: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    
    # Additional metrics
    job_pages_in_ground_truth: int
    job_pages_found_by_crawler: int
    relevant_pages_found: int
    blocked_pages: int
    
    # Efficiency
    efficiency: float
    exploration_rate: float


class Evaluator:
    """
    Realistic crawler evaluator with proper TP/FP/FN/TN calculations.
    
    GROUND TRUTH DEFINITION (Strict):
    A page is a "job page" if URL contains /job/ or /jobs/ or /career/ etc.
    
    CATEGORY DEFINITION:
    - AI: URL contains AI/ML keywords (machine learning, data scientist, etc.)
    - Cloud: URL contains Cloud/DevOps keywords (AWS, kubernetes, docker, etc.)
    - Other: Neither
    
    For each category, we compute how well the crawler identifies JOB PAGES.
    """
    
    # Strict patterns for ground truth (definitive job pages)
    # Supports both path patterns (/job/) and suffix patterns (-jobs)
    GROUND_TRUTH_JOB_PATTERNS = [
    '/job/', '/job-search/', '/career/', '/careers/', '/vacancy/',
    '/vacancies/', '/position/', '/positions/', '/hiring/',
    '/employment/', '/recruitment/',
    'job-detail', 'job-listings', 'job-listing', 'job_openings',
    '-jobs-in-', '-jobs/'
    ]
    # AI-related keywords for category classification (more inclusive)
    AI_KEYWORDS = [
        'ai', 'artificial-intelligence', 'machine-learning',
        'machine_learning', 'deep-learning', 'data-scientist',
        'data scientist', 'data analyst', 'data engineer', 'ml-engineer', 
        'nlp', 'computer-vision', 'chatgpt', 'llm', 'neural',
        'software', 'engineer', 'developer', 'programmer', 'technical'
    ]
    
    # Cloud-related keywords for category classification
    CLOUD_KEYWORDS = [
        'cloud', 'aws', 'azure', 'gcp', 'devops',
        'kubernetes', 'k8s', 'docker', 'terraform', 'sre',
        'serverless', 'cloudformation'
    ]

    # Error patterns for false positive classification
    ERROR_PATTERNS = {
        'listing_page': [
            '/jobs?', '/search', '/browse', '/results',
            'job?search', 'jobs?', 'in-jobs'
        ],
        'keyword_noise': [
            '/blog', '/news', '/article', '/post',
            '/about', '/contact', '/faq'
        ],
        'navigation': [
            '/user', '/login', '/signin', '/register',
            '/profile', '/settings', '/account'
        ]
    }

    # HTML signals for listing pages (NOT individual job pages)
    LISTING_SIGNALS = [
        'search results', 'browse jobs', 'all jobs',
        'jobs found', 'showing', 'page', 'of', 'results'
    ]

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        self.results: List[PageResult] = []
        self.ground_truth_urls: Set[str] = set()
        self.visited_urls: Set[str] = set()
        
    def add_result(
        self,
        url: str,
        crawler_predicted_job: bool,
        crawler_predicted_relevant: bool,
        crawler_relevance_score: float,
        was_blocked: bool,
        depth: int,
        html_content: str = None,
        job_title: str = "",
        company_name: str = ""
    ):
        """
        Add a crawl result with BOTH prediction and ground truth.
        """
        domain = urlparse(url).netloc

        # Compute GROUND TRUTH: Is this actually a job page?
        ground_truth_is_job = self._is_ground_truth_job(url, html_content)

        # Compute URL CATEGORY: What type of job is this (AI, Cloud, Other)?
        url_category = self._categorize_url(url)

        # Classify error type for false positives
        error_type = ""
        is_fp = False
        if crawler_predicted_job and not ground_truth_is_job:
            is_fp = True
            error_type = self._classify_error(url, html_content)

        result = PageResult(
            url=url,
            domain=domain,
            depth=depth,
            job_title=job_title or "",
            company_name=company_name or "",
            crawler_predicted_job=crawler_predicted_job,
            crawler_predicted_relevant=crawler_predicted_relevant,
            crawler_relevance_score=crawler_relevance_score,
            ground_truth_is_job=ground_truth_is_job,
            url_category=url_category,
            is_false_positive=is_fp,
            error_type=error_type,
            was_blocked=was_blocked,
            html_content=html_content
        )

        self.results.append(result)
        self.visited_urls.add(url)

        if ground_truth_is_job:
            self.ground_truth_urls.add(url)

    def _classify_error(self, url: str, html: str = None) -> str:
        """Classify the type of false positive error."""
        url_lower = url.lower()

        # Check URL patterns
        for pattern in self.ERROR_PATTERNS.get('listing_page', []):
            if pattern in url_lower:
                return 'listing_page'

        for pattern in self.ERROR_PATTERNS.get('keyword_noise', []):
            if pattern in url_lower:
                return 'keyword_noise'

        for pattern in self.ERROR_PATTERNS.get('navigation', []):
            if pattern in url_lower:
                return 'navigation'

        # Check HTML content for listing signals
        if html:
            html_lower = html.lower()
            listing_count = sum(1 for sig in self.LISTING_SIGNALS if sig in html_lower)
            if listing_count >= 2:
                return 'listing_page'

        return 'keyword_noise'
    
    def _is_ground_truth_job(self, url: str, html: str = None) -> bool:
        """
        Determine if a URL is ACTUALLY a job page (ground truth).
        Uses STRICT matching - only definitive job page patterns.
        """
        url_lower = url.lower()
        
        # Check for definitive job page patterns in URL
        for pattern in self.GROUND_TRUTH_JOB_PATTERNS:
            if pattern in url_lower:
                return True
        
        # Additional checks from HTML content if available
        if html:
            html_lower = html.lower()
            job_indicators = ['job title', 'job description', 'apply now', 
                            'job requirements', 'salary range']
            indicator_count = sum(1 for ind in job_indicators if ind in html_lower)
            if indicator_count >= 2:
                return True
        
        return False
    
    def _categorize_url(self, url: str) -> str:
        """
        Categorize URL by tech category (AI, Cloud, Other).
        Based on URL keywords, NOT on whether it's a job page.
        """
        url_lower = url.lower()
        
        if any(kw in url_lower for kw in self.AI_KEYWORDS):
            return 'ai'
        if any(kw in url_lower for kw in self.CLOUD_KEYWORDS):
            return 'cloud'
        return 'other'
    
    def compute_confusion_matrix(self) -> ConfusionMatrix:
        """
        Compute confusion matrix for overall job page detection.
        """
        cm = ConfusionMatrix()
        
        for result in self.results:
            predicted_job = result.crawler_predicted_job
            actual_job = result.ground_truth_is_job
            
            if predicted_job and actual_job:
                cm.true_positives += 1
            elif predicted_job and not actual_job:
                cm.false_positives += 1
            elif not predicted_job and actual_job:
                cm.false_negatives += 1
            else:
                cm.true_negatives += 1
        
        return cm
    
    def compute_metrics(self) -> EvaluationMetrics:
        """
        Compute ALL metrics with proper TP/FP/FN/TN.
        """
        cm = self.compute_confusion_matrix()
        
        total = len(self.results)
        blocked = sum(1 for r in self.results if r.was_blocked)
        
        # Include ALL pages in metrics (including blocked ones with job URLs)
        evaluable = [r for r in self.results]
        evaluable_count = len(evaluable) if evaluable else 1
        
        # Compute confusion for ALL pages
        tp = sum(1 for r in evaluable if r.crawler_predicted_job and r.ground_truth_is_job)
        fp = sum(1 for r in evaluable if r.crawler_predicted_job and not r.ground_truth_is_job)
        fn = sum(1 for r in evaluable if not r.crawler_predicted_job and r.ground_truth_is_job)
        tn = sum(1 for r in evaluable if not r.crawler_predicted_job and not r.ground_truth_is_job)
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1 = 2 * (P * R) / (P + R)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy = (TP + TN) / Total
        accuracy = (tp + tn) / evaluable_count if evaluable_count > 0 else 0.0
        
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Efficiency
        relevant = sum(1 for r in self.results if r.crawler_predicted_relevant)
        efficiency = relevant / total if total > 0 else 0.0
        
        # Miss rate
        exploration_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
        
        job_pages_gt = tp + fn
        job_pages_predicted = tp + fp
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            specificity=specificity,
            total_pages=total,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            job_pages_in_ground_truth=job_pages_gt,
            job_pages_found_by_crawler=job_pages_predicted,
            relevant_pages_found=relevant,
            blocked_pages=blocked,
            efficiency=efficiency,
            exploration_rate=exploration_rate
        )
    
    def compute_category_metrics(self) -> Dict[str, dict]:
        """
        Compute metrics SPLIT BY CATEGORY (AI vs Cloud vs Other).
        
        For EACH category, we compute:
        - How many pages in that category are actually job pages
        - How well did the crawler identify job pages in that category
        
        Example:
        - AI category: pages with AI keywords in URL
          - Of these, how many are job pages? (TP/FN)
          - Of non-job pages, how many did crawler wrongly call jobs? (FP)
        """
        categories = defaultdict(lambda: {
            'total': 0, 
            'is_job': 0,  # Actually job pages
            'not_job': 0,  # Not job pages
            'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0
        })
        
        for result in self.results:
            if result.was_blocked:
                continue
            
            cat = result.url_category  # AI, Cloud, or Other
            categories[cat]['total'] += 1
            
            if result.ground_truth_is_job:
                categories[cat]['is_job'] += 1
            else:
                categories[cat]['not_job'] += 1
            
            # Standard confusion matrix per category
            if result.crawler_predicted_job and result.ground_truth_is_job:
                categories[cat]['tp'] += 1
            elif result.crawler_predicted_job and not result.ground_truth_is_job:
                categories[cat]['fp'] += 1
            elif not result.crawler_predicted_job and result.ground_truth_is_job:
                categories[cat]['fn'] += 1
            else:
                categories[cat]['tn'] += 1
        
        # Compute metrics per category
        results = {}
        for cat in ['ai', 'cloud', 'other']:
            if cat not in categories:
                continue
                
            c = categories[cat]
            tp, fp, fn, tn = c['tp'], c['fp'], c['fn'], c['tn']
            is_job = c['is_job']
            
            # Precision: Of predicted jobs, how many are real?
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # Recall: Of actual jobs in this category, how many found?
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # F1
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            # Job density: what % of pages in this category are job pages?
            job_density = is_job / c['total'] if c['total'] > 0 else 0.0
            
            results[cat] = {
                'total': c['total'],
                'actual_jobs': is_job,
                'not_jobs': c['not_job'],
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'job_density': job_density
            }
        
        return results

    def analyze_false_positives(self) -> Dict:
        """Analyze false positive patterns."""
        fps = [r for r in self.results if r.is_false_positive]

        error_counts = defaultdict(int)
        for fp in fps:
            error_counts[fp.error_type] += 1

        return {
            'total_false_positives': len(fps),
            'by_type': dict(error_counts),
            'examples': [(r.url, r.error_type) for r in fps[:5]]
        }

    def analyze_by_domain(self) -> Dict:
        """Analyze performance by domain."""
        domain_stats = defaultdict(lambda: {
            'total': 0, 'job_pages': 0, 'predicted_jobs': 0,
            'true_positives': 0, 'false_positives': 0
        })

        for r in self.results:
            if r.was_blocked:
                continue
            d = domain_stats[r.domain]
            d['total'] += 1

            if r.ground_truth_is_job:
                d['job_pages'] += 1
            if r.crawler_predicted_job:
                d['predicted_jobs'] += 1
            if r.crawler_predicted_job and r.ground_truth_is_job:
                d['true_positives'] += 1
            if r.crawler_predicted_job and not r.ground_truth_is_job:
                d['false_positives'] += 1

        # Calculate precision per domain
        for domain, stats in domain_stats.items():
            if stats['predicted_jobs'] > 0:
                stats['precision'] = stats['true_positives'] / stats['predicted_jobs']
            else:
                stats['precision'] = 0.0
            if stats['job_pages'] > 0:
                stats['recall'] = stats['true_positives'] / stats['job_pages']
            else:
                stats['recall'] = 0.0

        return dict(domain_stats)

    def print_summary(self):
        """Print comprehensive summary with REAL metrics."""
        metrics = self.compute_metrics()
        cm = self.compute_confusion_matrix()
        cat_metrics = self.compute_category_metrics()
        fp_analysis = self.analyze_false_positives()
        domain_analysis = self.analyze_by_domain()

        print("\n" + "="*70)
        print("CRAWLER EVALUATION - REALISTIC METRICS")
        print("="*70)

        print("\nOVERALL CONFUSION MATRIX:")
        print("-"*45)
        print("                    Predicted")
        print("                 Job    Not Job")
        print(f"Actual Job      {cm.true_positives:4d}     {cm.false_negatives:4d}")
        print(f"       Not Job  {cm.false_positives:4d}     {cm.true_negatives:4d}")

        print("\n" + "-"*45)
        print("OVERALL COUNTS:")
        print(f"  Total pages visited:       {metrics.total_pages}")
        print(f"  Blocked pages:              {metrics.blocked_pages}")
        print(f"  Evaluable pages:            {metrics.total_pages - metrics.blocked_pages}")
        print(f"  Actual job pages (GT):      {metrics.job_pages_in_ground_truth}")
        print(f"  Predicted as job:          {metrics.job_pages_found_by_crawler}")

        print("\n" + "-"*45)
        print("OVERALL PERFORMANCE:")
        print(f"  Precision:  {metrics.precision:.4f}  (predicted jobs that are real)")
        print(f"  Recall:     {metrics.recall:.4f}  (real jobs that were found)")
        print(f"  F1-Score:  {metrics.f1_score:.4f}")
        print(f"  Accuracy:  {metrics.accuracy:.4f}")

        # Precision-Recall Trade-off Interpretation
        print("\n" + "-"*45)
        print("PRECISION-RECALL TRADE-OFF:")
        if metrics.precision > metrics.recall:
            print(f"  High Precision ({metrics.precision:.2f}), Lower Recall ({metrics.recall:.2f})")
            print("  -> Conservative: Few false positives, but may miss some jobs")
        else:
            print(f"  Lower Precision ({metrics.precision:.2f}), High Recall ({metrics.recall:.2f})")
            print("  -> Aggressive: Finds more jobs, but more false positives")

        # False Positive Analysis
        print("\n" + "-"*45)
        print("FALSE POSITIVE ANALYSIS:")
        print(f"  Total False Positives: {fp_analysis['total_false_positives']}")
        for error_type, count in fp_analysis['by_type'].items():
            print(f"    {error_type}: {count}")
        if fp_analysis['examples']:
            print("  Examples:")
            for url, err_type in fp_analysis['examples'][:3]:
                print(f"    [{err_type}] {url[:50]}...")

        # Domain Analysis
        print("\n" + "-"*45)
        print("DOMAIN ANALYSIS:")
        for domain, stats in sorted(domain_analysis.items(),
                                    key=lambda x: x[1]['total'], reverse=True):
            print(f"  {domain}:")
            print(f"    Pages: {stats['total']}, Jobs: {stats['job_pages']}, "
                  f"Precision: {stats['precision']:.2f}")
        
        print("\n" + "-"*45)
        print("PERFORMANCE BY CATEGORY (based on URL keywords):")
        print("-"*45)
        
        # AI category
        if 'ai' in cat_metrics:
            m = cat_metrics['ai']
            print(f"\n  AI JOBS (URLs with: ML, data scientist, AI, etc.):")
            print(f"    Total pages:      {m['total']}")
            print(f"    Actual jobs:      {m['actual_jobs']} ({m['job_density']:.1%} job density)")
            print(f"    Predicted jobs:   {m['tp'] + m['fp']}")
            print(f"    TP={m['tp']}, FP={m['fp']}, FN={m['fn']}")
            print(f"    Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}, F1: {m['f1_score']:.4f}")
        
        # Cloud category
        if 'cloud' in cat_metrics:
            m = cat_metrics['cloud']
            print(f"\n  CLOUD JOBS (URLs with: AWS, Kubernetes, Docker, etc.):")
            print(f"    Total pages:      {m['total']}")
            print(f"    Actual jobs:      {m['actual_jobs']} ({m['job_density']:.1%} job density)")
            print(f"    Predicted jobs:   {m['tp'] + m['fp']}")
            print(f"    TP={m['tp']}, FP={m['fp']}, FN={m['fn']}")
            print(f"    Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}, F1: {m['f1_score']:.4f}")
        
        # Other category
        if 'other' in cat_metrics:
            m = cat_metrics['other']
            print(f"\n  OTHER (URLs with no AI/Cloud keywords):")
            print(f"    Total pages:      {m['total']}")
            print(f"    Actual jobs:      {m['actual_jobs']} ({m['job_density']:.1%} job density)")
            print(f"    TP={m['tp']}, FP={m['fp']}, FN={m['fn']}")
            print(f"    Precision: {m['precision']:.4f}, Recall: {m['recall']:.4f}, F1: {m['f1_score']:.4f}")
        
        print("\n" + "="*70)
        
        # Show some examples
        print("\nSAMPLE URLs BY CATEGORY:")
        for cat in ['ai', 'cloud', 'other']:
            cat_results = [r for r in self.results[:20] if r.url_category == cat][:3]
            if cat_results:
                print(f"\n  {cat.upper()} examples:")
                for r in cat_results:
                    status = "JOB" if r.ground_truth_is_job else "---"
                    pred = "JOB" if r.crawler_predicted_job else "---"
                    print(f"    [{status}/{pred}] {r.url[:55]}...")
    
    def save_to_csv(self, filename: str = None):
        """Save results with both prediction and ground truth."""
        if filename is None:
            filename = "visited_urls.csv"
        # Check if filename contains path separator (already has directory)
        if os.sep in filename or filename.startswith('.'):
            filepath = filename
        else:
            filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'URL', 'Domain', 'Title', 'Company',
                'Is_Job_Page', 'Relevance_Score', 'Depth',
                'Ground_Truth_Is_Job', 'Correct', 'Error_Type',
                'URL_Category', 'Was_Blocked'
            ])

            for result in self.results:
                correct = (result.crawler_predicted_job == result.ground_truth_is_job)
                writer.writerow([
                    result.url,
                    result.domain,
                    result.job_title[:100] if result.job_title else "",
                    result.company_name[:50] if result.company_name else "",
                    result.crawler_predicted_job,
                    f"{result.crawler_relevance_score:.4f}",
                    result.depth,
                    result.ground_truth_is_job,
                    correct,
                    result.error_type if result.error_type else "none",
                    result.url_category,
                    result.was_blocked
                ])

        logger.info(f"Saved {len(self.results)} results to {filepath}")
    
    def save_ground_truth(self, filename: str = None):
        """Save ground truth URLs by category."""
        if filename is None:
            filename = "ground_truth.csv"
        if os.sep in filename or filename.startswith('.'):
            filepath = filename
        else:
            filepath = os.path.join(self.output_dir, filename)
        
        # Group by category
        ai_jobs = []
        cloud_jobs = []
        other_jobs = []
        
        for url in self.ground_truth_urls:
            cat = self._categorize_url(url)
            if cat == 'ai':
                ai_jobs.append(url)
            elif cat == 'cloud':
                cloud_jobs.append(url)
            else:
                other_jobs.append(url)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['URL', 'Domain', 'Category', 'Is_Job_Page', 'Was_Found'])
            
            for url in sorted(ai_jobs):
                writer.writerow([url, urlparse(url).netloc, 'ai', True, url in self.visited_urls])
            for url in sorted(cloud_jobs):
                writer.writerow([url, urlparse(url).netloc, 'cloud', True, url in self.visited_urls])
            for url in sorted(other_jobs):
                writer.writerow([url, urlparse(url).netloc, 'other', True, url in self.visited_urls])
        
        logger.info(f"Saved {len(self.ground_truth_urls)} ground truth URLs")
    
    def save_evaluation_report(self, filename: str = None,
                               baseline_metrics: EvaluationMetrics = None):
        """Save comprehensive evaluation report."""
        if filename is None:
            filename = "evaluation_results.txt"
        if os.sep in filename or filename.startswith('.'):
            filepath = filename
        else:
            filepath = os.path.join(self.output_dir, filename)
        metrics = self.compute_metrics()
        cat_metrics = self.compute_category_metrics()
        cm = self.compute_confusion_matrix()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("INTELLIGENT CRAWLER - EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Confusion Matrix
            f.write("OVERALL CONFUSION MATRIX\n")
            f.write("-"*50 + "\n")
            f.write("                    Predicted\n")
            f.write("                 Job    Not Job\n")
            f.write(f"Actual Job      {cm.true_positives:4d}     {cm.false_negatives:4d}\n")
            f.write(f"       Not Job  {cm.false_positives:4d}     {cm.true_negatives:4d}\n\n")
            
            # Overall metrics
            f.write("OVERALL PERFORMANCE\n")
            f.write("-"*50 + "\n")
            f.write(f"Precision:     {metrics.precision:.4f}\n")
            f.write(f"Recall:        {metrics.recall:.4f}\n")
            f.write(f"F1-Score:      {metrics.f1_score:.4f}\n")
            f.write(f"Accuracy:      {metrics.accuracy:.4f}\n\n")
            
            # Category breakdown
            f.write("PERFORMANCE BY CATEGORY\n")
            f.write("-"*50 + "\n")
            
            for cat in ['ai', 'cloud', 'other']:
                if cat not in cat_metrics:
                    continue
                m = cat_metrics[cat]
                f.write(f"\n{cat.upper()} JOBS:\n")
                f.write(f"  Total pages:      {m['total']}\n")
                f.write(f"  Actual jobs:      {m['actual_jobs']}\n")
                f.write(f"  Job density:      {m['job_density']:.1%}\n")
                f.write(f"  TP/FP/FN/TN:      {m['tp']}/{m['fp']}/{m['fn']}/{m['tn']}\n")
                f.write(f"  Precision:        {m['precision']:.4f}\n")
                f.write(f"  Recall:           {m['recall']:.4f}\n")
                f.write(f"  F1-Score:        {m['f1_score']:.4f}\n")
            
            # Comparison
            if baseline_metrics:
                f.write("\n" + "="*70 + "\n")
                f.write("COMPARISON WITH BASELINE\n")
                f.write("="*70 + "\n\n")
                f.write(f"{'Metric':<20} {'Intelligent':<15} {'Baseline':<15}\n")
                f.write("-"*50 + "\n")
                f.write(f"{'Precision':<20} {metrics.precision:.4f}{'':<10} {baseline_metrics.precision:.4f}\n")
                f.write(f"{'Recall':<20} {metrics.recall:.4f}{'':<10} {baseline_metrics.recall:.4f}\n")
                f.write(f"{'F1-Score':<20} {metrics.f1_score:.4f}{'':<10} {baseline_metrics.f1_score:.4f}\n")
                f.write(f"{'Accuracy':<20} {metrics.accuracy:.4f}{'':<10} {baseline_metrics.accuracy:.4f}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        logger.info(f"Saved report to {filepath}")
    
    def get_confusion_matrix_text(self) -> str:
        """Get confusion matrix as text."""
        cm = self.compute_confusion_matrix()
        return f"TP={cm.true_positives}, FP={cm.false_positives}, FN={cm.false_negatives}, TN={cm.true_negatives}"
    
    def reset(self):
        """Reset for new crawl."""
        self.results.clear()
        self.ground_truth_urls.clear()
        self.visited_urls.clear()
