"""
Main entry point for the Intelligent Web Crawling System.

This system implements an intelligent crawler with Q-learning and Multi-Armed Bandits
for adaptive job market analysis.

Usage:
    python main.py                         # Run intelligent crawler (default)
    python main.py --mode baseline        # Run baseline FIFO crawler
    python main.py --mode compare         # Run both and compare
    python main.py --keywords ai          # Use AI-specific keywords
    python main.py --keywords cloud       # Use Cloud/DevOps keywords

Requirements:
    - Seeds must be in seeds.txt (one URL per line)
    - Output: output/visited_urls.csv, output/evaluation_results.txt
"""
import argparse
import logging
import os
import sys
from typing import List, Dict

OUTPUT_DIR = "output"

# Predefined keyword sets for different job categories
KEYWORD_SETS: Dict[str, List[str]] = {
    "general": [
        "python", "java", "javascript", "typescript",
        "machine learning", "deep learning", "ai", "artificial intelligence",
        "aws", "azure", "gcp", "cloud", "docker", "kubernetes", "k8s",
        "sql", "database", "postgresql", "mongodb", "mysql",
        "react", "angular", "vue", "frontend", "full stack",
        "backend", "api", "rest", "graphql",
        "git", "agile", "scrum", "devops", "sre",
        "software engineer", "developer", "programmer",
        "data scientist", "data engineer", "ml engineer",
        "job", "jobs", "career", "hiring", "vacancy"
    ],
    "ai": [
        "machine learning", "deep learning", "artificial intelligence", "ai",
        "ml", "nlp", "natural language", "computer vision",
        "neural network", "tensorflow", "pytorch", "keras",
        "data scientist", "data analyst", "ml engineer",
        "python", "r", "statistics", "analytics",
        "chatgpt", "llm", "gpt", "transformer", "bert"
    ],
    "cloud": [
        "aws", "azure", "gcp", "google cloud", "cloud",
        "devops", "sre", "site reliability",
        "docker", "kubernetes", "k8s", "container",
        "terraform", "ansible", "cloudformation",
        "ci/cd", "jenkins", "gitlab", "github actions",
        "serverless", "lambda", "microservices"
    ],
    "web": [
        "javascript", "typescript", "react", "vue", "angular",
        "node", "nodejs", "express", "django", "flask",
        "html", "css", "sass", "less", "bootstrap",
        "frontend", "full stack", "web developer",
        "api", "rest", "graphql", "http"
    ]
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory: {OUTPUT_DIR}")


def load_seeds(filepath: str = "seeds.txt") -> List[str]:
    """Load seed URLs from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            seeds = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(seeds)} seed URLs from {filepath}")
        return seeds
    except FileNotFoundError:
        logger.error(f"Seeds file not found: {filepath}")
        return []


def run_intelligent_crawler(args):
    """Run the intelligent priority-based crawler."""
    try:
        from crawler.intelligent_crawler import IntelligentCrawler

        logger.info("=" * 70)
        logger.info("INTELLIGENT CRAWLER MODE")
        logger.info("=" * 70)

        # Get keyword set
        keyword_set_name = getattr(args, 'keywords', 'general') or 'general'
        keywords = KEYWORD_SETS.get(keyword_set_name, KEYWORD_SETS["general"])
        logger.info(f"Using keyword set: {keyword_set_name}")
        logger.info(f"Keywords: {', '.join(keywords[:10])}...")

        seeds = load_seeds(args.seeds)
        if not seeds:
            logger.error("No seeds available. Exiting.")
            return None, None, None

        crawler = IntelligentCrawler(
            skills=keywords,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            output_dir=args.output_dir,
            keyword_set=keyword_set_name,
            epsilon=0.30
        )
        
        crawler.initialize(seeds)
        results = crawler.crawl()
        crawler.print_summary()
        
        if args.save:
            crawler.save_results()
            logger.info(f"Results saved to {args.output_dir}/")
        
        return crawler, results, crawler.evaluator
        
    except Exception as e:
        logger.error(f"Intelligent crawler failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def run_baseline_crawler(args):
    """Run the baseline FIFO crawler."""
    try:
        from crawler.baseline import BaselineCrawler
        from crawler.evaluator import Evaluator
        
        logger.info("=" * 70)
        logger.info("BASELINE CRAWLER (FIFO) MODE")
        logger.info("=" * 70)
        
        seeds = load_seeds(args.seeds)
        if not seeds:
            logger.error("No seeds available. Exiting.")
            return None, None, None
        
        crawler = BaselineCrawler(
            max_pages=args.max_pages,
            max_depth=args.max_depth
        )
        crawler.initialize(seeds)
        
        evaluator = Evaluator(output_dir=args.output_dir)
        results = crawler.crawl()
        
        for url, is_job_predicted, was_blocked, depth in results:
            evaluator.add_result(
                url=url,
                crawler_predicted_job=is_job_predicted,
                crawler_predicted_relevant=False,
                crawler_relevance_score=0.0,
                was_blocked=was_blocked,
                depth=depth,
                html_content=None
            )
        
        crawler.print_summary()
        evaluator.print_summary()
        
        return crawler, results, evaluator
        
    except Exception as e:
        logger.error(f"Baseline crawler failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def run_comparison(args):
    """Run both crawlers and compare results."""
    from crawler.evaluator import EvaluationMetrics
    
    logger.info("=" * 70)
    logger.info("COMPARISON MODE: Running both crawlers")
    logger.info("=" * 70)
    
    print("\n" + "#" * 70)
    print("# BASELINE CRAWLER (FIFO)")
    print("#" * 70)
    _, _, baseline_evaluator = run_baseline_crawler(args)
    
    if baseline_evaluator is None:
        logger.error("Baseline crawler failed, skipping comparison")
        return
    
    baseline_metrics = baseline_evaluator.compute_metrics()
    
    print("\n" + "#" * 70)
    print("# INTELLIGENT CRAWLER")
    print("#" * 70)
    intelligent_crawler, _, intelligent_evaluator = run_intelligent_crawler(args)
    
    if intelligent_crawler is None:
        logger.error("Intelligent crawler failed, cannot complete comparison")
        return
    
    int_metrics = intelligent_evaluator.compute_metrics()
    
    if args.save:
        intelligent_evaluator.save_evaluation_report(
            os.path.join(args.output_dir, "evaluation_results.txt"),
            baseline_metrics=baseline_metrics
        )
        intelligent_crawler.save_results()
    
    print_comparison_table(int_metrics, baseline_metrics, intelligent_evaluator, baseline_evaluator)


def print_comparison_table(int_metrics, baseline_metrics, int_evaluator, base_evaluator):
    """Print side-by-side comparison table."""
    def calc_improvement(smart, base):
        if base == 0:
            return "N/A"
        pct = ((smart - base) / base) * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"
    
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON TABLE")
    print("=" * 70)
    
    print(f"\n{'Metric':<20} {'Intelligent':<15} {'Baseline':<15} {'Improvement':<15}")
    print("-" * 65)
    
    print(f"{'--- Overall ---':^62}")
    print(f"{'Precision':<20} {int_metrics.precision:>12.4f}   {baseline_metrics.precision:>12.4f}   {calc_improvement(int_metrics.precision, baseline_metrics.precision):>12}")
    print(f"{'Recall':<20} {int_metrics.recall:>12.4f}   {baseline_metrics.recall:>12.4f}   {calc_improvement(int_metrics.recall, baseline_metrics.recall):>12}")
    print(f"{'F1-Score':<20} {int_metrics.f1_score:>12.4f}   {baseline_metrics.f1_score:>12.4f}   {calc_improvement(int_metrics.f1_score, baseline_metrics.f1_score):>12}")
    print(f"{'Accuracy':<20} {int_metrics.accuracy:>12.4f}   {baseline_metrics.accuracy:>12.4f}   {calc_improvement(int_metrics.accuracy, baseline_metrics.accuracy):>12}")
    
    print(f"{'--- Counts ---':^62}")
    print(f"{'Total Pages':<20} {int_metrics.total_pages:>12d}   {baseline_metrics.total_pages:>12d}   {'---':>12}")
    print(f"{'Actual Jobs (GT)':<20} {int_metrics.job_pages_in_ground_truth:>12d}   {baseline_metrics.job_pages_in_ground_truth:>12d}   {calc_improvement(int_metrics.job_pages_in_ground_truth, baseline_metrics.job_pages_in_ground_truth):>12}")
    print(f"{'Predicted Jobs':<20} {int_metrics.job_pages_found_by_crawler:>12d}   {baseline_metrics.job_pages_found_by_crawler:>12d}   {calc_improvement(int_metrics.job_pages_found_by_crawler, baseline_metrics.job_pages_found_by_crawler):>12}")
    
    int_cat = int_evaluator.compute_category_metrics()
    base_cat = base_evaluator.compute_category_metrics()
    
    for cat in ['ai', 'cloud']:
        if cat in int_cat or cat in base_cat:
            print(f"{'--- ' + cat.upper() + ' Category ---':^62}")
            i_p = int_cat.get(cat, {}).get('precision', 0)
            b_p = base_cat.get(cat, {}).get('precision', 0)
            i_r = int_cat.get(cat, {}).get('recall', 0)
            b_r = base_cat.get(cat, {}).get('recall', 0)
            i_f = int_cat.get(cat, {}).get('f1_score', 0)
            b_f = base_cat.get(cat, {}).get('f1_score', 0)
            print(f"{'Precision':<20} {i_p:>12.4f}   {b_p:>12.4f}   {calc_improvement(i_p, b_p):>12}")
            print(f"{'Recall':<20} {i_r:>12.4f}   {b_r:>12.4f}   {calc_improvement(i_r, b_r):>12}")
            print(f"{'F1-Score':<20} {i_f:>12.4f}   {b_f:>12.4f}   {calc_improvement(i_f, b_f):>12}")
    
    print("\n" + "-" * 65)
    print("CONFUSION MATRICES")
    print("-" * 65)
    cm_int = int_evaluator.compute_confusion_matrix()
    cm_base = base_evaluator.compute_confusion_matrix()
    
    print(f"{'Metric':<15} {'Intelligent':>15} {'Baseline':>15}")
    print("-" * 45)
    print(f"{'TP':<15} {cm_int.true_positives:>15d} {cm_base.true_positives:>15d}")
    print(f"{'FP':<15} {cm_int.false_positives:>15d} {cm_base.false_positives:>15d}")
    print(f"{'FN':<15} {cm_int.false_negatives:>15d} {cm_base.false_negatives:>15d}")
    print(f"{'TN':<15} {cm_int.true_negatives:>15d} {cm_base.true_negatives:>15d}")
    
    print("\n" + "=" * 70)


def prompt_for_keywords() -> str:
    """Interactive prompt for keyword selection."""
    print("\n" + "="*50)
    print("SELECT KEYWORD DOMAIN")
    print("="*50)
    print("Available options:")
    print("  1. ai         - AI/ML/Data Science jobs")
    print("  2. cloud      - Cloud/DevOps/SRE jobs")
    print("  3. web        - Web Development jobs")
    print("  4. general    - General programming jobs")
    print("="*50)

    while True:
        choice = input("\nEnter keyword domain (1-4) or name: ").strip().lower()

        if choice in ['1', 'ai']:
            return 'ai'
        elif choice in ['2', 'cloud']:
            return 'cloud'
        elif choice in ['3', 'web']:
            return 'web'
        elif choice in ['4', 'general']:
            return 'general'
        else:
            print("Invalid choice. Please enter 1-4 or ai/cloud/web/general")


def prompt_for_max_pages() -> int:
    """Interactive prompt for max pages."""
    while True:
        try:
            pages = input("\nEnter max pages to crawl (default: 20): ").strip()
            if not pages:
                return 20
            pages = int(pages)
            if pages > 0:
                return pages
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")


def prompt_for_mode() -> str:
    """Interactive prompt for crawler mode."""
    print("\n" + "="*50)
    print("SELECT CRAWLER MODE")
    print("="*50)
    print("  1. intelligent  - Q-learning based (recommended)")
    print("  2. baseline     - Simple FIFO crawler")
    print("  3. compare      - Run both and compare")
    print("="*50)

    while True:
        choice = input("\nEnter mode (1-3, default: 1): ").strip().lower()

        if choice in ['1', 'intelligent', '']:
            return 'intelligent'
        elif choice in ['2', 'baseline']:
            return 'baseline'
        elif choice in ['3', 'compare']:
            return 'compare'
        else:
            print("Invalid choice. Please enter 1-3.")


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent Web Crawler for Job Market Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          Run intelligent crawler (default)
  python main.py --mode baseline          Run baseline FIFO crawler
  python main.py --mode compare           Run both and compare
  python main.py --max-pages 50           Set max pages to 50

Output:
  Results saved to ./output/ directory:
  - visited_urls.csv     : All crawled URLs
  - evaluation_results.txt : Detailed metrics report
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['intelligent', 'baseline', 'compare'],
        default='intelligent',
        help='Crawler mode (default: intelligent)'
    )
    
    parser.add_argument(
        '--seeds',
        default='seeds.txt',
        help='Path to seeds file (default: seeds.txt)'
    )
    
    parser.add_argument(
        '--max-pages',
        type=int,
        default=20,
        help='Maximum pages to crawl (default: 20)'
    )
    
    parser.add_argument(
        '--max-depth',
        type=int,
        default=10,
        help='Maximum crawl depth (default: 10)'
    )
    
    parser.add_argument(
        '--output-dir',
        default=OUTPUT_DIR,
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help="Don't save results to files"
    )
    
    parser.add_argument(
        '--keywords',
        choices=['general', 'ai', 'cloud', 'web'],
        default='general',
        help='Keyword set for relevance scoring (default: general)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive mode - prompt for input'
    )

    args = parser.parse_args()

    # Interactive mode - prompt for inputs (only with -i flag)
    if args.interactive:
        print("\n" + "="*50)
        print("INTELLIGENT JOB CRAWLER - INTERACTIVE MODE")
        print("="*50)
        args.mode = prompt_for_mode()
        args.keywords = prompt_for_keywords()
        args.max_pages = prompt_for_max_pages()
        # Auto-detect seeds path
        import os
        if os.path.exists("seeds.txt"):
            args.seeds = "seeds.txt"
        print("\n" + "="*50)
        print(f"Starting crawl with:")
        print(f"  Mode: {args.mode}")
        print(f"  Keywords: {args.keywords}")
        print(f"  Max pages: {args.max_pages}")
        print("="*50 + "\n")

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    args.save = not args.no_save
    
    if args.save:
        ensure_output_dir()
    
    try:
        if args.mode == 'intelligent':
            run_intelligent_crawler(args)
        elif args.mode == 'baseline':
            run_baseline_crawler(args)
        elif args.mode == 'compare':
            run_comparison(args)
    except KeyboardInterrupt:
        logger.info("Crawl interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Crawl failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
