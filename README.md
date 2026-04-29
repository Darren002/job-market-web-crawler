# Intelligent Web Crawling System for Job Market Analysis

An adaptive web crawler that uses Q-learning and Multi-Armed Bandits to intelligently discover and classify job listings from multiple job sites.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Quick Start](#quick-start)
3. [How to Run](#how-to-run)
4. [Command Options](#command-options)
5. [Output Files](#output-files)
6. [Project Structure](#project-structure)
7. [Core Components](#core-components)
8. [Intelligent Features](#intelligent-features)
9. [Troubleshooting](#troubleshooting)
10. [Requirements](#requirements)

---

## Project Overview

This system crawls job websites, learns which patterns lead to job listings, and prioritizes high-value pages. It includes both an intelligent (AI-powered) crawler and a baseline FIFO crawler for comparison.

### Key Features

- **Q-Learning Agent**: Learns which URLs lead to job pages and prioritizes them
- **Multi-Armed Bandit (UCB1)**: Balances exploration vs exploitation for optimal crawling
- **TF-IDF Content Analysis**: Uses text similarity to avoid duplicate content
- **Realistic Evaluation**: Ground truth vs predictions with precision, recall, F1 metrics
- **Blocking Detection**: Automatic fallback to Selenium when blocked
- **Interactive Visualization**: Generates an animated HTML map of the crawl

### System Architecture

```
Seed URLs -> Q-Learning Agent -> Priority Queue -> Fetch URL
                    |                               |
                    |                               v
                    |                        Extract Links
                    v                               |
              Learn Q-Values <--------------------+
```

The crawler:
1. Starts with seed URLs (job sites)
2. Agent selects next URL using epsilon-greedy policy
3. Page is fetched; content analyzed
4. Q-values updated based on reward (job found = +1)
5. Discovered links added to priority queue
6. Repeat until max pages reached

---

## Quick Start

### Step 1: Navigate to the Project Directory

```bash
cd comp3071-coursework-u-main
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Intelligent Crawler

```bash
python main.py
```

That's it! The crawler will run and save results to the `output/` directory.

---

## How to Run

### Basic Usage

```bash
# Run intelligent crawler (default)
python main.py

# Run baseline (FIFO) crawler for comparison
python main.py --mode baseline

# Run both and compare them
python main.py --mode compare
```

### Interactive Mode (Recommended for First-Time Use)

```bash
python main.py -i
```

This will prompt you for:
- Mode selection (intelligent, baseline, or compare)
- Keyword domain (general, ai, cloud, or web)
- Maximum pages to crawl

Example session:
```
==================================================
INTELLIGENT JOB CRAWLER - INTERACTIVE MODE
==================================================

==================================================
SELECT CRAWLER MODE
==================================================
  1. intelligent  - Q-learning based (recommended)
  2. baseline     - Simple FIFO crawler
  3. compare      - Run both and compare
==================================================

Enter mode (1-3, default: 1): 1

==================================================
SELECT KEYWORD DOMAIN
==================================================
Available options:
  1. ai         - AI/ML/Data Science jobs
  2. cloud      - Cloud/DevOps/SRE jobs
  3. web        - Web Development jobs
  4. general    - General programming jobs
==================================================

Enter keyword domain (1-4) or name: 4

Enter max pages to crawl (default: 20): 20
```

### Generate Crawl Visualization

After running a crawl, generate an interactive visualization:

```bash
python crawlviz.py
```

This creates `crawl_map.html` with an animated view of the crawl showing:
- Nodes for each crawled page (color-coded by type)
- Links between pages showing crawl paths
- Play/Pause controls for animation
- Filter by batch/domain

---

## Command Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Crawler mode: `intelligent`, `baseline`, or `compare` | `intelligent` |
| `--max-pages` | Maximum pages to crawl | `20` |
| `--max-depth` | Maximum link depth from seed | `10` |
| `--seeds` | Path to seeds file | `seeds.txt` |
| `--keywords` | Keyword set: `general`, `ai`, `cloud`, `web` | `general` |
| `--output-dir` | Output directory | `output` |
| `--no-save` | Skip saving results to files | Saves by default |
| `--verbose` | Enable debug logging | Off |
| `-i`, `--interactive` | Interactive mode with prompts | Off |

### Examples

```bash
# Run intelligent crawler with 50 pages
python main.py --max-pages 50

# Run baseline with 10 pages
python main.py --mode baseline --max-pages 10

# Compare both crawlers with 30 pages
python main.py --mode compare --max-pages 30

# Focus on AI jobs
python main.py --keywords ai --max-pages 30

# Run without saving (faster for testing)
python main.py --no-save

# Enable verbose logging
python main.py --verbose
```

### One-Command Execution (Windows)

```bash
run.bat
```

### One-Command Execution (Mac/Linux)

```bash
./run.sh
```

---

## Output Files

Results are saved to the `./output/` directory:

| File | Description |
|------|-------------|
| `visited_urls.csv` | All crawled URLs with predictions, relevance scores, and ground truth |
| `evaluation_results.txt` | Detailed metrics report with precision, recall, F1, accuracy |
| `ground_truth.csv` | Ground truth job URLs by category (ai, cloud, general) |

### Output CSV Format

The `visited_urls.csv` file contains:
- `URL`: The crawled URL
- `Domain`: Domain name
- `Title`: Page title (if available)
- `Is_Job_Page`: Whether crawler predicted it's a job page (True/False)
- `Relevance_Score`: Relevance score (0.0 to 1.0)
- `Depth`: Crawl depth
- `Ground_Truth_Is_Job`: Ground truth label (True/False)
- `Was_Blocked`: Whether the page was blocked (True/False)
- `URL_Category`: Category (ai, cloud, general, other)

### Example Output

```
======================================================================
BASELINE COMPARISON TABLE
======================================================================

Metric               Intelligent     Baseline        Improvement
-----------------------------------------------------------------
                       --- Overall ---
Precision                  0.8889         0.7143         +24.4%
Recall                     1.0000         1.0000          +0.0%
F1-Score                   0.9412         0.8333         +12.9%
Accuracy                   0.9000         0.7778         +15.7%
                        --- Counts ---
Total Pages                    10             10            ---
Actual Jobs (GT)                8              5         +60.0%

======================================================================
EXPLORATION VS EXPLOITATION ANALYSIS
======================================================================
Total selections:      15
  EXPLORATION steps:      2 (13.3%)
  EXPLOITATION steps:    13 (86.7%)

MAB Method:            UCB
Domain bandit arms:    2
```

---

## Project Structure

```
comp3071-coursework-u/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── seeds.txt            # Seed URLs for crawling
├── README.md           # This file
├── run.bat             # Windows launcher
├── run.sh             # Mac/Linux launcher
├── crawlviz.py        # Crawl visualization generator
├── crawl_map.html     # Generated visualization (output)
├── .gitignore         # Git ignore rules
├── output/            # Results directory (generated)
│   ├── visited_urls.csv
│   ├── evaluation_results.txt
│   └── ground_truth.csv
├── crawler/           # Crawler modules
│   ├── __init__.py
│   ├── baseline.py        # FIFO baseline crawler
│   ├── crawler.py         # Core crawler logic
│   ├── evaluator.py       # Metrics evaluation
│   ├── extractor.py      # Link extraction
│   ├── fetcher.py        # HTTP fetching
│   ├── intelligent_crawler.py  # Main intelligent crawler
│   ├── integration.py    # Integration tests
│   ├── qlearning.py     # Q-learning agent
│   ├── agent/             # Agent subpackage
│   │   ├── __init__.py
│   │   ├── action.py      # Action definitions
│   │   ├── agent.py     # Multi-Armed Bandit agent
│   │   ├── scorer.py    # Page scoring
│   │   └── state.py    # State definitions
│   ├── data/             # Data processing subpackage
│   │   ├── __init__.py
│   │   ├── cleaner.py       # HTML/text cleaning
│   │   ├── deduplicator.py  # Content deduplication
│   │   ├── processor.py   # Content processing
│   │   ├── similarity.py  # TF-IDF similarity
│   │   └── vectorizer.py  # TF-IDF vectorizer
│   ├── env/             # Environment subpackage
│   │   ├── __init__.py
│   │   └── web_graph.py # Web graph management
│   └── resilience/      # Resilience subpackage
│       ├── __init__.py
│       ├── auth_handler.py   # Authentication handling
│       ├── block_detector.py  # Blocking detection
│       ├── browser_fetcher.py  # Selenium fetcher
│       ├── delay.py          # Rate limiting
│       ├── headers.py        # HTTP headers
│       └── retry.py          # Retry logic
└── tests/
    └── test_crawler.py  # Unit tests
```

---

## Core Components

### 1. Q-Learning Agent (`qlearning.py`)

The Q-learning agent learns which URLs lead to job pages:
- **State**: Current page context (domain, keywords, depth)
- **Action**: Select next URL from priority queue
- **Reward**: +1 for job page, -0.2 for blocked, 0 otherwise
- **Update**: `Q(s) = Q(s) + alpha * (reward + gamma * max(Q') - Q(s))`

### 2. Multi-Armed Bandit (`agent/agent.py`)

Balances exploration vs exploitation using UCB1:
- **Domain Bandit**: Selects which domain to explore
- **Keyword Bandit**: Selects which keywords to prioritize
- **UCB Formula**: `exploit + explore = avg_reward + sqrt(2*ln(n)/n_arm)`

### 3. Priority Queue

URLs are prioritized by:
- Learned Q-values from the agent
- Relevance score from TF-IDF
- Depth (closer to seed = higher priority)

### 4. Content Analysis (`data/`)

- **TF-IDF Vectorizer**: Converts page content to vectors
- **Similarity**: Detects duplicate/near-duplicate content
- **Deduplicator**: Avoids crawling duplicate pages

### 5. Resilience (`resilience/`)

- **Blocking Detection**: Detects when blocked (403, CAPTCHA, etc.)
- **Browser Fallback**: Uses Selenium/Chrome when blocked
- **Retry Logic**: Retries failed requests
- **Rate Limiting**: Respects server constraints

### 6. Baseline Crawler (`baseline.py`)

Simple FIFO (First-In-First-Out) crawler for comparison:
- No learning or prioritization
- Crawls pages in order discovered
- No relevance scoring

---

## Intelligent Features

### Q-Learning

The agent learns from crawl results:
- Updates Q-values after each page
- Rewards: Job found = +1, Blocked = -0.2, Non-job = 0
- Uses TD-error: `Q(s) = Q(s) + alpha * (reward + gamma * max(Q') - Q(s))`

### Multi-Armed Bandit (UCB1)

Balances exploration vs exploitation:
- **Exploitation**: Choose URLs with highest known reward
- **Exploration**: Try unknown URLs to learn
- **UCB1**: Adds confidence bound to handle uncertainty
- Domains and keywords are treated as separate "arms"

### Epsilon-Greedy Policy

- **Epsilon (30%)**: Random exploration
- **1 - Epsilon (70%)**: Exploitation of learned Q-values

### Ground Truth Classification

Pages are classified as job pages based on URL patterns:
- Contains: job, jobs, career, hiring, vacancy, etc.
- Categories: ai, cloud, general, web

---

## Troubleshooting

### Module Not Found Errors

```bash
pip install -r requirements.txt
```

### Selenium/ChromeDriver Issues

The system uses webdriver-manager to auto-download ChromeDriver. Ensure:
- Chrome browser is installed
- WebDriver can download automatically

If problems persist:
```bash
pip install webdriver-manager
```

### Blocked by Website

1. Reduce max-pages:
   ```bash
   python main.py --max-pages 5
   ```
2. Check the website's robots.txt
3. Increase delay between requests (edit `crawler/resilience/delay.py`)

### No Output Files

Make sure to run without `--no-save`:
```bash
python main.py
```

### Python Path Issues

Ensure you're in the correct directory:
```bash
cd comp3071-coursework-u
python main.py
```

### Permission Errors (Windows)

Run PowerShell as Administrator or use:
```bash
pip install --user -r requirements.txt
```

---

## Requirements

### Python

- Python 3.8 or higher

### Dependencies

```
beautifulsoup4>=4.9.0
requests>=2.25.0
selenium>=4.0.0
numpy>=1.20.0
scikit-learn>=1.0.0
urllib3>=1.26.0
webdriver-manager>=3.8.0
lxml>=4.6.0
```

