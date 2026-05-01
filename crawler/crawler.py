from .fetcher import fetch_page
from .extractor import extract_links
from .qlearning import QLearner
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from .integration import DataLayer


MAX_PAGES = 50


def load_seeds():

    with open("seeds.txt", "r") as f:
        return [line.strip() for line in f if line.strip()]


def is_job_page(url):

    url = url.lower()

    return any()


def link_priority(url):

    url = url.lower()

    if any():
        return 0

    return 1


def crawl_page(url):
    html = fetch_page(url)

    if not html:
        return None, []

    links = extract_links(html, url)

    return html, links


def fetch_page_html(url):
    return fetch_page(url)

def build_state(relevance, is_job, num_links):

    if num_links < 10:
        size = "small"
    elif num_links < 50:
        size = "medium"
    else:
        size = "large"

    return (round(relevance, 2), is_job, size)


def choose_next_link(qlearner, state, links):
    return qlearner.choose_action(state, links)

def crawl():
    qlearner = QLearner()
    seeds = load_seeds()

    queue = deque()
    
    data_layer = DataLayer()

    training_pages = [
        "<html><body>Software Engineer Python AWS</body></html>",
        "<html><body>Data Scientist Machine Learning SQL</body></html>"
    ]

    data_layer.initialize(training_pages)

    for seed in seeds:
        queue.append((link_priority(seed), seed))

    visited = set()

    pages = 0

    with ThreadPoolExecutor(max_workers=5) as executor:

        while queue and pages < MAX_PAGES:

            priority, url = queue.popleft()

            if url in visited:
                continue

            visited.add(url)

            pages += 1

            print(f"\nVisiting: {url}")

            future = executor.submit(crawl_page, url)

            html, links = future.result()

            if not html:
                continue
            links = [l for l in links if l not in visited]
            if not links:
             continue

            result = data_layer.process(html, url)
            relevance = result["relevance_score"]
            is_job = is_job_page(url)

            state = build_state(relevance, is_job, len(links))
            print("STATE:", state)
            if is_job_page(url):
                print("Job page detected")

            print(f"Found {len(links)} links")
            print("SAMPLE LINKS:", links[:3]) 
            next_link = choose_next_link(qlearner, state, links)
            print("CHOSEN LINK:", next_link)
            if next_link and next_link not in visited:
                queue.append((link_priority(next_link), next_link))

                # simple reward (can improve later)
                reward = relevance * 1.5 if is_job else relevance

                next_state = state  # (basic version)

                qlearner.update(
                    state,
                    action=next_link,
                    reward=reward,
                    next_state=next_state,
                    next_actions=links
                )
                print("Q-table size:", len(qlearner.q_table))


if __name__ == "__main__":
    crawl()
