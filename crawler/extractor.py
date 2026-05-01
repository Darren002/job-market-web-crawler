from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse


BLACKLIST = [
    "login",
    "signup",
    "register",
    "password",
    "privacy",
    "terms"
]


def extract_links(html, base_url):

    soup = BeautifulSoup(html, "html.parser")

    base_domain = urlparse(base_url).netloc

    links = set()

    for tag in soup.find_all("a", href=True):

        link = urljoin(base_url, tag["href"])

        if link.startswith("javascript:"):
            continue

        if link.startswith("mailto:"):
            continue

        if not link.startswith("http"):
            continue

        link_domain = urlparse(link).netloc

        if link_domain != base_domain:
            continue

        if any(word in link.lower() for word in BLACKLIST):
            continue

        links.add(link)

    return list(links)