"""
Browser-based page fetcher using Selenium.

Handles:
- JavaScript-rendered pages (SPAs like Hiredly, JobStreet)
- Login walls - automatically authenticates using auth_handler
- Falls back gracefully if login is not configured
- Thread-safe: each thread gets its own browser instance
"""

import threading
import time
import logging

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

logger = logging.getLogger(__name__)

# Thread-local storage — each thread gets its own browser, no sharing
_thread_local = threading.local()


def _create_browser():
    """Create a fresh Chrome browser instance."""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return webdriver.Chrome(options=options)


def get_browser():
    """Get or create a thread-local Selenium browser instance."""
    if not hasattr(_thread_local, 'browser') or _thread_local.browser is None:
        _thread_local.browser = _create_browser()
    else:
        # Check if the session is still alive
        try:
            _ = _thread_local.browser.current_url
        except Exception:
            logger.warning("Browser session died — restarting")
            _thread_local.browser = _create_browser()
    return _thread_local.browser


def fetch_with_browser(url):
    """
    Fetch a page using Selenium (thread-safe).

    Returns HTML string or None on failure.
    """
    b = get_browser()

    try:
        b.get(url)
        time.sleep(5)  # Wait for JS to render
        html = b.page_source
    except Exception as e:
        logger.error(f"Browser fetch failed for {url}: {e}")
        return None

    # Auth handler check is separate — failure here still returns the HTML
    try:
        from .auth_handler import detect_login_wall, fetch_with_auth
        if detect_login_wall(html, url):
            logger.info(f"Login wall detected at {url} — attempting authentication")
            authenticated_html = fetch_with_auth(b, url)
            if authenticated_html:
                return authenticated_html
            else:
                logger.warning(f"Auth not available for {url} — returning raw HTML")
    except Exception as e:
        logger.warning(f"Auth handler unavailable ({e}) — returning raw HTML")

    return html