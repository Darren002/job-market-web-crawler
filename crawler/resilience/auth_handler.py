"""
Authentication Handler for job sites that require login.

How it works:
1. detect_login_wall() checks if the current page is asking for credentials
2. attempt_login() uses Selenium to fill in the form and submit
3. Once logged in, the browser session is reused for all future pages on that domain
4. Credentials are stored in a config dict - never hardcoded into crawl logic

Supported sites (configured in SITE_CREDENTIALS):
- Hiredly (hiredly.com)
- JobStreet (jobstreet.com.my)
- Add more by following the same pattern

IMPORTANT: Only use accounts you own. Respect each site's Terms of Service.
"""

import time
import logging
from urllib.parse import urlparse
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION: Add your credentials here
# Each entry maps a domain to its login page and form selectors
# ─────────────────────────────────────────────────────────────────
SITE_CREDENTIALS = {
    "hiredly.com": {
        "login_url": "https://hiredly.com/login",
        "email": "mukthykrishnan@gmail.com",       # <-- Replace
        "password": "(Mukthy040506)",              # <-- Replace
        "email_selector": "input[name='email']",
        "password_selector": "input[name='password']",
        "submit_selector": "button[type='submit']",
        "success_indicator": "dashboard",          # string that appears in URL after login
    },
    "www.jobstreet.com.my": {
        "login_url": "https://www.jobstreet.com.my/en/cms/candidate/login/",
        "email": "mukthykrishnan@gmail.com",       # <-- Replace
        "password": "(Mukthy040506)",              # <-- Replace
        "email_selector": "input[data-automation='emailAddress']",
        "password_selector": "input[data-automation='password']",
        "submit_selector": "button[data-automation='login-button']",
        "success_indicator": "profile",
    },
    # Add more sites following the same structure
}

# Track which domains we've already logged into this session
_logged_in_domains = set()


# ─────────────────────────────────────────────────────────────────
# LOGIN WALL DETECTION
# ─────────────────────────────────────────────────────────────────

LOGIN_WALL_KEYWORDS = [
    "sign in to view",
    "login to view",
    "please log in",
    "please sign in",
    "create an account to",
    "register to view",
    "sign up to apply",
    "login required",
    "you must be logged in",
]

def detect_login_wall(html: str, url: str) -> bool:
    """
    Returns True if this page is a login wall (not a crawl block).

    This is DIFFERENT from block detection:
    - Block = site doesn't want bots at all (403, CAPTCHA)
    - Login wall = site wants you to authenticate as a user

    The distinction matters because:
    - A block → slow down, rotate headers, or skip
    - A login wall → authenticate and retry
    """
    if not html:
        return False

    page = html.lower()

    # Check for login wall phrases
    for keyword in LOGIN_WALL_KEYWORDS:
        if keyword in page:
            logger.info(f"Login wall detected at {url} (keyword: '{keyword}')")
            return True

    # Check if we're on a login/signin page itself
    parsed = urlparse(url)
    path = parsed.path.lower()
    if any(word in path for word in ["/login", "/signin", "/sign-in", "/auth"]):
        logger.info(f"Login page URL detected: {url}")
        return True

    return False


def is_domain_authenticated(url: str) -> bool:
    """Check if we've already logged into this domain in the current session."""
    domain = urlparse(url).netloc
    return domain in _logged_in_domains


def get_credentials(url: str) -> dict | None:
    """Look up credentials for a given URL's domain."""
    domain = urlparse(url).netloc
    # Try exact match first, then partial match
    if domain in SITE_CREDENTIALS:
        return SITE_CREDENTIALS[domain]
    for configured_domain, creds in SITE_CREDENTIALS.items():
        if configured_domain in domain or domain in configured_domain:
            return creds
    return None


# ─────────────────────────────────────────────────────────────────
# LOGIN EXECUTION
# ─────────────────────────────────────────────────────────────────

def attempt_login(browser, url: str) -> bool:
    """
    Use Selenium to log into the site for a given URL.

    Flow:
    1. Navigate to the login page for this domain
    2. Wait for the email/password fields to appear
    3. Type credentials and submit
    4. Wait to confirm login succeeded
    5. Mark domain as authenticated

    Returns True if login succeeded, False otherwise.
    """
    creds = get_credentials(url)
    if not creds:
        logger.warning(f"No credentials configured for {url}")
        return False

    domain = urlparse(url).netloc

    # Check placeholder credentials
    if "YOUR_EMAIL" in creds["email"] or "YOUR_PASSWORD" in creds["password"]:
        logger.warning(
            f"Credentials for {domain} are still placeholder values. "
            "Edit SITE_CREDENTIALS in auth_handler.py to set real credentials."
        )
        return False

    logger.info(f"Attempting login for {domain}...")

    try:
        # Navigate to login page
        browser.get(creds["login_url"])
        wait = WebDriverWait(browser, 15)

        # Wait for and fill email field
        email_field = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, creds["email_selector"]))
        )
        email_field.clear()
        email_field.send_keys(creds["email"])
        time.sleep(0.5)

        # Fill password field
        password_field = browser.find_element(By.CSS_SELECTOR, creds["password_selector"])
        password_field.clear()
        password_field.send_keys(creds["password"])
        time.sleep(0.5)

        # Click submit
        submit_btn = browser.find_element(By.CSS_SELECTOR, creds["submit_selector"])
        submit_btn.click()

        # Wait for login to complete - check URL or page content
        time.sleep(3)

        current_url = browser.current_url.lower()
        page_source = browser.page_source.lower()

        success_indicator = creds.get("success_indicator", "")

        if (success_indicator and success_indicator in current_url) or \
           "login" not in current_url:
            logger.info(f"Login succeeded for {domain}")
            _logged_in_domains.add(domain)
            return True
        else:
            logger.warning(f"Login may have failed for {domain}. Current URL: {current_url}")
            return False

    except TimeoutException:
        logger.error(f"Timeout waiting for login form at {domain}")
        return False
    except NoSuchElementException as e:
        logger.error(f"Login form element not found for {domain}: {e}")
        return False
    except Exception as e:
        logger.error(f"Login failed for {domain}: {e}")
        return False


def fetch_with_auth(browser, url: str) -> str | None:
    """
    Fetch a URL after ensuring the session is authenticated.

    Called by browser_fetcher.py when a login wall is detected.

    Steps:
    1. If already logged in → just fetch the page
    2. If not logged in → attempt login, then fetch
    3. Returns HTML string or None if auth failed
    """
    domain = urlparse(url).netloc

    if domain not in _logged_in_domains:
        success = attempt_login(browser, url)
        if not success:
            logger.warning(f"Could not authenticate for {domain}. Skipping authenticated fetch.")
            return None

    # Now fetch the originally requested URL
    logger.info(f"Fetching {url} with authenticated session")
    browser.get(url)
    time.sleep(4)  # Wait for JS rendering

    html = browser.page_source
    logger.info(f"Authenticated fetch complete. HTML length: {len(html)}")
    return html