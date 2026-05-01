import requests

from .resilience.headers import get_headers
from .resilience.delay import human_delay
from .resilience.browser_fetcher import fetch_with_browser
from .resilience.block_detector import detect_block


session = requests.Session()


def fetch_page(url):

    human_delay()

    headers = get_headers()

    try:

        response = session.get(
            url,
            headers=headers,
            timeout=10
        )

        if response.status_code == 200:

            html = response.text

            if detect_block(html):

                print("CAPTCHA or bot block detected -> switching to browser")

                return fetch_with_browser(url)

            return html

        if response.status_code in [403, 429]:

            print("Blocked -> switching to browser")

            return fetch_with_browser(url)

        print(f"Failed: {response.status_code}")

        return None

    except Exception:

        print("Request failed -> using browser")

        return fetch_with_browser(url)