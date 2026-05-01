import time


MAX_RETRIES = 3


def retry_request(fetch_function, url):

    attempts = 0

    while attempts < MAX_RETRIES:

        html, status = fetch_function(url)

        if status == 200:
            return html

        if status in [403, 429]:

            print(f"Blocked ({status}). Retrying...")

            time.sleep(5)

            attempts += 1

        else:
            return None

    print("Max retries reached")

    return None