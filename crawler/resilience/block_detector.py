CAPTCHA_KEYWORDS = [
    "captcha",
    "verify you are human",
    "cloudflare",
    "access denied"
]


def detect_block(html):

    if not html:
        return False

    page = html.lower()

    for word in CAPTCHA_KEYWORDS:

        if word in page:
            return True

    return False