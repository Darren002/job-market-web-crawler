from bs4 import BeautifulSoup
import re


class HTMLCleaner:
    """Cleans raw HTML into plain text."""

    def __init__(self):
        self._script_style_pattern = re.compile(
            r'<(script|style)[^>]*>.*?</\1>',
            re.DOTALL | re.IGNORECASE
        )
        self._tag_pattern = re.compile(r'<[^>]+>')
        self._whitespace_pattern = re.compile(r'\s+')
        self._special_chars_pattern = re.compile(r'[^\w\s\-]')

    def clean(self, html: str) -> str:
        if not html or not isinstance(html, str):
            return ""

        text = self._script_style_pattern.sub('', html)
        text = self._tag_pattern.sub(' ', text)
        text = self._special_chars_pattern.sub(' ', text)
        text = self._whitespace_pattern.sub(' ', text)
        text = text.lower().strip()

        return text
