from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import logging

from .cleaner import HTMLCleaner

logger = logging.getLogger(__name__)


class TFIDFVectorizer:
    """
    Reusable TF-IDF vectorizer that fits once and transforms many times.
    Handles small corpuses robustly (e.g., starting crawler with few docs).
    """

    def __init__(
        self,
        max_features: int = 5000,
        min_df: int = 1,
        max_df: float = 1.0,
        ngram_range: tuple = (1, 2)
    ):
        self._max_features = max_features
        self._ngram_range = ngram_range
        self._vectorizer = None
        self._cleaner = HTMLCleaner()
        self._fitted = False

    def fit(self, html_documents: list[str]) -> 'TFIDFVectorizer':
        if not html_documents:
            raise ValueError("Cannot fit on empty document list")

        logger.info(f"Fitting vectorizer on {len(html_documents)} documents")

        cleaned = [self._cleaner.clean(doc) for doc in html_documents]
        doc_count = len(cleaned)

        if doc_count < 2:
            min_df_val = 1
            max_df_val = 1.0
            logger.debug("Small corpus detected (<2 docs) - using safe TF-IDF config")
        else:
            min_df_val = 1 if doc_count < 5 else 2
            max_df_val = 0.9

        self._vectorizer = TfidfVectorizer(
            max_features=self._max_features,
            min_df=min_df_val,
            max_df=max_df_val,
            ngram_range=self._ngram_range,
            sublinear_tf=True,
            strip_accents='unicode',
            stop_words='english'
        )

        self._vectorizer.fit(cleaned)
        self._fitted = True
        logger.info(f"Vectorizer fitted. Vocab size: {self.vocab_size}")
        return self

    def transform(self, html_document: str) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fitted before transform")

        cleaned = self._cleaner.clean(html_document)
        if not cleaned:
            logger.debug("Empty document after cleaning - returning zero vector")
            return np.zeros(self._vectorizer.transform([""]).shape[1])

        vec = self._vectorizer.transform([cleaned])
        return vec.toarray()[0]

    def transform_batch(self, html_documents: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Vectorizer must be fitted before transform")

        cleaned = [self._cleaner.clean(doc) for doc in html_documents]
        for i, text in enumerate(cleaned):
            if not text:
                cleaned[i] = " "

        return self._vectorizer.transform(cleaned).toarray()

    @property
    def feature_names(self) -> list[str]:
        return self._vectorizer.get_feature_names_out().tolist()

    @property
    def vocab_size(self) -> int:
        return len(self._vectorizer.vocabulary_)
