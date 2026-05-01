from typing import Optional
from dataclasses import dataclass
import numpy as np
import logging
import re

from .vectorizer import TFIDFVectorizer
from .deduplicator import Deduplicator
from .similarity import cosine_similarity

logger = logging.getLogger(__name__)


AI_KEYWORDS = [
    "machine learning", "deep learning", "artificial intelligence", "ai",
    "ml", "nlp", "natural language", "computer vision", "neural network",
    "tensorflow", "pytorch", "keras", "data scientist", "ai engineer",
    "machine learning engineer", "ai researcher", "ml engineer"
]

NEGATIVE_JOB_KEYWORDS = [
    "clerk", "driver", "admin", "assistant", "telemarketing", "sales",
    "cashier", "waiter", "receptionist", "security guard", "cleaner",
    "helper", "labour", "packer", "merchandiser", "crew", "stocker",
    "cashier", "barista", "server", "attendant"
]


@dataclass
class PageAnalysis:
    vector: Optional[np.ndarray]
    relevance_score: float
    is_duplicate: bool
    duplicate_of_index: Optional[int]
    similarity_score: float
    word_count: int
    keyword_matches: list[str]
    is_relevant_job: bool = False
    is_irrelevant_job: bool = False


class PageProcessor:
    """
    Processes HTML pages: vectorizes, scores relevance, deduplicates.

    Flow:
        1. Transform HTML to TF-IDF vector
        2. Compute relevance based on DIRECT keyword matching
        3. Check if duplicate (BEFORE adding)
        4. Add if not duplicate
    """

    def __init__(
        self,
        skills: list[str],
        dedup_threshold: float = 0.8
    ):
        if not skills:
            raise ValueError("At least one skill keyword is required")

        self._skills = [s.lower() for s in skills]
        self._vectorizer = TFIDFVectorizer()
        self._deduplicator = Deduplicator(threshold=dedup_threshold)
        self._skill_vector: Optional[np.ndarray] = None

    def _compute_keyword_relevance(self, html: str, url: str = "") -> tuple[float, list[str], bool, bool]:
        """
        STRICT keyword-based relevance with negative filtering.
        
        Returns:
            - relevance_score: 0.05-1.0
            - keyword_matches: list of matched AI keywords
            - is_relevant_job: True if AI keywords found
            - is_irrelevant_job: True if negative job keywords found (clerk, driver, etc.)
        """
        import re
        
        if not html:
            return 0.05, [], False, False

        html_lower = html.lower()

        # STRICT AI keywords - use word boundaries to avoid partial matches
        ai_matches = []
        for kw in AI_KEYWORDS:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, html_lower):
                ai_matches.append(kw)

        # Check for negative job keywords (use word boundaries)
        has_negative = False
        for neg in NEGATIVE_JOB_KEYWORDS:
            pattern = r'\b' + re.escape(neg) + r'\b'
            if re.search(pattern, html_lower):
                has_negative = True
                break

        # Also check skill keywords (use word boundaries)
        skill_matches = []
        for kw in self._skills:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, html_lower) and kw not in ai_matches:
                skill_matches.append(kw)

        all_matches = ai_matches + skill_matches

        # Determine if this is a relevant vs irrelevant job
        # A job is RELEVANT only if AI keywords found AND NO negative keywords
        is_relevant_job = len(ai_matches) > 0 and not has_negative
        # A job is IRRELEVANT if it has negative keywords
        is_irrelevant_job = has_negative

        # Calculate base score
        if all_matches:
            match_ratio = len(all_matches) / max(len(self._skills), 1)

            if len(all_matches) >= 3:
                base_score = min(1.0, match_ratio * 3.0)
            elif len(all_matches) >= 1:
                base_score = match_ratio * 1.5
            else:
                base_score = match_ratio
        else:
            base_score = 0.0

        # Apply negative penalty for irrelevant jobs
        if has_negative and base_score > 0.2:
            base_score *= 0.2

        relevance = max(0.05, min(1.0, base_score))

        if not all_matches:
            relevance = 0.05

        return relevance, all_matches, is_relevant_job, is_irrelevant_job

    def fit(self, training_htmls: list[str]) -> 'PageProcessor':
        logger.info(f"Training PageProcessor with {len(training_htmls)} documents")
        self._vectorizer.fit(training_htmls)
        skill_text = " ".join(self._skills)
        self._skill_vector = self._vectorizer.transform(skill_text)
        logger.info(f"Skill vector created with {len(self._skill_vector)} features")
        return self

    def process_page(self, html: str, url: str = "") -> PageAnalysis:
        logger.info(f"Processing page: {url}")

        # First: compute STRICT keyword-based relevance
        keyword_score, keyword_matches, is_relevant_job, is_irrelevant_job = self._compute_keyword_relevance(html, url)

        # Second: also compute TF-IDF similarity as backup
        vector = self._vectorizer.transform(html)
        word_count = int(np.sum(vector > 0))
        logger.debug(f"Vector extracted. Non-zero features: {word_count}")

        tfidf_score = 0.0
        if self._skill_vector is not None:
            tfidf_score = cosine_similarity(vector, self._skill_vector)
            # Reduce TF-IDF weight
            tfidf_score *= 0.3

        # Combine: STRICT keyword score is primary
        # Use weighted combination to prevent TF-IDF dominance
        relevance_score = 0.7 * keyword_score + 0.3 * tfidf_score

        # Final clamp
        relevance_score = max(0.05, min(1.0, relevance_score))

        logger.info(f"Relevance score: {relevance_score:.4f} (keywords: {len(keyword_matches)})")

        is_duplicate, dup_idx, sim_score = self._deduplicator.check(vector)
        logger.info(f"Duplicate check: {is_duplicate} (similarity={sim_score:.4f})")

        if not is_duplicate:
            self._deduplicator.add(vector)
            logger.info(f"Added to deduplicator. Total unique: {self._deduplicator.count}")
        else:
            logger.info(f"Skipped (duplicate of index {dup_idx})")

        return PageAnalysis(
            vector=vector,
            relevance_score=relevance_score,
            is_duplicate=is_duplicate,
            duplicate_of_index=dup_idx,
            similarity_score=sim_score,
            word_count=word_count,
            keyword_matches=keyword_matches,
            is_relevant_job=is_relevant_job,
            is_irrelevant_job=is_irrelevant_job
        )

    def finalize(self) -> None:
        self._deduplicator.finalize()
        logger.info(f"Processor finalized. Total unique pages: {self._deduplicator.count}")

    @property
    def vocab_size(self) -> int:
        return self._vectorizer.vocab_size

    @property
    def unique_pages(self) -> int:
        return self._deduplicator.count
    
