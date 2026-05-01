import numpy as np
from typing import Optional
import logging

from .similarity import cosine_similarity_batch

logger = logging.getLogger(__name__)


class Deduplicator:
    """
    Deduplication using batch similarity computations.
    
    IMPORTANT: Check BEFORE adding to avoid inserting duplicates.
    Flow: is_duplicate = check(vector) -> if not is_duplicate: add(vector)
    """

    def __init__(self, threshold: float = 0.9, batch_size: int = 100):
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        if batch_size < 1:
            raise ValueError("Batch size must be positive")

        self.threshold = threshold
        self.batch_size = batch_size
        self._vectors: list[np.ndarray] = []
        self._pending: list[np.ndarray] = []

    def check(self, vector: np.ndarray) -> tuple[bool, Optional[int], float]:
        """
        Check if vector is duplicate WITHOUT adding it.
        Returns: (is_duplicate, duplicate_index, max_similarity)
        """
        if vector.size == 0 or np.linalg.norm(vector) == 0:
            logger.debug("Empty/zero vector - not duplicate")
            return False, None, 0.0

        all_vectors = list(self._vectors) + list(self._pending)

        if not all_vectors:
            logger.debug("No stored vectors yet - not duplicate")
            return False, None, 0.0

        similarities = cosine_similarity_batch(vector, np.array(all_vectors))
        max_sim = float(np.max(similarities))
        max_idx = int(np.argmax(similarities)) if max_sim > self.threshold else None

        logger.debug(f"Max similarity: {max_sim:.4f}, threshold: {self.threshold}")
        
        is_dup = max_sim > self.threshold
        if is_dup:
            logger.info(f"DUPLICATE detected (similarity={max_sim:.4f})")
        else:
            logger.debug(f"Not duplicate (similarity={max_sim:.4f} < {self.threshold})")

        return is_dup, max_idx, max_sim

    def add(self, vector: np.ndarray) -> None:
        """Add vector to storage. Call AFTER check to avoid duplicates."""
        if vector.size == 0 or np.linalg.norm(vector) == 0:
            return

        self._pending.append(vector.copy())
        logger.debug(f"Added vector to pending. Pending count: {len(self._pending)}")

        if len(self._pending) >= self.batch_size:
            self._flush_pending()

    def _flush_pending(self) -> None:
        if not self._pending:
            return
        self._vectors.extend(self._pending)
        self._pending.clear()
        logger.info(f"Flushed pending vectors. Total stored: {len(self._vectors)}")

    def finalize(self) -> None:
        self._flush_pending()

    @property
    def count(self) -> int:
        return len(self._vectors) + len(self._pending)

    def reset(self) -> None:
        self._vectors.clear()
        self._pending.clear()
        logger.info("Deduplicator reset")
