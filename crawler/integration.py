"""
DataLayer - Integration layer for TF-IDF processing and deduplication.

This module provides a unified interface for:
- TF-IDF vectorization
- Content similarity computation
- Duplicate detection
- Relevance scoring
"""
import logging
from typing import Dict, List, Optional
import numpy as np

from crawler.data.processor import PageProcessor
from crawler.data.cleaner import HTMLCleaner

logger = logging.getLogger(__name__)


class DataLayer:
    """
    Data processing layer for the crawler.
    
    Provides:
    - TF-IDF vectorization
    - Relevance scoring against skill keywords
    - Duplicate detection
    - HTML cleaning
    """
    
    def __init__(self, skills: List[str] = None):
        self.skills = skills or []
        self.processor = None
        self.cleaner = HTMLCleaner()
        self._stats = {
            'pages_processed': 0,
            'duplicates_found': 0,
            'vocab_size': 0
        }
    
    def initialize(self, training_htmls: List[str]):
        """Initialize the data layer with training documents."""
        if not self.skills:
            self.skills = [
                "python", "java", "javascript", "typescript",
                "machine learning", "deep learning", "ai",
                "aws", "azure", "cloud", "docker", "kubernetes",
                "sql", "database", "postgresql", "mongodb",
                "react", "angular", "vue", "frontend",
                "backend", "api", "rest", "graphql",
                "git", "agile", "scrum", "devops"
            ]
        
        self.processor = PageProcessor(skills=self.skills)
        self.processor.fit(training_htmls)
        self._stats['vocab_size'] = self.processor.vocab_size
        logger.info(f"DataLayer initialized. Vocab size: {self._stats['vocab_size']}")
    
    def process(self, html: str, url: str) -> Dict:
        """
        Process a page: vectorize, score relevance, check duplicates.
        
        Returns:
            Dict with keys: vector, relevance_score, is_duplicate, 
                           duplicate_of_index, similarity_score, word_count,
                           is_relevant_job, is_irrelevant_job
        """
        if self.processor is None:
            raise RuntimeError("DataLayer not initialized. Call initialize() first.")
        
        self._stats['pages_processed'] += 1
        
        result = self.processor.process_page(html, url)
        
        if result.is_duplicate:
            self._stats['duplicates_found'] += 1
        
        return {
            'vector': result.vector,
            'relevance_score': result.relevance_score,
            'is_duplicate': result.is_duplicate,
            'duplicate_of_index': result.duplicate_of_index,
            'similarity_score': result.similarity_score,
            'word_count': result.word_count,
            'is_relevant_job': result.is_relevant_job,
            'is_irrelevant_job': result.is_irrelevant_job
        }
    
    def clean_html(self, html: str) -> str:
        """Clean HTML to plain text."""
        return self.cleaner.clean(html)
    
    def get_stats(self) -> Dict:
        """Get processing statistics."""
        stats = self._stats.copy()
        if self.processor:
            stats['vocab_size'] = self.processor.vocab_size
            stats['unique_pages'] = self.processor.unique_pages
        return stats
