"""
State representation for the web crawling agent.
Models each crawlable page as a state with URL features, content features, depth, and domain info.
"""
from dataclasses import dataclass, field
from urllib.parse import urlparse
from typing import Optional
import numpy as np


@dataclass
class URLFeatures:
    """Features extracted from URL."""
    url: str
    has_job_keywords: bool
    has_career_keywords: bool
    has_tech_keywords: bool
    path_depth: int
    domain: str
    is_secure: bool
    url_length: int
    keyword_score: float = 0.0


@dataclass
class ContentFeatures:
    """Features extracted from page content."""
    keyword_counts: dict = field(default_factory=dict)
    relevance_score: float = 0.0
    word_count: int = 0
    has_dynamic_content: bool = False
    is_job_page: bool = False


@dataclass
class State:
    """
    Complete state representation for a crawlable page.
    
    Attributes:
        url_features: URL-based features
        content_features: Content-based features (populated after fetching)
        crawl_depth: How many hops from seed
        visit_count: Times this URL has been queued
        estimated_relevance: Agent's estimated relevance (Q-value)
        discovered_at: Step when first discovered
    """
    url_features: URLFeatures
    content_features: Optional[ContentFeatures] = None
    crawl_depth: int = 0
    visit_count: int = 0
    estimated_relevance: float = 0.0
    discovered_at: int = 0
    is_terminal: bool = False

    @classmethod
    def from_url(cls, url: str, depth: int = 0, discovered_at: int = 0) -> 'State':
        """Create state from URL before fetching content."""
        url_features = cls._extract_url_features(url, depth)
        return cls(
            url_features=url_features,
            crawl_depth=depth,
            discovered_at=discovered_at
        )

    @staticmethod
    def _extract_url_features(url: str, depth: int) -> URLFeatures:
        """Extract features from URL."""
        parsed = urlparse(url)
        path = parsed.path.lower()
        query = parsed.query.lower()
        full_text = path + " " + query
        
        job_keywords = ['job', 'jobs', 'career', 'careers', 'vacancy', 'vacancies', 
                       'position', 'positions', 'opening', 'openings', 'hiring',
                       'employment', 'recruit', 'hire']
        
        career_keywords = ['apply', 'application', 'resume', 'cv', 'interview',
                          'salary', 'benefits', 'requirements', 'qualifications']
        
        tech_keywords = ['python', 'java', 'javascript', 'aws', 'azure', 'cloud',
                        'machine-learning', 'ml', 'ai', 'data', 'developer',
                        'engineer', 'devops', 'kubernetes', 'docker', 'sql']
        
        job_score = sum(1 for kw in job_keywords if kw in full_text)
        career_score = sum(0.5 for kw in career_keywords if kw in full_text)
        tech_score = sum(0.3 for kw in tech_keywords if kw in full_text)
        
        path_depth = len([p for p in parsed.path.split('/') if p])
        
        return URLFeatures(
            url=url,
            has_job_keywords=job_score > 0,
            has_career_keywords=career_score > 0,
            has_tech_keywords=tech_score > 0,
            path_depth=path_depth,
            domain=parsed.netloc,
            is_secure=parsed.scheme == 'https',
            url_length=len(url),
            keyword_score=job_score + career_score + tech_score
        )

    def update_content_features(self, html: str, relevance: float, is_job: bool):
        """Update state with content features after fetching."""
        self.content_features = ContentFeatures(
            relevance_score=relevance,
            is_job_page=is_job
        )

    def get_feature_vector(self) -> np.ndarray:
        """Convert state to numerical feature vector for scoring."""
        features = [
            1.0 if self.url_features.has_job_keywords else 0.0,
            1.0 if self.url_features.has_career_keywords else 0.0,
            1.0 if self.url_features.has_tech_keywords else 0.0,
            float(self.url_features.path_depth) / 10.0,
            1.0 if self.url_features.is_secure else 0.0,
            self.url_features.keyword_score,
            float(self.crawl_depth) / 10.0,
            1.0 / (self.visit_count + 1),
            self.estimated_relevance,
            1.0 if self.content_features and self.content_features.is_job_page else 0.0
        ]
        return np.array(features)

    def to_dict(self) -> dict:
        """Convert state to dictionary for logging."""
        return {
            'url': self.url_features.url,
            'domain': self.url_features.domain,
            'depth': self.crawl_depth,
            'job_keywords': self.url_features.has_job_keywords,
            'career_keywords': self.url_features.has_career_keywords,
            'tech_keywords': self.url_features.has_tech_keywords,
            'url_keyword_score': self.url_features.keyword_score,
            'is_job_page': self.content_features.is_job_page if self.content_features else False,
            'relevance': self.content_features.relevance_score if self.content_features else 0.0,
            'q_value': self.estimated_relevance,
            'visit_count': self.visit_count
        }
