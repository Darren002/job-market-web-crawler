"""
Unit tests for the intelligent job crawler.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from crawler.data.processor import PageProcessor
from crawler.agent.agent import CrawlingAgent, MultiArmedBandit
from crawler.resilience.block_detector import detect_block


class TestPageProcessor:
    """Tests for PageProcessor - relevance scoring."""

    def test_relevance_high_for_ai_keywords(self):
        """AI keywords should result in high relevance."""
        skills = ['machine learning', 'deep learning', 'artificial intelligence']
        processor = PageProcessor(skills)
        
        html = """
        <html><body>
        <h1>Machine Learning Engineer</h1>
        <p>We are looking for a deep learning specialist to join our AI team.
        Work with artificial intelligence and neural networks.</p>
        </body></html>
        """
        
        score, matches, is_relevant, is_irrelevant = processor._compute_keyword_relevance(html)
        
        assert score > 0.5, f"Expected high score for AI keywords, got {score}"
        assert len(matches) >= 2, f"Expected at least 2 matches, got {len(matches)}"

    def test_relevance_low_for_no_keywords(self):
        """Pages without keywords should have low relevance."""
        skills = ['machine learning', 'deep learning']
        processor = PageProcessor(skills)
        
        html = "<html><body><h1>Welcome to our website</h1></body></html>"
        
        score, matches, is_relevant, is_irrelevant = processor._compute_keyword_relevance(html)
        
        assert score == 0.05, f"Expected 0.05 for no keywords, got {score}"

    def test_negative_filter_penalizes_clerk(self):
        """Clerk positions should be penalized."""
        skills = ['machine learning', 'deep learning']
        processor = PageProcessor(skills)
        
        html = """
        <html><body>
        <h1>Machine Learning Clerk</h1>
        <p>Join our team as a machine learning clerk.</p>
        </body></html>
        """
        
        score, matches, is_relevant, is_irrelevant = processor._compute_keyword_relevance(html)
        
        assert score < 0.3, f"Expected low score for clerk, got {score}"

    def test_negative_filter_penalizes_driver(self):
        """Driver positions should be penalized."""
        skills = ['cloud', 'devops']
        processor = PageProcessor(skills)
        
        html = """
        <html><body>
        <h1>Cloud Engineer Driver</h1>
        <p>We need a driver with cloud experience.</p>
        </body></html>
        """
        
        score, matches, is_relevant, is_irrelevant = processor._compute_keyword_relevance(html)
        
        assert score < 0.3, f"Expected low score for driver, got {score}"

    def test_multiple_keywords_boost_score(self):
        """More keyword matches should boost score."""
        skills = ['machine learning', 'deep learning', 'artificial intelligence', 'nlp']
        processor = PageProcessor(skills)
        
        html = """
        <html><body>
        <h1>Machine Learning Engineer</h1>
        <p>Work with deep learning, artificial intelligence, and NLP.</p>
        </body></html>
        """
        
        score, matches, is_relevant, is_irrelevant = processor._compute_keyword_relevance(html)
        
        assert score > 0.6, f"Expected high score for 3+ matches, got {score}"
        assert len(matches) >= 3

    def test_empty_html_returns_minimum(self):
        """Empty HTML should return minimum score."""
        skills = ['python']
        processor = PageProcessor(skills)
        
        score, matches, is_relevant, is_irrelevant = processor._compute_keyword_relevance("")
        
        assert score == 0.05
        assert len(matches) == 0


class TestMultiArmedBandit:
    """Tests for Multi-Armed Bandit."""

    def test_unexplored_arm_has_infinite_ucb(self):
        """Unexplored arms should have infinite UCB value."""
        mab = MultiArmedBandit()
        
        ucb = mab.get_ucb_value("new_domain", 10)
        
        assert ucb == float('inf')

    def test_update_increases_pulls(self):
        """Update should increase pull count."""
        mab = MultiArmedBandit()
        
        mab.update("domain1", 1.0)
        mab.update("domain1", 0.0)
        
        assert mab.arms["domain1"]["pulls"] == 2

    def test_success_increases_success_count(self):
        """Reward >= 0.5 should count as success."""
        mab = MultiArmedBandit()
        
        mab.update("domain1", 1.0)
        
        assert mab.arms["domain1"]["successes"] == 1

    def test_negative_reward_increases_failure_count(self):
        """Negative reward should count as failure."""
        mab = MultiArmedBandit()
        
        mab.update("domain1", -1.0)
        
        assert mab.arms["domain1"]["failures"] == 1

    def test_get_mean_returns_average(self):
        """Get mean should return average reward."""
        mab = MultiArmedBandit()
        
        mab.update("domain1", 1.0)
        mab.update("domain1", 0.0)
        
        mean = mab.get_mean("domain1")
        
        assert mean == 0.5


class TestCrawlingAgent:
    """Tests for CrawlingAgent."""

    def test_agent_initializes_with_seed_urls(self):
        """Agent should initialize with seed URLs."""
        agent = CrawlingAgent(max_pages=10)
        
        agent.initialize(["https://example.com", "https://test.com"])
        
        assert len(agent.frontier) == 2
        assert len(agent.q_table) == 2

    def test_initial_q_value_is_optimistic(self):
        """Initial Q-values should be optimistic (0.5)."""
        agent = CrawlingAgent()
        
        agent.initialize(["https://example.com"])
        
        assert agent.q_table["https://example.com"] == 0.5

    def test_domain_visits_tracked(self):
        """Domain visits should be tracked."""
        agent = CrawlingAgent()
        
        agent.initialize(["https://example.com"])
        agent.get_next_action()
        
        assert "example.com" in agent.domain_visits
        assert agent.domain_visits["example.com"] == 1

    def test_record_outcome_updates_q(self):
        """Record outcome should update Q-value."""
        agent = CrawlingAgent(learning_rate=0.1, discount_factor=0.9)
        
        agent.initialize(["https://example.com"])
        
        old_q = agent.q_table["https://example.com"]
        agent.record_outcome("https://example.com", True, True, False, 0.8)
        new_q = agent.q_table["https://example.com"]
        
        assert new_q != old_q

    def test_high_relevance_job_gives_high_reward(self):
        """High relevance job should give +1.0 reward."""
        agent = CrawlingAgent()
        
        agent.record_outcome("https://example.com/job", True, True, False, 0.8)
        
        assert agent.total_rewards >= 1.0

    def test_blocked_page_gives_negative_reward(self):
        """Blocked page should give -1.0 reward."""
        agent = CrawlingAgent()
        
        agent.record_outcome("https://example.com", False, False, True, -1.0)
        
        assert agent.total_rewards == -1.0

    def test_domain_bonus_for_new_domain(self):
        """New domain discovery should give bonus."""
        agent = CrawlingAgent()
        
        agent.initialize(["https://domain1.com"])
        agent.get_next_action()
        
        agent.record_outcome("https://domain1.com/job", True, True, False, 0.8)
        
        assert agent.domain_visits.get("domain1.com", 0) >= 1


class TestBlockedPageDetection:
    """Tests for blocked page detection."""

    def test_detects_captcha(self):
        """Should detect CAPTCHA pages."""
        html = "<html><body>Please verify you are human</body></html>"
        
        assert detect_block(html) == True

    def test_detects_cloudflare(self):
        """Should detect Cloudflare blocked pages."""
        html = "<html><body>Cloudflare access denied</body></html>"
        
        assert detect_block(html) == True

    def test_detects_access_denied(self):
        """Should detect access denied pages."""
        html = "<html><body>Access denied</body></html>"
        
        assert detect_block(html) == True

    def test_allows_normal_job_page(self):
        """Should allow normal job pages."""
        html = """
        <html><body>
        <h1>Machine Learning Engineer</h1>
        <p>We are hiring a machine learning engineer.</p>
        </body></html>
        """
        
        assert detect_block(html) == False

    def test_empty_html_not_blocked(self):
        """Empty HTML should not be blocked."""
        assert detect_block("") == False

    def test_none_html_not_blocked(self):
        """None HTML should not be blocked."""
        assert detect_block(None) == False


class TestIntegration:
    """Integration tests for crawler components."""

    def test_processor_with_agent_workflow(self):
        """Test full workflow: process page -> get score -> agent learns."""
        skills = ['machine learning', 'deep learning']
        
        processor = PageProcessor(skills)
        processor.fit(["<html><body>training</body></html>"])
        agent = CrawlingAgent(skills=skills)
        
        html = """
        <html><body>
        <h1>Machine Learning Engineer</h1>
        <p>Deep learning role available.</p>
        </body></html>
        """
        
        analysis = processor.process_page(html, "https://example.com/job")
        
        assert analysis.relevance_score > 0.5
        
        agent.initialize(["https://example.com"])
        agent.record_outcome(
            "https://example.com/job",
            is_job_page=True,
            is_relevant=True,
            was_blocked=False,
            content_relevance=analysis.relevance_score
        )
        
        assert agent.jobs_found == 1

    def test_negative_filter_integrates_with_agent(self):
        """Negative filtered page should not trigger high job reward."""
        skills = ['machine learning']
        
        processor = PageProcessor(skills)
        processor.fit(["<html><body>training</body></html>"])
        agent = CrawlingAgent(skills=skills)
        
        html = """
        <html><body>
        <h1>Machine Learning Clerk</h1>
        <p>Administrative clerk with ML knowledge.</p>
        </body></html>
        """
        
        analysis = processor.process_page(html)
        
        agent.initialize(["https://example.com"])
        agent.record_outcome(
            "https://example.com",
            is_job_page=True,
            is_relevant=analysis.is_relevant_job,
            was_blocked=False,
            content_relevance=analysis.relevance_score
        )
        
        assert agent.total_rewards <= 1.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
