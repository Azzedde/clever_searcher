"""Simple content scorer without embeddings for fallback mode"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..utils.config import settings
from .fetcher import ContentDocument
from .summarizer import StructuredSummary

logger = logging.getLogger(__name__)


class SimpleContentScorer:
    """Simple content scorer without embeddings"""
    
    def __init__(self) -> None:
        # Scoring weights
        self.weights: Dict[str, float] = {
            'content_quality': 0.30,
            'source_trust': 0.25,
            'freshness': 0.20,
            'relevance': 0.15,
            'length_score': 0.10,
        }
        
        # Source trust scores
        self.source_trust_scores = {
            'arxiv.org': 0.95,
            'nature.com': 0.95,
            'science.org': 0.95,
            'github.com': 0.85,
            'stackoverflow.com': 0.85,
            'techcrunch.com': 0.75,
            'venturebeat.com': 0.75,
            'coindesk.com': 0.70,
            'medium.com': 0.60,
            'reddit.com': 0.50,
        }
    
    def score_content(
        self,
        document: ContentDocument,
        summary: Optional[StructuredSummary] = None,
        query: str = "",
        category: str = "",
    ) -> float:
        """Score content based on simple factors"""
        
        scores = {}
        
        # Content quality score
        scores['content_quality'] = self._score_content_quality(document, summary)
        
        # Source trust score
        scores['source_trust'] = self._score_source_trust(document.domain)
        
        # Freshness score
        scores['freshness'] = self._score_freshness(document.published_date)
        
        # Relevance score
        scores['relevance'] = self._score_relevance(document, summary, query, category)
        
        # Length score
        scores['length_score'] = self._score_length(document)
        
        # Calculate weighted final score
        final_score = sum(
            scores[factor] * self.weights[factor]
            for factor in scores
        )
        
        logger.debug(f"Scored {document.url}: {final_score:.3f} {scores}")
        return float(final_score)
    
    def _score_content_quality(
        self,
        document: ContentDocument,
        summary: Optional[StructuredSummary] = None,
    ) -> float:
        """Score based on content quality indicators"""
        score = 0.5  # Base score
        
        # Title quality
        if document.title and len(document.title) > 10:
            score += 0.2
        
        # Author presence
        if document.author:
            score += 0.1
        
        # Content length indicators
        if document.content_length > 1000:
            score += 0.2
        elif document.content_length > 500:
            score += 0.1
        
        # Summary quality (if available)
        if summary:
            if summary.confidence_score > 0.8:
                score += 0.1
            if len(summary.entities) > 3:
                score += 0.05
            if len(summary.bullets) >= 3:
                score += 0.05
        
        return min(1.0, score)
    
    def _score_source_trust(self, domain: str) -> float:
        """Score based on source trustworthiness"""
        return float(self.source_trust_scores.get(domain.lower(), 0.5))
    
    def _score_freshness(self, published_date: Optional[datetime]) -> float:
        """Score based on content freshness"""
        if not published_date:
            return 0.5  # Neutral score for unknown dates
        
        now = datetime.utcnow()
        age_days = (now - published_date).days
        
        # Decay function: newer content gets higher scores
        if age_days <= 1:
            return 1.0
        elif age_days <= 7:
            return 0.9
        elif age_days <= 30:
            return 0.7
        elif age_days <= 90:
            return 0.5
        elif age_days <= 365:
            return 0.3
        else:
            return 0.1
    
    def _score_relevance(
        self,
        document: ContentDocument,
        summary: Optional[StructuredSummary],
        query: str,
        category: str,
    ) -> float:
        """Score based on relevance to query and category"""
        if not query and not category:
            return 0.5
        
        # Combine text for relevance checking
        text_parts = [document.title, document.content[:1000]]
        if summary:
            text_parts.extend([summary.tldr, ' '.join(summary.tags)])
        
        combined_text = ' '.join(filter(None, text_parts)).lower()
        
        score = 0.0
        
        # Query relevance
        if query:
            query_words = set(query.lower().split())
            text_words = set(combined_text.split())
            overlap = len(query_words.intersection(text_words))
            query_score = min(1.0, overlap / max(1, len(query_words)))
            score += query_score * 0.6
        
        # Category relevance (simple keyword matching)
        if category:
            category_keywords = category.lower().replace('_', ' ').split()
            category_overlap = sum(1 for kw in category_keywords if kw in combined_text)
            category_score = min(1.0, category_overlap / max(1, len(category_keywords)))
            score += category_score * 0.4
        
        return score
    
    def _score_length(self, document: ContentDocument) -> float:
        """Score based on content length (sweet spot around 1000-5000 chars)"""
        length = document.content_length
        
        if length < 200:
            return 0.1  # Too short
        elif length < 500:
            return 0.3
        elif length < 1000:
            return 0.6
        elif length < 5000:
            return 1.0  # Sweet spot
        elif length < 10000:
            return 0.8
        else:
            return 0.6  # Too long


class SimplePersonalizationEngine:
    """Simple personalization without embeddings"""
    
    def __init__(self) -> None:
        self.scorer = SimpleContentScorer()
        self.user_feedback: Dict[str, List[float]] = {}
    
    def score_documents(
        self,
        documents: List[ContentDocument],
        summaries: Optional[List[StructuredSummary]] = None,
        query: str = "",
        category: str = "",
        user_id: str = "default",
    ) -> List[Tuple[ContentDocument, float]]:
        """Score and rank documents"""
        
        scored_docs = []
        for i, doc in enumerate(documents):
            summary = summaries[i] if summaries and i < len(summaries) else None
            
            score = self.scorer.score_content(
                document=doc,
                summary=summary,
                query=query,
                category=category,
            )
            
            # Apply simple user preference adjustment
            if user_id in self.user_feedback:
                feedback_scores = self.user_feedback[user_id]
                if feedback_scores:
                    avg_feedback = sum(feedback_scores) / len(feedback_scores)
                    # Slight adjustment based on historical feedback
                    score = score * (0.9 + 0.2 * avg_feedback)
            
            scored_docs.append((doc, score))
        
        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs
    
    def record_feedback(
        self,
        content_hash: str,
        feedback_type: str,
        feedback_value: float,
        user_id: str = "default",
        content_text: str = "",
    ) -> None:
        """Record user feedback"""
        
        if user_id not in self.user_feedback:
            self.user_feedback[user_id] = []
        
        self.user_feedback[user_id].append(feedback_value)
        
        # Keep only last 50 feedback items
        if len(self.user_feedback[user_id]) > 50:
            self.user_feedback[user_id] = self.user_feedback[user_id][-50:]
        
        logger.info(f"Added simple feedback for user {user_id}: {feedback_type}={feedback_value}")
    
    def get_personalization_stats(self, user_id: str = "default") -> Dict[str, Any]:
        """Get personalization statistics"""
        feedback_count = len(self.user_feedback.get(user_id, []))
        avg_feedback = 0.0
        if feedback_count > 0:
            avg_feedback = sum(self.user_feedback[user_id]) / feedback_count
        
        return {
            'user_id': user_id,
            'feedback_count': feedback_count,
            'avg_feedback': avg_feedback,
            'scorer_type': 'simple',
        }


# Simple fallback instances
simple_scorer = SimpleContentScorer()
simple_personalization = SimplePersonalizationEngine()