"""Personalization and content scoring system"""

import logging
import numpy as np
import warnings
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..storage.models import Page, Summary, Feedback, UserProfile
from ..utils.config import settings
from .fetcher import ContentDocument
from .summarizer import StructuredSummary

logger = logging.getLogger(__name__)


class ContentScorer:
    """Scores content based on multiple factors"""
    
    def __init__(self, embedding_model: str = None):
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.embedding_model = None
        self._load_embedding_model()
        
        # Scoring weights
        self.weights = {
            'content_quality': 0.25,
            'source_trust': 0.20,
            'freshness': 0.15,
            'relevance': 0.15,
            'user_preference': 0.15,
            'engagement_signals': 0.10,
        }
        
        # Source trust scores (can be learned over time)
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
    
    def _load_embedding_model(self) -> None:
        """Load the sentence transformer model"""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.embedding_model = None
    
    def score_content(
        self,
        document: ContentDocument,
        summary: Optional[StructuredSummary] = None,
        user_profile: Optional['UserPreferences'] = None,
        query: str = "",
        category: str = "",
    ) -> float:
        """Score content based on multiple factors"""
        
        scores = {}
        
        # Content quality score
        scores['content_quality'] = self._score_content_quality(document, summary)
        
        # Source trust score
        scores['source_trust'] = self._score_source_trust(document.domain)
        
        # Freshness score
        scores['freshness'] = self._score_freshness(document.published_date)
        
        # Relevance score
        scores['relevance'] = self._score_relevance(document, summary, query, category)
        
        # User preference score
        scores['user_preference'] = self._score_user_preference(
            document, summary, user_profile
        )
        
        # Engagement signals (placeholder for now)
        scores['engagement_signals'] = 0.5
        
        # Calculate weighted final score
        final_score = sum(
            scores[factor] * self.weights[factor]
            for factor in scores
        )
        
        logger.debug(f"Scored {document.url}: {final_score:.3f} {scores}")
        return final_score
    
    def _score_content_quality(
        self,
        document: ContentDocument,
        summary: Optional[StructuredSummary] = None,
    ) -> float:
        """Score based on content quality indicators"""
        score = 0.5  # Base score
        
        # Length indicators
        if document.content_length > 1000:
            score += 0.2
        elif document.content_length > 500:
            score += 0.1
        
        # Title quality
        if document.title and len(document.title) > 10:
            score += 0.1
        
        # Author presence
        if document.author:
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
        return self.source_trust_scores.get(domain.lower(), 0.5)
    
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
    
    def _score_user_preference(
        self,
        document: ContentDocument,
        summary: Optional[StructuredSummary],
        user_profile: Optional['UserPreferences'],
    ) -> float:
        """Score based on user preferences"""
        if not user_profile or not self.embedding_model:
            return 0.5
        
        try:
            # Get document embedding
            doc_text = f"{document.title} {document.content[:1000]}"
            doc_embedding = self.embedding_model.encode([doc_text])
            
            # Compare with user profile embedding
            if user_profile.profile_embedding is not None:
                similarity = cosine_similarity(
                    doc_embedding,
                    [user_profile.profile_embedding]
                )[0][0]
                return max(0.0, min(1.0, (similarity + 1) / 2))  # Normalize to 0-1
            
        except Exception as e:
            logger.warning(f"User preference scoring failed: {e}")
        
        return 0.5
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text"""
        if not self.embedding_model:
            return None
        
        try:
            return self.embedding_model.encode([text])[0]
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None


class UserPreferences:
    """Manages user preferences and learning"""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.interests: List[str] = []
        self.liked_content: List[str] = []  # Content hashes
        self.disliked_content: List[str] = []
        self.profile_embedding: Optional[np.ndarray] = None
        self.feedback_history: List[Dict[str, Any]] = []
        self.learning_rate = settings.feedback_learning_rate
    
    def add_feedback(
        self,
        content_hash: str,
        feedback_type: str,
        feedback_value: float,
        content_embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Add user feedback and update preferences"""
        
        feedback = {
            'content_hash': content_hash,
            'feedback_type': feedback_type,
            'feedback_value': feedback_value,
            'timestamp': datetime.utcnow(),
        }
        
        self.feedback_history.append(feedback)
        
        # Update liked/disliked lists
        if feedback_value > 0.5:
            if content_hash not in self.liked_content:
                self.liked_content.append(content_hash)
            if content_hash in self.disliked_content:
                self.disliked_content.remove(content_hash)
        elif feedback_value < -0.5:
            if content_hash not in self.disliked_content:
                self.disliked_content.append(content_hash)
            if content_hash in self.liked_content:
                self.liked_content.remove(content_hash)
        
        # Update profile embedding if available
        if content_embedding is not None:
            self._update_profile_embedding(content_embedding, feedback_value)
        
        logger.info(f"Added feedback for user {self.user_id}: {feedback_type}={feedback_value}")
    
    def _update_profile_embedding(
        self,
        content_embedding: np.ndarray,
        feedback_value: float,
    ) -> None:
        """Update user profile embedding based on feedback"""
        
        if self.profile_embedding is None:
            # Initialize with first embedding
            self.profile_embedding = content_embedding.copy()
        else:
            # Update using exponential moving average
            weight = self.learning_rate * abs(feedback_value)
            if feedback_value > 0:
                # Move towards liked content
                self.profile_embedding = (
                    (1 - weight) * self.profile_embedding +
                    weight * content_embedding
                )
            else:
                # Move away from disliked content
                self.profile_embedding = (
                    (1 + weight) * self.profile_embedding -
                    weight * content_embedding
                )
        
        # Normalize the embedding
        norm = np.linalg.norm(self.profile_embedding)
        if norm > 0:
            self.profile_embedding = self.profile_embedding / norm
    
    def get_preference_score(self, content_embedding: np.ndarray) -> float:
        """Get preference score for content"""
        if self.profile_embedding is None:
            return 0.5
        
        try:
            similarity = cosine_similarity(
                [content_embedding],
                [self.profile_embedding]
            )[0][0]
            return max(0.0, min(1.0, (similarity + 1) / 2))
        except Exception:
            return 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'interests': self.interests,
            'liked_content_count': len(self.liked_content),
            'disliked_content_count': len(self.disliked_content),
            'feedback_count': len(self.feedback_history),
            'has_profile_embedding': self.profile_embedding is not None,
        }


class PersonalizationEngine:
    """Main personalization engine"""
    
    def __init__(self):
        self.scorer = ContentScorer()
        self.user_preferences: Dict[str, UserPreferences] = {}
        self.feedback_buffer: List[Dict[str, Any]] = []
    
    def get_user_preferences(self, user_id: str = "default") -> UserPreferences:
        """Get or create user preferences"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = UserPreferences(user_id)
        return self.user_preferences[user_id]
    
    def score_documents(
        self,
        documents: List[ContentDocument],
        summaries: Optional[List[StructuredSummary]] = None,
        query: str = "",
        category: str = "",
        user_id: str = "default",
    ) -> List[Tuple[ContentDocument, float]]:
        """Score and rank documents for a user"""
        
        user_prefs = self.get_user_preferences(user_id)
        
        scored_docs = []
        for i, doc in enumerate(documents):
            summary = summaries[i] if summaries and i < len(summaries) else None
            
            score = self.scorer.score_content(
                document=doc,
                summary=summary,
                user_profile=user_prefs,
                query=query,
                category=category,
            )
            
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
        
        user_prefs = self.get_user_preferences(user_id)
        
        # Get content embedding if possible
        content_embedding = None
        if content_text and self.scorer.embedding_model:
            content_embedding = self.scorer.get_embedding(content_text)
        
        user_prefs.add_feedback(
            content_hash=content_hash,
            feedback_type=feedback_type,
            feedback_value=feedback_value,
            content_embedding=content_embedding,
        )
        
        # Buffer for batch processing
        self.feedback_buffer.append({
            'user_id': user_id,
            'content_hash': content_hash,
            'feedback_type': feedback_type,
            'feedback_value': feedback_value,
            'timestamp': datetime.utcnow(),
        })
    
    def get_personalization_stats(self, user_id: str = "default") -> Dict[str, Any]:
        """Get personalization statistics for a user"""
        user_prefs = self.get_user_preferences(user_id)
        
        return {
            'user_preferences': user_prefs.to_dict(),
            'scorer_weights': self.scorer.weights,
            'source_trust_scores': len(self.scorer.source_trust_scores),
            'feedback_buffer_size': len(self.feedback_buffer),
        }


# Default instances
default_scorer = ContentScorer()
default_personalization = PersonalizationEngine()