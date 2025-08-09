"""Preference data collection for GRPO training dataset"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
import uuid

from ..utils.config import get_data_dir

logger = logging.getLogger(__name__)


@dataclass
class PreferenceData:
    """Data structure for preference learning dataset"""
    
    # Core identifiers (required)
    session_id: str
    timestamp: datetime
    original_query: str
    llm_category: str
    llm_queries: List[str]
    llm_preferred_sites: List[str]
    llm_keywords: List[str]
    llm_avoid_keywords: List[str]
    actual_domains_searched: List[str]
    search_results_count: int
    unique_documents_found: int
    top_scoring_domains: List[str]
    
    # Optional LLM fields
    llm_max_pages: Optional[int] = None
    llm_freshness_days: Optional[int] = None
    llm_include_news: Optional[bool] = None
    llm_time_range: Optional[str] = None
    llm_reasoning: Optional[str] = None
    
    # Optional fields (with defaults)
    user_intent: Optional[str] = None
    preferred_category: Optional[str] = None
    preferred_queries: Optional[List[str]] = None
    preferred_domains: Optional[List[str]] = None
    preferred_keywords: Optional[List[str]] = None
    preferred_avoid_keywords: Optional[List[str]] = None
    preferred_max_pages: Optional[int] = None
    preferred_freshness_days: Optional[int] = None
    preferred_include_news: Optional[bool] = None
    preferred_time_range: Optional[str] = None
    domain_rankings: Optional[Dict[str, float]] = None  # domain -> preference score
    query_effectiveness: Optional[Dict[str, float]] = None  # query -> effectiveness score
    overall_satisfaction: Optional[float] = None  # 1-5 scale
    execution_metadata: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreferenceData":
        """Create from dictionary"""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class PreferenceCollector:
    """Collector for preference learning dataset"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (get_data_dir() / "preferences")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Files for different data types
        self.raw_data_file = self.data_dir / "raw_preference_data.jsonl"
        self.training_data_file = self.data_dir / "training_dataset.jsonl"
        self.human_feedback_file = self.data_dir / "human_feedback.jsonl"
    
    def collect_llm_decision_data(
        self,
        session_id: str,
        original_query: str,
        llm_plan: Dict[str, Any],
        search_results: Dict[str, Any],
        execution_metadata: Optional[Dict[str, Any]] = None
    ) -> PreferenceData:
        """Collect LLM decision data for preference learning"""
        
        preference_data = PreferenceData(
            session_id=session_id,
            timestamp=datetime.utcnow(),
            original_query=original_query,
            
            # Extract from LLM plan
            llm_category=llm_plan.get("category", "unknown"),
            llm_queries=llm_plan.get("queries", []),
            llm_preferred_sites=llm_plan.get("preferred_sites", []),
            llm_keywords=llm_plan.get("must_have_keywords", []),
            llm_avoid_keywords=llm_plan.get("avoid_keywords", []),
            llm_reasoning=llm_plan.get("reasoning"),
            
            # Extract from search results
            actual_domains_searched=search_results.get("domains_searched", []),
            search_results_count=search_results.get("total_results", 0),
            unique_documents_found=search_results.get("unique_documents", 0),
            top_scoring_domains=search_results.get("top_domains", []),
            
            execution_metadata=execution_metadata,
        )
        
        # Save raw data
        self._save_raw_data(preference_data)
        
        logger.info(f"Collected preference data for session {session_id}")
        return preference_data
    
    def add_human_feedback(
        self,
        session_id: str,
        preferred_category: Optional[str] = None,
        preferred_queries: Optional[List[str]] = None,
        preferred_domains: Optional[List[str]] = None,
        preferred_keywords: Optional[List[str]] = None,
        preferred_avoid_keywords: Optional[List[str]] = None,
        preferred_max_pages: Optional[int] = None,
        preferred_freshness_days: Optional[int] = None,
        preferred_include_news: Optional[bool] = None,
        domain_rankings: Optional[Dict[str, float]] = None,
        query_effectiveness: Optional[Dict[str, float]] = None,
        overall_satisfaction: Optional[float] = None,
        notes: Optional[str] = None
    ) -> bool:
        """Add human feedback to existing preference data"""
        
        # Load existing data
        preference_data = self._load_session_data(session_id)
        if not preference_data:
            logger.error(f"No preference data found for session {session_id}")
            return False
        
        # Update with human feedback
        preference_data.preferred_category = preferred_category
        preference_data.preferred_queries = preferred_queries
        preference_data.preferred_domains = preferred_domains
        preference_data.preferred_keywords = preferred_keywords
        preference_data.preferred_avoid_keywords = preferred_avoid_keywords
        preference_data.preferred_max_pages = preferred_max_pages
        preference_data.preferred_freshness_days = preferred_freshness_days
        preference_data.preferred_include_news = preferred_include_news
        preference_data.domain_rankings = domain_rankings
        preference_data.query_effectiveness = query_effectiveness
        preference_data.overall_satisfaction = overall_satisfaction
        preference_data.notes = notes
        
        # Save updated data
        self._save_human_feedback(preference_data)
        self._update_training_dataset(preference_data)
        
        logger.info(f"Added human feedback for session {session_id}")
        return True
    
    def create_training_example(self, preference_data: PreferenceData) -> Dict[str, Any]:
        """Create a training example for GRPO from preference data"""
        
        # Base prompt for the LLM
        prompt = f"""Given the user query: "{preference_data.original_query}"
        
Please generate a search plan including:
1. Category classification
2. Search queries to use
3. Preferred domains/sites
4. Keywords to include/avoid
"""
        
        # LLM's original response
        llm_response = {
            "category": preference_data.llm_category,
            "queries": preference_data.llm_queries,
            "preferred_sites": preference_data.llm_preferred_sites,
            "keywords": preference_data.llm_keywords,
            "avoid_keywords": preference_data.llm_avoid_keywords,
        }
        
        # Human preferred response (if available)
        human_response = None
        if preference_data.preferred_category or preference_data.preferred_queries:
            human_response = {
                "category": preference_data.preferred_category or preference_data.llm_category,
                "queries": preference_data.preferred_queries or preference_data.llm_queries,
                "preferred_sites": preference_data.preferred_domains or preference_data.llm_preferred_sites,
                "keywords": preference_data.llm_keywords,  # Keep original if not specified
                "avoid_keywords": preference_data.llm_avoid_keywords,
            }
        
        training_example = {
            "session_id": preference_data.session_id,
            "timestamp": preference_data.timestamp.isoformat(),
            "prompt": prompt,
            "llm_response": llm_response,
            "human_response": human_response,
            "preference_scores": {
                "domain_rankings": preference_data.domain_rankings,
                "query_effectiveness": preference_data.query_effectiveness,
                "overall_satisfaction": preference_data.overall_satisfaction,
            },
            "execution_results": {
                "domains_searched": preference_data.actual_domains_searched,
                "results_count": preference_data.search_results_count,
                "unique_documents": preference_data.unique_documents_found,
                "top_domains": preference_data.top_scoring_domains,
            }
        }
        
        return training_example
    
    def export_training_dataset(self, output_file: Optional[Path] = None) -> Path:
        """Export complete training dataset for GRPO"""
        
        output_file = output_file or (self.data_dir / f"grpo_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        
        # Load all preference data with human feedback
        training_examples = []
        
        # Check both raw data file and human feedback file
        data_sources = []
        if self.raw_data_file.exists():
            data_sources.append(self.raw_data_file)
        if self.human_feedback_file.exists():
            data_sources.append(self.human_feedback_file)
        
        processed_sessions = set()
        
        for data_file in data_sources:
            with open(data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        preference_data = PreferenceData.from_dict(data)
                        
                        # Skip if already processed (avoid duplicates)
                        if preference_data.session_id in processed_sessions:
                            continue
                        
                        # Only include if we have human feedback
                        if (preference_data.preferred_category or
                            preference_data.preferred_queries or
                            preference_data.overall_satisfaction is not None):
                            
                            training_example = self.create_training_example(preference_data)
                            training_examples.append(training_example)
                            processed_sessions.add(preference_data.session_id)
                            
                    except Exception as e:
                        logger.warning(f"Failed to process preference data: {e}")
        
        # Save training dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in training_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Exported {len(training_examples)} training examples to {output_file}")
        return output_file
    
    def get_pending_feedback_sessions(self) -> List[Dict[str, Any]]:
        """Get sessions that need human feedback"""
        
        pending_sessions: List[Dict[str, Any]] = []
        
        if not self.raw_data_file.exists():
            return pending_sessions
        
        # Get sessions with feedback to exclude them
        sessions_with_feedback = set()
        if self.human_feedback_file.exists():
            with open(self.human_feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        sessions_with_feedback.add(data.get("session_id"))
                    except Exception as e:
                        logger.warning(f"Failed to process feedback data: {e}")
        
        # Load raw data and check for missing feedback
        with open(self.raw_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    preference_data = PreferenceData.from_dict(data)
                    
                    # Check if feedback is missing (not in feedback file)
                    if preference_data.session_id not in sessions_with_feedback:
                        pending_sessions.append({
                            "session_id": preference_data.session_id,
                            "timestamp": preference_data.timestamp.isoformat(),
                            "original_query": preference_data.original_query,
                            "llm_category": preference_data.llm_category,
                            "llm_queries": preference_data.llm_queries,
                            "results_count": preference_data.search_results_count,
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to process raw data: {e}")
        
        return pending_sessions
    
    def _save_raw_data(self, preference_data: PreferenceData) -> None:
        """Save raw preference data"""
        with open(self.raw_data_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(preference_data.to_dict(), ensure_ascii=False) + '\n')
    
    def _save_human_feedback(self, preference_data: PreferenceData) -> None:
        """Save preference data with human feedback"""
        with open(self.human_feedback_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(preference_data.to_dict(), ensure_ascii=False) + '\n')
    
    def _update_training_dataset(self, preference_data: PreferenceData) -> None:
        """Update training dataset with new feedback"""
        training_example = self.create_training_example(preference_data)
        
        with open(self.training_data_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
    
    def _load_session_data(self, session_id: str) -> Optional[PreferenceData]:
        """Load preference data for a specific session"""
        
        if not self.raw_data_file.exists():
            return None
        
        with open(self.raw_data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("session_id") == session_id:
                        return PreferenceData.from_dict(data)
                except Exception as e:
                    logger.warning(f"Failed to process raw data: {e}")
        
        return None
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the preference dataset"""
        
        stats: Dict[str, Any] = {
            "total_sessions": 0,
            "sessions_with_feedback": 0,
            "pending_feedback": 0,
            "avg_satisfaction": 0.0,
            "common_categories": {},
            "common_domains": {},
        }
        
        satisfaction_scores: List[float] = []
        categories: Dict[str, int] = {}
        domains: Dict[str, int] = {}
        
        # Process raw data
        if self.raw_data_file.exists():
            with open(self.raw_data_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        stats["total_sessions"] += 1
                        
                        # Count categories
                        category = data.get("llm_category", "unknown")
                        categories[category] = categories.get(category, 0) + 1
                        
                        # Count domains
                        for domain in data.get("top_scoring_domains", []):
                            domains[domain] = domains.get(domain, 0) + 1
                        
                        # Check for feedback
                        if (data.get("preferred_category") or 
                            data.get("preferred_queries") or 
                            data.get("overall_satisfaction") is not None):
                            stats["sessions_with_feedback"] += 1
                            
                            if data.get("overall_satisfaction") is not None:
                                satisfaction_scores.append(data["overall_satisfaction"])
                        else:
                            stats["pending_feedback"] += 1
                            
                    except Exception as e:
                        logger.warning(f"Failed to process stats data: {e}")
        
        # Calculate averages and top items
        if satisfaction_scores:
            stats["avg_satisfaction"] = sum(satisfaction_scores) / len(satisfaction_scores)
        
        stats["common_categories"] = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10])
        stats["common_domains"] = dict(sorted(domains.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return stats


# Global preference collector instance
preference_collector = PreferenceCollector()