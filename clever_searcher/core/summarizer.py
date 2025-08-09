"""LLM-powered content summarization with structured output"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from openai import OpenAI

from ..utils.config import settings
from .fetcher import ContentDocument

logger = logging.getLogger(__name__)


class StructuredSummary:
    """Represents a structured summary of content"""
    
    def __init__(
        self,
        tldr: str,
        bullets: List[str],
        tags: List[str],
        entities: List[str],
        key_points: Optional[List[str]] = None,
        sentiment: str = "neutral",
        complexity: str = "medium",
        read_time_minutes: int = 0,
        confidence_score: float = 0.8,
        model_used: str = "",
    ):
        self.tldr = tldr
        self.bullets = bullets or []
        self.tags = tags or []
        self.entities = entities or []
        self.key_points = key_points or []
        self.sentiment = sentiment
        self.complexity = complexity
        self.read_time_minutes = read_time_minutes
        self.confidence_score = confidence_score
        self.model_used = model_used
        self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tldr": self.tldr,
            "bullets": self.bullets,
            "tags": self.tags,
            "entities": self.entities,
            "key_points": self.key_points,
            "sentiment": self.sentiment,
            "complexity": self.complexity,
            "read_time_minutes": self.read_time_minutes,
            "confidence_score": self.confidence_score,
            "model_used": self.model_used,
            "created_at": self.created_at.isoformat(),
        }


class ContentSummarizer:
    """LLM-powered content summarizer with structured output"""
    
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or settings.model_summary
        self.base_url = base_url or settings.openai_base_url
        self.api_key = api_key or settings.openai_api_key
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "ollama",
        )
        
        # Category-specific prompts
        self.category_prompts = {
            "papers": "Focus on methodology, findings, and implications. Extract technical terms and research concepts.",
            "news": "Focus on who, what, when, where, why. Extract key people, organizations, and events.",
            "jobs": "Focus on requirements, responsibilities, and company details. Extract skills and technologies.",
            "crypto": "Focus on price movements, market analysis, and regulatory news. Extract coins, exchanges, and metrics.",
            "tech": "Focus on product features, company news, and industry trends. Extract technologies and companies.",
        }
    
    async def summarize(
        self,
        document: ContentDocument,
        query: str,
        max_bullets: int = 5,
        max_tags: int = 8,
        max_entities: int = 10,
    ) -> StructuredSummary:
        """Generate a structured summary of the document"""
        
        logger.info(f"Summarizing document: {document.title[:50]}...")
        
        # Prepare content for summarization
        content_preview = self._prepare_content(document.content)
        
        try:
            summary_data = await self._generate_summary(
                title=document.title,
                content=content_preview,
                url=document.url,
                query=query,
                max_bullets=max_bullets,
                max_tags=max_tags,
                max_entities=max_entities,
            )
            
            # Calculate read time
            read_time = self._estimate_read_time(document.content)
            
            summary = StructuredSummary(
                tldr=summary_data.get("tldr", ""),
                bullets=summary_data.get("bullets", [])[:max_bullets],
                tags=summary_data.get("tags", [])[:max_tags],
                entities=summary_data.get("entities", [])[:max_entities],
                key_points=summary_data.get("key_points", []),
                sentiment=summary_data.get("sentiment", "neutral"),
                complexity=summary_data.get("complexity", "medium"),
                read_time_minutes=read_time,
                confidence_score=summary_data.get("confidence_score", 0.8),
                model_used=self.model,
            )
            
            logger.info(f"Successfully summarized: {document.title[:50]}...")
            return summary
            
        except Exception as e:
            logger.error(f"Summarization failed for {document.url}: {e}")
            # Return fallback summary
            return self._create_fallback_summary(document, query)
    
    def _prepare_content(self, content: str, max_length: int = 8000) -> str:
        """Prepare content for summarization by truncating if needed"""
        if len(content) <= max_length:
            return content
        
        # Try to truncate at sentence boundaries
        truncated = content[:max_length]
        last_period = truncated.rfind('.')
        last_newline = truncated.rfind('\n')
        
        # Use the later boundary
        boundary = max(last_period, last_newline)
        if boundary > max_length * 0.8:  # If boundary is reasonably close to end
            return content[:boundary + 1]
        else:
            return content[:max_length] + "..."
    
    async def _generate_summary(
        self,
        title: str,
        content: str,
        url: str,
        query: str,
        max_bullets: int,
        max_tags: int,
        max_entities: int,
    ) -> Dict[str, Any]:
        """Generate summary using LLM"""
        
        system_prompt = f"""You are an expert content analyst. Your task is to create a structured summary of web content based on a user's query.

Analyze the content and generate a JSON response with these fields:
- tldr: A concise 1-2 sentence summary (max 200 characters) relevant to the user's query.
- bullets: An array of {max_bullets} key bullet points (each max 150 characters) that directly address the user's query.
- tags: An array of {max_tags} relevant tags/keywords.
- entities: An array of {max_entities} important people, organizations, or concepts.
- key_points: An array of 3-5 most important insights related to the query.
- sentiment: The sentiment of the content regarding the query ("positive", "negative", "neutral").
- complexity: The complexity of the content ("beginner", "intermediate", "advanced").
- confidence_score: A float (0.0-1.0) indicating your confidence in the summary's relevance to the query.

Guidelines:
- Prioritize information that directly answers or relates to the user's query.
- Be concise, factual, and use clear language.
- Ensure all arrays contain only strings.
- Tags should be lowercase, using underscores for multi-word tags."""

        user_prompt = f"""User Query: "{query}"

Title: {title}
URL: {url}

Content:
{content}

Please analyze this content in the context of the user's query and provide a structured summary in the specified JSON format."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=1500,
            )
            
            content_str = response.choices[0].message.content
            if not content_str:
                raise ValueError("Empty response from LLM")
            
            summary_data: Dict[str, Any] = json.loads(content_str)
            return summary_data
            
        except Exception as e:
            logger.error(f"LLM summarization failed: {e}")
            raise
    
    def _create_fallback_summary(self, document: ContentDocument, query: str) -> StructuredSummary:
        """Create a basic fallback summary when LLM fails"""
        
        # Extract first few sentences as TLDR
        sentences = document.content.split('. ')
        tldr = '. '.join(sentences[:2])[:200] + "..."
        
        # Basic bullet points from first paragraphs
        paragraphs = document.content.split('\n\n')[:3]
        bullets = [p.strip()[:150] + "..." for p in paragraphs if p.strip()]
        
        # Simple tags from title
        title_words = document.title.lower().split()
        tags = [word for word in title_words if len(word) > 3][:5]
        
        return StructuredSummary(
            tldr=tldr,
            bullets=bullets,
            tags=tags,
            entities=[],
            key_points=[],
            sentiment="neutral",
            complexity="medium",
            read_time_minutes=self._estimate_read_time(document.content),
            confidence_score=0.3,  # Low confidence for fallback
            model_used="fallback",
        )
    
    def _estimate_read_time(self, content: str) -> int:
        """Estimate reading time in minutes (average 200 words per minute)"""
        word_count = len(content.split())
        return max(1, round(word_count / 200))
    
    async def summarize_batch(
        self,
        documents: List[ContentDocument],
        query: str,
        max_concurrent: int = 3,
    ) -> List[StructuredSummary]:
        """Summarize multiple documents concurrently"""
        
        import asyncio
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def summarize_with_semaphore(doc: ContentDocument) -> StructuredSummary:
            async with semaphore:
                return await self.summarize(doc, query)
        
        logger.info(f"Summarizing {len(documents)} documents with max_concurrent={max_concurrent}")
        
        tasks = [summarize_with_semaphore(doc) for doc in documents]
        summaries = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return valid summaries
        valid_summaries = []
        for i, result in enumerate(summaries):
            if isinstance(result, StructuredSummary):
                valid_summaries.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Batch summarization error for document {i}: {result}")
                # Create fallback summary
                fallback = self._create_fallback_summary(documents[i], query)
                valid_summaries.append(fallback)
        
        logger.info(f"Successfully summarized {len(valid_summaries)} out of {len(documents)} documents")
        return valid_summaries


class SummaryRanker:
    """Ranks summaries by quality and relevance"""
    
    def __init__(self) -> None:
        self.quality_weights = {
            'confidence_score': 0.3,
            'content_length': 0.2,
            'entity_count': 0.15,
            'bullet_quality': 0.15,
            'tag_relevance': 0.1,
            'freshness': 0.1,
        }
    
    def rank_summaries(
        self,
        summaries: List[StructuredSummary],
        query: str = "",
        category: str = "",
    ) -> List[StructuredSummary]:
        """Rank summaries by quality and relevance"""
        
        scored_summaries = []
        
        for summary in summaries:
            score = self._calculate_quality_score(summary, query, category)
            scored_summaries.append((summary, score))
        
        # Sort by score (descending)
        scored_summaries.sort(key=lambda x: x[1], reverse=True)
        
        return [summary for summary, score in scored_summaries]
    
    def _calculate_quality_score(
        self,
        summary: StructuredSummary,
        query: str,
        category: str,
    ) -> float:
        """Calculate quality score for a summary"""
        
        score = 0.0
        
        # Confidence score
        score += summary.confidence_score * self.quality_weights['confidence_score']
        
        # Content completeness
        content_score = min(1.0, (len(summary.bullets) + len(summary.entities)) / 10)
        score += content_score * self.quality_weights['content_length']
        
        # Entity richness
        entity_score = min(1.0, len(summary.entities) / 5)
        score += entity_score * self.quality_weights['entity_count']
        
        # Bullet point quality (based on length and count)
        bullet_score = 0.0
        if summary.bullets:
            avg_bullet_length = sum(len(b) for b in summary.bullets) / len(summary.bullets)
            bullet_score = min(1.0, avg_bullet_length / 100)  # Normalize to 100 chars
        score += bullet_score * self.quality_weights['bullet_quality']
        
        # Tag relevance (simple keyword matching)
        tag_score = 0.0
        if query and summary.tags:
            query_words = set(query.lower().split())
            tag_words = set(' '.join(summary.tags).lower().split())
            overlap = len(query_words.intersection(tag_words))
            tag_score = min(1.0, overlap / max(1, len(query_words)))
        score += tag_score * self.quality_weights['tag_relevance']
        
        # Freshness (newer summaries get slight boost)
        hours_old = (datetime.utcnow() - summary.created_at).total_seconds() / 3600
        freshness_score = max(0.0, 1.0 - (hours_old / 168))  # Decay over a week
        score += freshness_score * self.quality_weights['freshness']
        
        return score


# Default instances
default_summarizer = ContentSummarizer()
default_ranker = SummaryRanker()