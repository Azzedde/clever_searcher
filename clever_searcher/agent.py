"""Main agent orchestrator that coordinates all components"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .core.planner import default_planner, CrawlPlan
from .core.searcher import default_searcher, SearchResult
from .core.fetcher import default_fetcher, ContentDocument
from .core.deduper import default_deduplicator, default_tracker
from .core.summarizer import default_summarizer, StructuredSummary
from .core.scorer import default_personalization
from .core.simple_scorer import simple_personalization
from .output.digest import default_manager, DigestItem
from .storage.database import get_db_session
from .storage.models import CrawlRun, Page, Summary
from .utils.config import settings

logger = logging.getLogger(__name__)


class DiscoveryResult:
    """Results from a discovery run"""
    
    def __init__(
        self,
        documents: List[ContentDocument],
        summaries: List[StructuredSummary],
        scores: List[float],
        plan: CrawlPlan,
        stats: Dict[str, Any],
        digest_path: Optional[Path] = None,
    ):
        self.documents = documents
        self.summaries = summaries
        self.scores = scores
        self.plan = plan
        self.stats = stats
        self.digest_path = digest_path
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'document_count': len(self.documents),
            'summary_count': len(self.summaries),
            'plan': self.plan.to_dict(),
            'stats': self.stats,
            'digest_path': str(self.digest_path) if self.digest_path else None,
            'timestamp': self.timestamp.isoformat(),
        }


class CleverSearcherAgent:
    """Main agent that orchestrates the discovery process"""
    
    def __init__(self, search_engine: str = None):
        self.planner = default_planner
        
        # Initialize searcher with specified engine
        from .core.searcher import MultiSearcher
        if search_engine:
            self.searcher = MultiSearcher(default_engine=search_engine)
        else:
            self.searcher = default_searcher
        
        self.fetcher = default_fetcher
        self.deduplicator = default_deduplicator
        self.summarizer = default_summarizer
        
        # Try to use full personalization, fallback to simple
        try:
            # Test if embeddings work
            test_scorer = default_personalization.scorer
            if test_scorer.embedding_model is None:
                logger.warning("Embedding model failed to load, using simple scorer")
                self.personalization = simple_personalization
            else:
                self.personalization = default_personalization
        except Exception as e:
            logger.warning(f"Failed to initialize full personalization: {e}, using simple scorer")
            self.personalization = simple_personalization
        
        self.digest_manager = default_manager
        
        # Reset deduplicator for each run
        self.reset_state()
    
    def reset_state(self) -> None:
        """Reset agent state for a new discovery run"""
        self.deduplicator.reset()
        default_tracker.duplicates.clear()
        default_tracker.stats = {
            'total_checked': 0,
            'url_duplicates': 0,
            'content_hash_duplicates': 0,
            'similarity_duplicates': 0,
            'unique_content': 0,
        }
    
    async def discover(
        self,
        category: str,
        user_query: str = "",
        max_pages: int = None,
        max_queries: int = 6,
        custom_sites: List[str] = None,
        output_format: str = "markdown",
        user_id: str = "default",
        save_to_db: bool = True,
    ) -> DiscoveryResult:
        """Run a complete discovery cycle"""
        
        logger.info(f"Starting discovery for category: {category}")
        start_time = datetime.utcnow()
        
        # Reset state
        self.reset_state()
        
        # Step 1: Create crawl plan
        logger.info("Step 1: Creating crawl plan...")
        plan = await self.planner.create_plan(
            category=category,
            user_query=user_query,
            max_queries=max_queries,
            max_pages=max_pages,
            custom_sites=custom_sites,
        )
        
        # Step 2: Search for content
        logger.info("Step 2: Searching for content...")
        search_results = await self.searcher.search_category(
            category=plan.category,
            queries=plan.queries,
            max_results=plan.max_pages,
            site_preferences={plan.category: plan.preferred_sites} if plan.preferred_sites else None,
        )
        
        logger.info(f"Found {len(search_results)} search results")
        
        # Step 3: Fetch and extract content
        logger.info("Step 3: Fetching content...")
        urls = [result.url for result in search_results]
        documents = await self.fetcher.fetch_batch(
            urls=urls,
            max_concurrent=5,
        )
        
        logger.info(f"Fetched {len(documents)} documents")
        
        # Step 4: Deduplicate content
        logger.info("Step 4: Deduplicating content...")
        unique_documents = []
        
        for doc in documents:
            dup_result = self.deduplicator.check_duplicate(
                url=doc.url,
                title=doc.title,
                content=doc.content,
                content_id=doc.content_hash,
            )
            
            default_tracker.record_duplicate(dup_result)
            
            if not dup_result['is_duplicate']:
                unique_documents.append(doc)
        
        logger.info(f"After deduplication: {len(unique_documents)} unique documents")
        
        # Step 5: Generate summaries
        logger.info("Step 5: Generating summaries...")
        summaries = await self.summarizer.summarize_batch(
            documents=unique_documents,
            category=category,
            max_concurrent=3,
        )
        
        logger.info(f"Generated {len(summaries)} summaries")
        
        # Step 6: Score and rank content
        logger.info("Step 6: Scoring and ranking content...")
        scored_docs = self.personalization.score_documents(
            documents=unique_documents,
            summaries=summaries,
            query=user_query,
            category=category,
            user_id=user_id,
        )
        
        # Sort by score and extract components
        final_documents = [doc for doc, score in scored_docs]
        final_scores = [score for doc, score in scored_docs]
        
        # Ensure summaries match the reordered documents
        doc_to_summary = {doc.content_hash: summary for doc, summary in zip(unique_documents, summaries)}
        final_summaries = [doc_to_summary.get(doc.content_hash) for doc in final_documents]
        
        logger.info(f"Final ranking: {len(final_documents)} documents")
        
        # Step 7: Generate digest
        logger.info("Step 7: Generating digest...")
        digest_path = self.digest_manager.create_digest_from_results(
            documents=final_documents,
            summaries=final_summaries,
            scores=final_scores,
            title=f"{category.title()} Discovery",
            category=category,
            query=user_query,
            format_type=output_format,
        )
        
        # Step 8: Save to database (optional)
        if save_to_db:
            logger.info("Step 8: Saving to database...")
            await self._save_to_database(
                plan=plan,
                documents=final_documents,
                summaries=final_summaries,
                scores=final_scores,
                start_time=start_time,
            )
        
        # Compile statistics
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        stats = {
            'duration_seconds': duration,
            'search_results_found': len(search_results),
            'documents_fetched': len(documents),
            'unique_documents': len(unique_documents),
            'summaries_generated': len(summaries),
            'final_results': len(final_documents),
            'deduplication_stats': default_tracker.get_report(),
            'avg_score': sum(final_scores) / len(final_scores) if final_scores else 0,
            'top_domains': self._get_top_domains(final_documents),
        }
        
        logger.info(f"Discovery completed in {duration:.1f} seconds")
        
        return DiscoveryResult(
            documents=final_documents,
            summaries=final_summaries,
            scores=final_scores,
            plan=plan,
            stats=stats,
            digest_path=digest_path,
        )
    
    async def _save_to_database(
        self,
        plan: CrawlPlan,
        documents: List[ContentDocument],
        summaries: List[StructuredSummary],
        scores: List[float],
        start_time: datetime,
    ) -> None:
        """Save results to database"""
        
        try:
            with get_db_session() as session:
                # Create crawl run record
                crawl_run = CrawlRun(
                    category=plan.category,
                    query=' | '.join(plan.queries),
                    pages_discovered=len(documents),
                    pages_processed=len(documents),
                    pages_new=len(documents),  # All are new in this simple implementation
                    pages_duplicate=0,
                    avg_score=sum(scores) / len(scores) if scores else 0,
                    duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
                    status='completed',
                    config=plan.to_dict(),
                    started_at=start_time,
                    completed_at=datetime.utcnow(),
                )
                session.add(crawl_run)
                session.flush()  # Get the ID
                
                # Save pages and summaries
                for i, (doc, summary, score) in enumerate(zip(documents, summaries, scores)):
                    # Create page record
                    page = Page(
                        canonical_url=doc.url,
                        original_url=doc.url,
                        domain=doc.domain,
                        title=doc.title,
                        author=doc.author,
                        published_at=doc.published_date,
                        content_hash=doc.content_hash,
                        content_length=doc.content_length,
                    )
                    session.add(page)
                    session.flush()  # Get the page ID
                    
                    # Create summary record
                    if summary:
                        summary_record = Summary(
                            page_id=page.id,
                            category=plan.category,
                            tldr=summary.tldr,
                            bullets=summary.bullets,
                            tags=summary.tags,
                            entities=summary.entities,
                            read_time_sec=summary.read_time_minutes * 60,
                            score=score,
                            model_used=summary.model_used,
                        )
                        session.add(summary_record)
                
                session.commit()
                logger.info(f"Saved {len(documents)} documents to database")
                
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
    
    def _get_top_domains(self, documents: List[ContentDocument], top_n: int = 5) -> List[str]:
        """Get top domains from documents"""
        domain_counts = {}
        for doc in documents:
            domain_counts[doc.domain] = domain_counts.get(doc.domain, 0) + 1
        
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        return [domain for domain, count in sorted_domains[:top_n]]
    
    async def quick_search(
        self,
        query: str,
        max_results: int = 10,
        include_summaries: bool = False,
    ) -> List[Tuple[ContentDocument, Optional[StructuredSummary], float]]:
        """Quick search without full discovery pipeline"""
        
        logger.info(f"Quick search: {query}")
        
        # Search
        search_results = await self.searcher.search(
            query=query,
            max_total_results=max_results,
        )
        
        # Fetch
        urls = [result.url for result in search_results]
        documents = await self.fetcher.fetch_batch(urls, max_concurrent=3)
        
        # Optional summaries
        summaries = None
        if include_summaries:
            summaries = await self.summarizer.summarize_batch(
                documents=documents,
                category="general",
                max_concurrent=2,
            )
        
        # Score
        scored_docs = self.personalization.score_documents(
            documents=documents,
            summaries=summaries,
            query=query,
        )
        
        # Combine results
        results = []
        for i, (doc, score) in enumerate(scored_docs):
            summary = summaries[i] if summaries and i < len(summaries) else None
            results.append((doc, summary, score))
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        return {
            'deduplication_stats': self.deduplicator.get_stats(),
            'personalization_stats': self.personalization.get_personalization_stats(),
            'digest_history': len(self.digest_manager.digest_history),
        }


# Default agent instance
default_agent = CleverSearcherAgent()