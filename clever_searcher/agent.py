"""Main agent orchestrator that coordinates all components"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path

from .core.planner import default_planner, CrawlPlan
from .core.searcher import default_searcher, SearchResult
from .core.fetcher import default_fetcher, ContentDocument
from .core.deduper import default_deduplicator, default_tracker
from .core.summarizer import default_summarizer, StructuredSummary
from .core.scorer import default_personalization, PersonalizationEngine
from .core.simple_scorer import simple_personalization, SimplePersonalizationEngine
from .output.digest import default_manager, DigestItem
from .storage.database import get_db_session
from .storage.models import CrawlRun, Page, Summary
from .utils.config import settings
from .logging.operation_logger import operation_logger, LogLevel
from .logging.preference_collector import preference_collector

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
    
    # Type annotations for attributes
    personalization: Union[PersonalizationEngine, SimplePersonalizationEngine]
    
    def __init__(self, search_engine: Optional[str] = None):
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
        query: str,
        max_pages: Optional[int] = None,
        max_queries: int = 6,
        custom_sites: Optional[List[str]] = None,
        output_format: str = "markdown",
        user_id: str = "default",
        save_to_db: bool = True,
    ) -> DiscoveryResult:
        """Run a complete discovery cycle"""
        
        # Start operation logging
        operation_id = operation_logger.start_operation(
            user_query=query,
            metadata={
                "max_pages": max_pages,
                "max_queries": max_queries,
                "custom_sites": custom_sites,
                "output_format": output_format,
                "user_id": user_id,
                "save_to_db": save_to_db,
            }
        )
        
        try:
            logger.info(f"Starting discovery for query: {query}")
            start_time = datetime.utcnow()
            
            # Reset state
            self.reset_state()
            
            # Step 1: Create crawl plan
            step_start = operation_logger.log_step_start(
                component="planner",
                operation="create_plan",
                message="Creating crawl plan",
                data={"query": query, "max_queries": max_queries}
            )
            
            plan = await self.planner.create_plan(
                query=query,
                max_queries=max_queries,
                max_pages=max_pages or 10,
                custom_sites=custom_sites or [],
            )
            
            operation_logger.log_step_end(
                component="planner",
                operation="create_plan",
                message="Crawl plan created",
                start_time=step_start,
                data={
                    "category": plan.category,
                    "queries": plan.queries,
                    "preferred_sites": plan.preferred_sites,
                    "max_pages": plan.max_pages,
                }
            )
        
            # Step 2: Search for content
            step_start = operation_logger.log_step_start(
                component="searcher",
                operation="search_category",
                message="Searching for content",
                data={
                    "category": plan.category,
                    "queries": plan.queries,
                    "max_results": plan.max_pages,
                }
            )
            
            search_results = await self.searcher.search_category(
                category=plan.category,
                queries=plan.queries,
                max_results=plan.max_pages,
                site_preferences={plan.category: plan.preferred_sites} if plan.preferred_sites else {},
            )
            
            operation_logger.log_step_end(
                component="searcher",
                operation="search_category",
                message="Content search completed",
                start_time=step_start,
                data={
                    "results_found": len(search_results),
                    "domains": list(set(result.url.split('/')[2] for result in search_results if '/' in result.url)),
                }
            )
        
            # Step 3: Fetch and extract content
            step_start = operation_logger.log_step_start(
                component="fetcher",
                operation="fetch_batch",
                message="Fetching content",
                data={"url_count": len(search_results)}
            )
            
            urls = [result.url for result in search_results]
            documents = await self.fetcher.fetch_batch(
                urls=urls,
                max_concurrent=5,
            )
            
            operation_logger.log_step_end(
                component="fetcher",
                operation="fetch_batch",
                message="Content fetching completed",
                start_time=step_start,
                data={
                    "documents_fetched": len(documents),
                    "success_rate": len(documents) / len(urls) if urls else 0,
                }
            )
        
            # Step 4: Deduplicate content
            step_start = operation_logger.log_step_start(
                component="deduplicator",
                operation="deduplicate",
                message="Deduplicating content",
                data={"input_documents": len(documents)}
            )
            
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
            
            dedup_stats = default_tracker.get_report()
            operation_logger.log_step_end(
                component="deduplicator",
                operation="deduplicate",
                message="Content deduplication completed",
                start_time=step_start,
                data={
                    "unique_documents": len(unique_documents),
                    "duplicates_removed": len(documents) - len(unique_documents),
                    "deduplication_stats": dedup_stats,
                }
            )
        
            # Step 5: Generate summaries
            step_start = operation_logger.log_step_start(
                component="summarizer",
                operation="summarize_batch",
                message="Generating summaries",
                data={"document_count": len(unique_documents)}
            )
            
            summaries = await self.summarizer.summarize_batch(
                documents=unique_documents,
                query=query,
                max_concurrent=3,
            )
            
            operation_logger.log_step_end(
                component="summarizer",
                operation="summarize_batch",
                message="Summary generation completed",
                start_time=step_start,
                data={
                    "summaries_generated": len(summaries),
                    "avg_read_time": sum(s.read_time_minutes for s in summaries) / len(summaries) if summaries else 0,
                }
            )
        
            # Step 6: Score and rank content
            step_start = operation_logger.log_step_start(
                component="personalization",
                operation="score_documents",
                message="Scoring and ranking content",
                data={"document_count": len(unique_documents)}
            )
            
            scored_docs = self.personalization.score_documents(
                documents=unique_documents,
                summaries=summaries,
                query=query,
                category=plan.category,
                user_id=user_id,
            )
            
            # Sort by score and extract components
            final_documents = [doc for doc, score in scored_docs]
            final_scores = [score for doc, score in scored_docs]
            
            # Ensure summaries match the reordered documents
            doc_to_summary = {doc.content_hash: summary for doc, summary in zip(unique_documents, summaries)}
            final_summaries = [s for s in (doc_to_summary.get(doc.content_hash) for doc in final_documents) if s is not None]
            
            operation_logger.log_step_end(
                component="personalization",
                operation="score_documents",
                message="Content scoring completed",
                start_time=step_start,
                data={
                    "final_documents": len(final_documents),
                    "avg_score": sum(final_scores) / len(final_scores) if final_scores else 0,
                    "score_range": [min(final_scores), max(final_scores)] if final_scores else [0, 0],
                }
            )
        
            # Step 7: Generate digest
            step_start = operation_logger.log_step_start(
                component="digest_manager",
                operation="create_digest",
                message="Generating digest",
                data={"format": output_format}
            )
            
            digest_path = self.digest_manager.create_digest_from_results(
                documents=final_documents,
                summaries=final_summaries,
                scores=final_scores,
                title=f"{plan.category.title()} Discovery",
                category=plan.category,
                query=query,
                format_type=output_format,
            )
            
            operation_logger.log_step_end(
                component="digest_manager",
                operation="create_digest",
                message="Digest generation completed",
                start_time=step_start,
                data={"digest_path": str(digest_path) if digest_path else None}
            )
        
            # Step 8: Save to database (optional)
            if save_to_db:
                step_start = operation_logger.log_step_start(
                    component="database",
                    operation="save_results",
                    message="Saving to database"
                )
                
                await self._save_to_database(
                    plan=plan,
                    documents=final_documents,
                    summaries=final_summaries,
                    scores=final_scores,
                    start_time=start_time,
                )
                
                operation_logger.log_step_end(
                    component="database",
                    operation="save_results",
                    message="Database save completed",
                    start_time=step_start
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
            
            # Collect preference data for GRPO training
            domains_searched = list(set(result.url.split('/')[2] for result in search_results if '/' in result.url))
            
            # Enhanced LLM plan data with additional fields
            enhanced_plan_data = plan.to_dict()
            enhanced_plan_data.update({
                "llm_max_pages": plan.max_pages,
                "llm_freshness_days": plan.freshness_days,
                "llm_include_news": plan.include_news,
                "llm_time_range": plan.time_range,
                "llm_reasoning": getattr(plan, 'reasoning', None),
            })
            
            preference_collector.collect_llm_decision_data(
                session_id=operation_id,
                original_query=query,
                llm_plan=enhanced_plan_data,
                search_results={
                    "domains_searched": domains_searched,
                    "total_results": len(search_results),
                    "unique_documents": len(unique_documents),
                    "top_domains": self._get_top_domains(final_documents),
                },
                execution_metadata=stats
            )
            
            # Complete operation logging
            operation_logger.complete_operation(
                status="completed",
                final_data={
                    "stats": stats,
                    "operation_id": operation_id,
                }
            )
            
            logger.info(f"Discovery completed in {duration:.1f} seconds")
            
            return DiscoveryResult(
                documents=final_documents,
                summaries=final_summaries,
                scores=final_scores,
                plan=plan,
                stats=stats,
                digest_path=digest_path,
            )
            
        except Exception as e:
            # Log the error and fail the operation
            operation_logger.fail_operation(
                error_message=str(e),
                error_data={"exception_type": type(e).__name__}
            )
            raise
    
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
        domain_counts: Dict[str, int] = {}
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
                query="general search",
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