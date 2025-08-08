"""Search component with DuckDuckGo and Tavily integration"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from urllib.parse import urlparse

from duckduckgo_search import DDGS
import httpx

from ..utils.config import settings

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a single search result"""
    
    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        source: str = "",
        published_date: Optional[datetime] = None,
        score: float = 1.0,
    ):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.published_date = published_date
        self.score = score
        self.domain = urlparse(url).netloc
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "domain": self.domain,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "score": self.score,
        }


class DuckDuckGoSearcher:
    """DuckDuckGo search implementation"""
    
    def __init__(self, max_results: int = 50, timeout: int = 30):
        self.max_results = max_results
        self.timeout = timeout
        self.ddgs = DDGS(timeout=timeout)
    
    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        region: str = "us-en",
        safesearch: str = "moderate",
        time_range: Optional[str] = None,  # d, w, m, y for day, week, month, year
        site_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search DuckDuckGo for the given query
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            region: Search region (e.g., 'us-en', 'uk-en')
            safesearch: Safe search setting ('strict', 'moderate', 'off')
            time_range: Time range filter ('d', 'w', 'm', 'y')
            site_filter: Limit search to specific site (e.g., 'arxiv.org')
        """
        if max_results is None:
            max_results = self.max_results
        
        # Build the search query with filters
        search_query = query
        if site_filter:
            search_query = f"site:{site_filter} {query}"
        
        logger.info(f"Searching DuckDuckGo: '{search_query}' (max_results={max_results})")
        
        try:
            # Add delay to avoid rate limiting
            await asyncio.sleep(2)
            
            # Run the search in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._search_sync,
                search_query,
                max_results,
                region,
                safesearch,
                time_range,
            )
            
            logger.info(f"Found {len(results)} results for query: '{search_query}'")
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed for query '{search_query}': {e}")
            # Return empty results instead of failing
            return []
    
    def _search_sync(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
        time_range: Optional[str],
    ) -> List[SearchResult]:
        """Synchronous search implementation"""
        results = []
        
        try:
            # Use DuckDuckGo text search
            ddg_results = self.ddgs.text(
                keywords=query,
                region=region,
                safesearch=safesearch,
                timelimit=time_range,
                max_results=max_results,
            )
            
            for result in ddg_results:
                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("href", ""),
                    snippet=result.get("body", ""),
                    source=urlparse(result.get("href", "")).netloc,
                )
                results.append(search_result)
                
        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {e}")
        
        return results
    
    async def search_news(
        self,
        query: str,
        max_results: Optional[int] = None,
        region: str = "us-en",
        safesearch: str = "moderate",
        time_range: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search DuckDuckGo News"""
        if max_results is None:
            max_results = self.max_results
        
        logger.info(f"Searching DuckDuckGo News: '{query}' (max_results={max_results})")
        
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._search_news_sync,
                query,
                max_results,
                region,
                safesearch,
                time_range,
            )
            
            logger.info(f"Found {len(results)} news results for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo news search failed for query '{query}': {e}")
            return []
    
    def _search_news_sync(
        self,
        query: str,
        max_results: int,
        region: str,
        safesearch: str,
        time_range: Optional[str],
    ) -> List[SearchResult]:
        """Synchronous news search implementation"""
        results = []
        
        try:
            ddg_results = self.ddgs.news(
                keywords=query,
                region=region,
                safesearch=safesearch,
                timelimit=time_range,
                max_results=max_results,
            )
            
            for result in ddg_results:
                # Parse date if available
                published_date = None
                if "date" in result:
                    try:
                        # DuckDuckGo returns dates in various formats
                        date_str = result["date"]
                        # This is a simplified parser - you might want to use dateutil
                        published_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    except Exception:
                        pass
                
                search_result = SearchResult(
                    title=result.get("title", ""),
                    url=result.get("url", ""),
                    snippet=result.get("body", ""),
                    source=result.get("source", ""),
                    published_date=published_date,
                )
                results.append(search_result)
                
        except Exception as e:
            logger.error(f"Error in DuckDuckGo news search: {e}")
        
        return results


class TavilySearcher:
    """Tavily search implementation"""
    
    def __init__(self, api_key: str = None, max_results: int = 50):
        self.api_key = api_key or settings.tavily_api_key
        self.max_results = max_results
        
        if not self.api_key:
            logger.warning("No Tavily API key provided")
    
    async def search(
        self,
        query: str,
        max_results: Optional[int] = None,
        search_depth: str = "basic",  # basic, advanced
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Search using Tavily API"""
        
        if not self.api_key:
            logger.error("Tavily API key not configured")
            return []
        
        if max_results is None:
            max_results = self.max_results
        
        logger.info(f"Searching Tavily: '{query}' (max_results={max_results})")
        
        try:
            # Import tavily here to avoid dependency issues
            from tavily import TavilyClient
            
            client = TavilyClient(api_key=self.api_key)
            
            # Prepare search parameters
            search_params = {
                "query": query,
                "max_results": max_results,
                "search_depth": search_depth,
                "include_answer": False,
                "include_raw_content": False,
            }
            
            if include_domains:
                search_params["include_domains"] = include_domains
            if exclude_domains:
                search_params["exclude_domains"] = exclude_domains
            
            # Run search in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.search(**search_params)
            )
            
            results = []
            for item in response.get("results", []):
                # Parse published date if available
                published_date = None
                if "published_date" in item:
                    try:
                        published_date = datetime.fromisoformat(item["published_date"])
                    except Exception:
                        pass
                
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source=urlparse(item.get("url", "")).netloc,
                    published_date=published_date,
                    score=item.get("score", 1.0),
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} results from Tavily for query: '{query}'")
            return results
            
        except ImportError:
            logger.error("Tavily client not installed. Install with: pip install tavily-python")
            return []
        except Exception as e:
            logger.error(f"Tavily search failed for query '{query}': {e}")
            return []


class MultiSearcher:
    """Orchestrates multiple search sources"""
    
    def __init__(self, max_results_per_source: int = 25, default_engine: str = None):
        self.max_results_per_source = max_results_per_source
        self.default_engine = default_engine or settings.default_search_engine
        
        # Initialize searchers
        self.ddg_searcher = DuckDuckGoSearcher(max_results=max_results_per_source)
        self.tavily_searcher = TavilySearcher(max_results=max_results_per_source)
    
    async def search(
        self,
        query: str,
        sources: List[str] = None,
        max_total_results: int = 50,
        include_news: bool = False,
        site_filters: List[str] = None,
        time_range: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Search across multiple sources and combine results
        
        Args:
            query: Search query
            sources: List of search sources to use (currently only 'ddg')
            max_total_results: Maximum total results to return
            include_news: Whether to include news results
            site_filters: List of sites to filter by
            time_range: Time range filter
        """
        if sources is None:
            # Default sources based on engine preference
            if self.default_engine == "tavily":
                sources = ["tavily", "ddg"]
            else:
                sources = ["ddg", "tavily"]
        
        all_results = []
        
        # Try primary search engine first
        primary_engine = self.default_engine
        if primary_engine == "tavily" and "tavily" in sources:
            # Use Tavily as primary
            if site_filters:
                results = await self.tavily_searcher.search(
                    query=query,
                    max_results=self.max_results_per_source,
                    include_domains=site_filters,
                )
                all_results.extend(results)
            else:
                results = await self.tavily_searcher.search(
                    query=query,
                    max_results=self.max_results_per_source,
                )
                all_results.extend(results)
        
        # Fallback to DuckDuckGo or use as primary
        if (not all_results and "ddg" in sources) or (primary_engine == "duckduckgo" and "ddg" in sources):
            if site_filters:
                # Search each site filter separately
                for site in site_filters:
                    results = await self.ddg_searcher.search(
                        query=query,
                        max_results=self.max_results_per_source // len(site_filters),
                        site_filter=site,
                        time_range=time_range,
                    )
                    all_results.extend(results)
            else:
                results = await self.ddg_searcher.search(
                    query=query,
                    max_results=self.max_results_per_source,
                    time_range=time_range,
                )
                all_results.extend(results)
        
        # If still no results and we haven't tried Tavily, try it as fallback
        if not all_results and "tavily" in sources and primary_engine != "tavily":
            logger.info("No results from primary engine, trying Tavily as fallback")
            results = await self.tavily_searcher.search(
                query=query,
                max_results=self.max_results_per_source,
                include_domains=site_filters,
            )
            all_results.extend(results)
        
        # News search (only for DuckDuckGo for now)
        if include_news and "ddg" in sources:
            news_results = await self.ddg_searcher.search_news(
                query=query,
                max_results=self.max_results_per_source // 2,
                time_range=time_range,
            )
            all_results.extend(news_results)
        
        # Remove duplicates by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Sort by score (if available) and limit results
        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results[:max_total_results]
    
    async def search_category(
        self,
        category: str,
        queries: List[str],
        max_results: int = 50,
        site_preferences: Dict[str, List[str]] = None,
    ) -> List[SearchResult]:
        """
        Search for a category using multiple queries
        
        Args:
            category: Category name (e.g., 'ai_papers', 'crypto_news')
            queries: List of search queries for this category
            max_results: Maximum total results
            site_preferences: Preferred sites per category
        """
        all_results = []
        results_per_query = max_results // len(queries) if queries else max_results
        
        site_filters = None
        if site_preferences and category in site_preferences:
            site_filters = site_preferences[category]
        
        # Determine if this is a news category
        include_news = any(
            keyword in category.lower() 
            for keyword in ["news", "updates", "breaking", "latest"]
        )
        
        # Set time range based on category
        time_range = None
        if include_news:
            time_range = "w"  # Last week for news
        elif "papers" in category.lower():
            time_range = "m"  # Last month for papers
        
        for query in queries:
            logger.info(f"Searching for category '{category}' with query: '{query}'")
            
            # Use the configured search sources
            search_sources = None
            if self.default_engine == "tavily":
                search_sources = ["tavily", "ddg"]
            else:
                search_sources = ["ddg", "tavily"]
            
            results = await self.search(
                query=query,
                sources=search_sources,
                max_total_results=results_per_query,
                include_news=include_news,
                site_filters=site_filters,
                time_range=time_range,
            )
            
            all_results.extend(results)
        
        # Remove duplicates and limit results
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results[:max_results]


# Default searcher instance
default_searcher = MultiSearcher()