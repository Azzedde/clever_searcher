"""Content fetching with httpx and Playwright fallback"""

import asyncio
import logging
import hashlib
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse, urljoin
from datetime import datetime

import httpx
import trafilatura
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page

from ..utils.config import settings

logger = logging.getLogger(__name__)


class ContentDocument:
    """Represents extracted content from a web page"""
    
    def __init__(
        self,
        url: str,
        title: str = "",
        content: str = "",
        author: str = "",
        published_date: Optional[datetime] = None,
        language: str = "",
        content_hash: str = "",
        metadata: Dict[str, Any] = None,
        outlinks: List[str] = None,
        status_code: int = 200,
        content_length: int = 0,
    ):
        self.url = url
        self.title = title
        self.content = content
        self.author = author
        self.published_date = published_date
        self.language = language
        self.content_hash = content_hash or self._compute_hash()
        self.metadata = metadata or {}
        self.outlinks = outlinks or []
        self.status_code = status_code
        self.content_length = content_length or len(content)
        self.domain = urlparse(url).netloc
        self.fetched_at = datetime.utcnow()
    
    def _compute_hash(self) -> str:
        """Compute content hash for deduplication"""
        content_for_hash = f"{self.title}\n{self.content}".strip()
        return hashlib.blake2b(content_for_hash.encode("utf-8"), digest_size=16).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content[:1000] + "..." if len(self.content) > 1000 else self.content,
            "author": self.author,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "language": self.language,
            "content_hash": self.content_hash,
            "domain": self.domain,
            "content_length": self.content_length,
            "status_code": self.status_code,
            "fetched_at": self.fetched_at.isoformat(),
        }
    
    def is_valid(self) -> bool:
        """Check if the document has valid content"""
        return (
            self.status_code == 200 and
            len(self.content.strip()) >= settings.min_content_length and
            len(self.content) <= settings.max_content_length and
            self.title.strip() != ""
        )


class HttpxFetcher:
    """Fast HTTP fetcher using httpx"""
    
    def __init__(self, timeout: int = 30, max_redirects: int = 5):
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    
    async def fetch(self, url: str) -> Optional[ContentDocument]:
        """Fetch and extract content from URL"""
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                max_redirects=self.max_redirects,
                headers=self.headers,
            ) as client:
                logger.debug(f"Fetching URL: {url}")
                response = await client.get(url)
                
                if response.status_code != 200:
                    logger.warning(f"HTTP {response.status_code} for URL: {url}")
                    return None
                
                # Check content type
                content_type = response.headers.get("content-type", "").lower()
                if not any(ct in content_type for ct in ["text/html", "application/xhtml"]):
                    logger.warning(f"Unsupported content type '{content_type}' for URL: {url}")
                    return None
                
                html_content = response.text
                return self._extract_content(url, html_content, response.status_code)
                
        except Exception as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            return None
    
    def _extract_content(self, url: str, html: str, status_code: int) -> Optional[ContentDocument]:
        """Extract content from HTML using trafilatura"""
        try:
            # Use trafilatura for main content extraction
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_images=False,
                include_links=False,
                favor_precision=True,
            )
            
            if not extracted or len(extracted.strip()) < settings.min_content_length:
                logger.warning(f"Insufficient content extracted from {url}")
                return None
            
            # Extract metadata using trafilatura
            metadata = trafilatura.extract_metadata(html)
            
            # Parse HTML for additional information
            soup = BeautifulSoup(html, "html.parser")
            
            # Extract title
            title = ""
            if metadata and metadata.title:
                title = metadata.title
            elif soup.title:
                title = soup.title.get_text().strip()
            
            # Extract author
            author = ""
            if metadata and metadata.author:
                author = metadata.author
            else:
                # Try common author meta tags
                author_meta = soup.find("meta", attrs={"name": "author"}) or \
                             soup.find("meta", attrs={"property": "article:author"})
                if author_meta:
                    author = author_meta.get("content", "").strip()
            
            # Extract published date
            published_date = None
            if metadata and metadata.date:
                try:
                    published_date = datetime.fromisoformat(metadata.date.replace("Z", "+00:00"))
                except Exception:
                    pass
            
            # Extract outbound links
            outlinks = []
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.startswith("http"):
                    outlinks.append(href)
                elif href.startswith("/"):
                    outlinks.append(urljoin(url, href))
            
            # Limit outlinks
            outlinks = list(set(outlinks))[:50]
            
            return ContentDocument(
                url=url,
                title=title,
                content=extracted,
                author=author,
                published_date=published_date,
                language=metadata.language if metadata else "",
                metadata={
                    "description": metadata.description if metadata else "",
                    "sitename": metadata.sitename if metadata else "",
                    "categories": metadata.categories if metadata else [],
                    "tags": metadata.tags if metadata else [],
                },
                outlinks=outlinks,
                status_code=status_code,
            )
            
        except Exception as e:
            logger.error(f"Failed to extract content from {url}: {e}")
            return None


class PlaywrightFetcher:
    """Playwright-based fetcher for JavaScript-heavy sites"""
    
    def __init__(self, timeout: int = 30, headless: bool = True):
        self.timeout = timeout
        self.headless = headless
        self._browser: Optional[Browser] = None
    
    async def __aenter__(self):
        self.playwright = await async_playwright().start()
        self._browser = await self.playwright.chromium.launch(
            headless=self.headless,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._browser:
            await self._browser.close()
        await self.playwright.stop()
    
    async def close(self):
        """Close the browser and playwright instance"""
        if self._browser:
            await self._browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
    
    async def fetch(self, url: str) -> Optional[ContentDocument]:
        """Fetch content using Playwright"""
        # Initialize browser if not already done
        if not self._browser:
            await self._initialize_browser()
        
        page = None
        try:
            page = await self._browser.new_page()
            
            # Set user agent and viewport
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            await page.set_viewport_size({"width": 1920, "height": 1080})
            
            logger.debug(f"Fetching URL with Playwright: {url}")
            
            # Navigate to page
            response = await page.goto(url, timeout=self.timeout * 1000, wait_until="domcontentloaded")
            
            if not response or response.status != 200:
                logger.warning(f"HTTP {response.status if response else 'unknown'} for URL: {url}")
                return None
            
            # Wait for content to load
            await page.wait_for_timeout(2000)
            
            # Get page content
            html_content = await page.content()
            
            # Extract using the same method as httpx fetcher
            httpx_fetcher = HttpxFetcher()
            return httpx_fetcher._extract_content(url, html_content, response.status)
            
        except Exception as e:
            logger.error(f"Playwright fetch failed for {url}: {e}")
            return None
        finally:
            if page:
                await page.close()
    
    async def _initialize_browser(self):
        """Initialize browser if not already done"""
        if not self._browser:
            self.playwright = await async_playwright().start()
            self._browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )


class SmartFetcher:
    """Smart fetcher that tries httpx first, falls back to Playwright"""
    
    def __init__(self, use_playwright_fallback: bool = True):
        self.httpx_fetcher = HttpxFetcher(timeout=settings.request_timeout)
        self.use_playwright_fallback = use_playwright_fallback
        self.playwright_domains = {
            # Domains that typically require JavaScript
            "twitter.com", "x.com", "linkedin.com", "facebook.com",
            "instagram.com", "tiktok.com", "youtube.com",
        }
    
    async def fetch(self, url: str, force_playwright: bool = False) -> Optional[ContentDocument]:
        """Fetch content with smart fallback strategy"""
        domain = urlparse(url).netloc.lower()
        
        # Try Playwright first for known JS-heavy sites
        if force_playwright or domain in self.playwright_domains:
            if self.use_playwright_fallback:
                async with PlaywrightFetcher() as playwright_fetcher:
                    result = await playwright_fetcher.fetch(url)
                    if result and result.is_valid():
                        logger.info(f"Successfully fetched {url} with Playwright")
                        return result
            
            # If Playwright fails or is disabled, try httpx
            logger.info(f"Playwright failed for {url}, trying httpx")
        
        # Try httpx first (or as fallback)
        result = await self.httpx_fetcher.fetch(url)
        if result and result.is_valid():
            logger.info(f"Successfully fetched {url} with httpx")
            return result
        
        # Fallback to Playwright if httpx fails
        if not force_playwright and self.use_playwright_fallback:
            logger.info(f"httpx failed for {url}, trying Playwright fallback")
            async with PlaywrightFetcher() as playwright_fetcher:
                result = await playwright_fetcher.fetch(url)
                if result and result.is_valid():
                    logger.info(f"Successfully fetched {url} with Playwright fallback")
                    return result
        
        logger.warning(f"Failed to fetch content from {url}")
        return None
    
    async def fetch_batch(
        self,
        urls: List[str],
        max_concurrent: int = 5,
        delay: float = None,
    ) -> List[ContentDocument]:
        """Fetch multiple URLs concurrently"""
        if delay is None:
            delay = settings.request_delay
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_with_semaphore(url: str) -> Optional[ContentDocument]:
            async with semaphore:
                result = await self.fetch(url)
                if delay > 0:
                    await asyncio.sleep(delay)
                return result
        
        logger.info(f"Fetching {len(urls)} URLs with max_concurrent={max_concurrent}")
        
        tasks = [fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        valid_results = []
        for result in results:
            if isinstance(result, ContentDocument) and result.is_valid():
                valid_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Batch fetch error: {result}")
        
        logger.info(f"Successfully fetched {len(valid_results)} out of {len(urls)} URLs")
        return valid_results


# Default fetcher instance
default_fetcher = SmartFetcher()