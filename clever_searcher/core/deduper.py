"""De-duplication and URL canonicalization utilities"""

import hashlib
import logging
import re
from typing import List, Set, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..storage.models import Page
from ..utils.config import settings

logger = logging.getLogger(__name__)


class URLCanonicalizer:
    """Canonicalizes URLs for consistent deduplication"""
    
    def __init__(self):
        # Parameters to remove from URLs
        self.tracking_params = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'fbclid', 'gclid', 'msclkid', 'twclid', 'igshid',
            'ref', 'referrer', 'source', 'campaign',
            '_ga', '_gid', '_gac', '_gl',
            'mc_cid', 'mc_eid',  # MailChimp
            'pk_campaign', 'pk_kwd',  # Piwik
            'hsCtaTracking', 'hsa_acc', 'hsa_cam', 'hsa_grp', 'hsa_ad', 'hsa_src', 'hsa_tgt', 'hsa_kw', 'hsa_mt', 'hsa_net', 'hsa_ver',  # HubSpot
        }
        
        # Common URL patterns to normalize
        self.normalization_patterns = [
            # Remove trailing slashes
            (r'/$', ''),
            # Normalize www
            (r'^https?://www\.', 'https://'),
            # Remove default ports
            (r':80/', '/'),
            (r':443/', '/'),
            # Normalize case for domains
            (r'^(https?://[^/]+)', lambda m: m.group(1).lower()),
        ]
    
    def canonicalize(self, url: str) -> str:
        """Canonicalize a URL for consistent comparison"""
        try:
            # Parse URL
            parsed = urlparse(url.strip())
            
            # Normalize scheme
            scheme = parsed.scheme.lower() if parsed.scheme else 'https'
            
            # Normalize netloc (domain)
            netloc = parsed.netloc.lower()
            
            # Remove www prefix for consistency
            if netloc.startswith('www.'):
                netloc = netloc[4:]
            
            # Normalize path
            path = parsed.path
            if not path:
                path = '/'
            
            # Clean query parameters
            query_params = parse_qs(parsed.query, keep_blank_values=False)
            clean_params = {}
            
            for key, values in query_params.items():
                # Skip tracking parameters
                if key.lower() not in self.tracking_params:
                    # Keep only the first value for each parameter
                    clean_params[key] = values[0] if values else ''
            
            # Rebuild query string
            clean_query = urlencode(sorted(clean_params.items())) if clean_params else ''
            
            # Remove fragment (anchor)
            fragment = ''
            
            # Reconstruct URL
            canonical_url = urlunparse((
                scheme, netloc, path, parsed.params, clean_query, fragment
            ))
            
            # Apply additional normalization patterns
            for pattern, replacement in self.normalization_patterns:
                if callable(replacement):
                    canonical_url = re.sub(pattern, replacement, canonical_url)
                else:
                    canonical_url = re.sub(pattern, replacement, canonical_url)
            
            return canonical_url
            
        except Exception as e:
            logger.warning(f"Failed to canonicalize URL '{url}': {e}")
            return url
    
    def get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return ""


class ContentHasher:
    """Generates content hashes for deduplication"""
    
    def __init__(self):
        self.min_content_length = 50
    
    def hash_content(self, title: str, content: str) -> str:
        """Generate a hash for content deduplication"""
        # Combine title and content
        combined_text = f"{title.strip()}\n{content.strip()}"
        
        # Normalize text for hashing
        normalized = self._normalize_text(combined_text)
        
        # Generate hash
        return hashlib.blake2b(
            normalized.encode('utf-8'), 
            digest_size=16
        ).hexdigest()
    
    def hash_text_only(self, text: str) -> str:
        """Generate hash for text content only"""
        normalized = self._normalize_text(text)
        return hashlib.blake2b(
            normalized.encode('utf-8'), 
            digest_size=16
        ).hexdigest()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for consistent hashing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common punctuation that might vary
        text = re.sub(r'[^\w\s]', '', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def similarity_hash(self, text: str, shingle_size: int = 3) -> Set[str]:
        """Generate similarity hash using shingling"""
        normalized = self._normalize_text(text)
        words = normalized.split()
        
        if len(words) < shingle_size:
            return {normalized}
        
        shingles = set()
        for i in range(len(words) - shingle_size + 1):
            shingle = ' '.join(words[i:i + shingle_size])
            shingles.add(shingle)
        
        return shingles


class ContentDeduplicator:
    """Handles content deduplication using multiple strategies"""
    
    def __init__(self, similarity_threshold: float = None):
        self.similarity_threshold = similarity_threshold or settings.similarity_threshold
        self.url_canonicalizer = URLCanonicalizer()
        self.content_hasher = ContentHasher()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Cache for seen content
        self.seen_urls: Set[str] = set()
        self.seen_hashes: Set[str] = set()
        self.content_vectors: Dict[str, np.ndarray] = {}
    
    def is_duplicate_url(self, url: str) -> Tuple[bool, str]:
        """Check if URL is a duplicate"""
        canonical_url = self.url_canonicalizer.canonicalize(url)
        
        if canonical_url in self.seen_urls:
            return True, canonical_url
        
        self.seen_urls.add(canonical_url)
        return False, canonical_url
    
    def is_duplicate_content(self, title: str, content: str) -> Tuple[bool, str]:
        """Check if content is a duplicate by hash"""
        content_hash = self.content_hasher.hash_content(title, content)
        
        if content_hash in self.seen_hashes:
            return True, content_hash
        
        self.seen_hashes.add(content_hash)
        return False, content_hash
    
    def is_similar_content(
        self, 
        title: str, 
        content: str, 
        content_id: str = None
    ) -> Tuple[bool, float, Optional[str]]:
        """Check if content is similar to existing content using TF-IDF"""
        try:
            # Combine title and content
            text = f"{title} {content}"
            
            if len(self.content_vectors) == 0:
                # First document
                vector = self.vectorizer.fit_transform([text])
                if content_id:
                    self.content_vectors[content_id] = vector.toarray()[0]
                return False, 0.0, None
            
            # Transform new document
            try:
                vector = self.vectorizer.transform([text])
                new_vector = vector.toarray()[0]
            except Exception:
                # Vocabulary mismatch, refit with all documents
                all_texts = [text]
                all_ids = [content_id] if content_id else []
                
                for existing_id in list(self.content_vectors.keys()):
                    # We don't have the original text, so skip similarity check
                    pass
                
                vector = self.vectorizer.fit_transform(all_texts)
                if content_id:
                    self.content_vectors[content_id] = vector.toarray()[0]
                return False, 0.0, None
            
            # Check similarity with existing documents
            max_similarity = 0.0
            most_similar_id = None
            
            for existing_id, existing_vector in self.content_vectors.items():
                similarity = cosine_similarity([new_vector], [existing_vector])[0][0]
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_id = existing_id
            
            # Store vector for future comparisons
            if content_id:
                self.content_vectors[content_id] = new_vector
            
            is_duplicate = max_similarity >= self.similarity_threshold
            return is_duplicate, max_similarity, most_similar_id
            
        except Exception as e:
            logger.warning(f"Similarity check failed: {e}")
            return False, 0.0, None
    
    def check_duplicate(
        self, 
        url: str, 
        title: str, 
        content: str,
        content_id: str = None
    ) -> Dict[str, Any]:
        """Comprehensive duplicate check"""
        result = {
            'is_duplicate': False,
            'duplicate_type': None,
            'canonical_url': '',
            'content_hash': '',
            'similarity_score': 0.0,
            'similar_to': None,
        }
        
        # Check URL duplication
        is_url_dup, canonical_url = self.is_duplicate_url(url)
        result['canonical_url'] = canonical_url
        
        if is_url_dup:
            result['is_duplicate'] = True
            result['duplicate_type'] = 'url'
            return result
        
        # Check content hash duplication
        is_content_dup, content_hash = self.is_duplicate_content(title, content)
        result['content_hash'] = content_hash
        
        if is_content_dup:
            result['is_duplicate'] = True
            result['duplicate_type'] = 'content_hash'
            return result
        
        # Check content similarity
        is_similar, similarity_score, similar_to = self.is_similar_content(
            title, content, content_id
        )
        result['similarity_score'] = similarity_score
        result['similar_to'] = similar_to
        
        if is_similar:
            result['is_duplicate'] = True
            result['duplicate_type'] = 'content_similarity'
            return result
        
        return result
    
    def reset(self):
        """Reset the deduplicator state"""
        self.seen_urls.clear()
        self.seen_hashes.clear()
        self.content_vectors.clear()
        # Reset vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics"""
        return {
            'seen_urls': len(self.seen_urls),
            'seen_hashes': len(self.seen_hashes),
            'content_vectors': len(self.content_vectors),
            'similarity_threshold': self.similarity_threshold,
        }


class DuplicateTracker:
    """Tracks duplicates found during crawling"""
    
    def __init__(self):
        self.duplicates: List[Dict[str, Any]] = []
        self.stats = {
            'total_checked': 0,
            'url_duplicates': 0,
            'content_hash_duplicates': 0,
            'similarity_duplicates': 0,
            'unique_content': 0,
        }
    
    def record_duplicate(self, duplicate_info: Dict[str, Any]):
        """Record a duplicate finding"""
        self.duplicates.append({
            **duplicate_info,
            'timestamp': datetime.utcnow().isoformat(),
        })
        
        # Update stats
        self.stats['total_checked'] += 1
        
        if duplicate_info['is_duplicate']:
            dup_type = duplicate_info['duplicate_type']
            if dup_type == 'url':
                self.stats['url_duplicates'] += 1
            elif dup_type == 'content_hash':
                self.stats['content_hash_duplicates'] += 1
            elif dup_type == 'content_similarity':
                self.stats['similarity_duplicates'] += 1
        else:
            self.stats['unique_content'] += 1
    
    def get_report(self) -> Dict[str, Any]:
        """Generate a deduplication report"""
        total = self.stats['total_checked']
        if total == 0:
            return self.stats
        
        return {
            **self.stats,
            'duplicate_rate': (total - self.stats['unique_content']) / total,
            'url_duplicate_rate': self.stats['url_duplicates'] / total,
            'content_duplicate_rate': self.stats['content_hash_duplicates'] / total,
            'similarity_duplicate_rate': self.stats['similarity_duplicates'] / total,
        }


# Default instances
default_canonicalizer = URLCanonicalizer()
default_deduplicator = ContentDeduplicator()
default_tracker = DuplicateTracker()