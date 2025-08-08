# Clever Searcher

An autonomous web discovery and digest agent that plans, searches, fetches, extracts, de-duplicates, ranks, summarizes, and publishes content based on your interests.

## Architecture

```
[Category Input]
      ↓
   Planner (LLM) ──► Query generator + crawl strategy
      ↓
   Searcher (DuckDuckGo) ──► SERP links (seed frontier)
      ↓
   Crawler/Extractor (httpx/Playwright + trafilatura/readability)
      ↓
   De-duper + Canonicalizer (URL norm + content hash)
      ↓
   Personalization Scorer (embeddings + rules + feedback)
      ↓
   Summarizer (LLM map→reduce + structured fields)
      ↓
   Store (SQLite + FAISS) + CSV export
      ↓
   Digest (Markdown/HTML/Streamlit dashboard)
      ↓
   Scheduler (APS/cron) + Revisit policy
```

## Features

- **Intelligent Planning**: LLM-powered query generation and crawl strategy
- **Free Search**: DuckDuckGo integration (no API keys required!)
- **Smart Extraction**: Content extraction with trafilatura and readability
- **De-duplication**: URL canonicalization and content fingerprinting
- **Personalization**: Embedding-based scoring with feedback learning
- **Structured Summaries**: LLM-generated summaries with bullets, tags, entities
- **Flexible Storage**: SQLite + FAISS for semantic search
- **Automated Scheduling**: Configurable revisit policies per source
- **Rich Output**: Markdown digests, CSV exports, Streamlit dashboard

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -e .
   ```

2. **Initialize the system:**
   ```bash
   clever-searcher init
   ```

3. **Configure your LLM (choose one):**
   
   **Option A: Ollama (Free, Local)**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama3.1
   # The system will auto-detect Ollama
   ```
   
   **Option B: OpenAI API**
   ```bash
   # Edit .env file
   OPENAI_API_KEY=your_api_key_here
   OPENAI_BASE_URL=https://api.openai.com/v1
   ```

4. **Run your first discovery:**
   ```bash
   clever-searcher discover "AI papers" --max-pages 10 --dry-run
   clever-searcher discover "crypto news" --max-pages 15
   ```

5. **View results:**
   ```bash
   ls output/  # Check generated digests
   ```

## Usage Examples

### Discover AI Papers
```bash
clever-searcher discover "machine learning papers" \
  --sites arxiv.org,paperswithcode.com \
  --max-pages 25
```

### Job Search
```bash
clever-searcher discover "python developer jobs" \
  --sites ycombinator.com,stackoverflow.com \
  --max-pages 30
```

### Crypto News
```bash
clever-searcher discover "cryptocurrency news" \
  --sites coindesk.com,cointelegraph.com \
  --max-pages 20
```

### Custom Research
```bash
clever-searcher discover "startup funding" \
  --query "Series A venture capital" \
  --max-pages 40 \
  --output-format json
```

## Configuration

The system works out of the box with sensible defaults. For customization, edit `.env`:

```env
# LLM Configuration (Ollama - Free)
OPENAI_BASE_URL=http://localhost:11434/v1
MODEL_PLANNER=llama3.1
MODEL_SUMMARY=llama3.1

# Search Configuration
MAX_PAGES_PER_RUN=50
REQUEST_DELAY=1.0

# Content Processing
MIN_CONTENT_LENGTH=200
SIMILARITY_THRESHOLD=0.85

# Output
OUTPUT_DIR=output
DIGEST_FORMAT=markdown
```

## Project Structure

```
clever_searcher/
├── core/           # Core components
│   ├── planner.py     # LLM-based planning
│   ├── searcher.py    # DuckDuckGo search
│   ├── fetcher.py     # Content fetching
│   └── ...
├── storage/        # Data persistence
│   ├── models.py      # SQLAlchemy models
│   ├── database.py    # Database setup
│   └── ...
├── utils/          # Utilities
│   ├── config.py      # Configuration
│   └── ...
└── cli.py          # Command-line interface
```

## Why Clever Searcher?

- **🆓 Completely Free**: Uses DuckDuckGo (no API keys needed) + Ollama (local LLM)
- **🧠 Smart Planning**: LLM generates diverse, effective search queries
- **⚡ Fast & Efficient**: Async processing with intelligent fallbacks
- **🎯 Personalized**: Learns your preferences over time
- **📊 Rich Output**: Multiple formats (Markdown, JSON, CSV)
- **🔧 Extensible**: Clean architecture for adding new features

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy clever_searcher
```

## Troubleshooting

### Common Issues

1. **Installation fails**: Make sure you have Python 3.10+
2. **LLM errors**: Install Ollama or configure OpenAI API key
3. **Search failures**: DuckDuckGo might be rate-limiting; reduce `--max-pages`
4. **Import errors**: Run `pip install -e .` from the project root

### Get Help

```bash
clever-searcher --help
clever-searcher status
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read the contributing guidelines and submit pull requests.