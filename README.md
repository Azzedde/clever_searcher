# Clever Searcher
<img width="1413" height="783" alt="Screenshot from 2025-08-09 14-49-00" src="https://github.com/user-attachments/assets/e66923f8-e46a-4695-8d61-a6f34d5693d6" />

An intelligent web discovery and content analysis agent that autonomously searches, fetches, analyzes, and summarizes web content based on your queries. Built with LLM-powered planning, multi-source search, smart deduplication, and personalized scoring.

## What It Actually Does

Clever Searcher takes a natural language query and:

1. **🧠 Plans** - Uses LLM to generate diverse search queries and crawl strategies
2. **🔍 Searches** - Searches DuckDuckGo and/or Tavily for relevant content
3. **📥 Fetches** - Downloads content using httpx with Playwright fallback for JS-heavy sites
4. **🔄 Deduplicates** - Removes duplicate content using URL canonicalization and content hashing
5. **📝 Summarizes** - Generates structured summaries with key points, tags, and entities
6. **⭐ Scores** - Ranks content using embeddings and personalization (with simple fallback)
7. **📊 Outputs** - Creates markdown digests, JSON exports, and database storage
8. **🎯 Learns** - Collects preference data for GRPO training and personalization

## Key Features

- **🆓 Free to Use**: DuckDuckGo search (no API keys) + Ollama support (local LLM)
- **🤖 LLM-Powered Planning**: Intelligent query generation and crawl strategy
- **🔄 Smart Deduplication**: URL canonicalization + content fingerprinting
- **📊 Structured Summaries**: Key points, tags, entities, and read time estimates
- **🎯 Personalization**: Embedding-based scoring with user feedback learning
- **📈 Preference Learning**: GRPO dataset collection for model improvement
- **🔧 Flexible Architecture**: Modular components with fallback modes
- **📋 Rich Logging**: Complete operation tracking and analytics

## Installation

```bash
# Clone the repository
git clone https://github.com/Azzedde/clever_searcher.git
cd clever_searcher

# Install the package
pip install -e .
```

## Quick Start

### 1. Initialize the System
```bash
clever-searcher init
```

### 2. Configure LLM (Choose One)

**Option A: Ollama (Free, Local)**
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.2:3b
# System auto-detects Ollama at http://localhost:11434
```

**Option B: OpenAI API**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "OPENAI_BASE_URL=https://api.openai.com/v1" >> .env
```

### 3. Run Your First Discovery

```bash
# Dry run to see the plan
clever-searcher discover "AI research papers" --max-pages 20 --dry-run

# Full discovery
clever-searcher discover "AI research papers" --max-pages 20

# With specific sites
clever-searcher discover "crypto news" --sites "coindesk.com,cointelegraph.com" --max-pages 15

# Different output format
clever-searcher discover "python tutorials" --output-format json --max-pages 10
```

## CLI Commands

### Core Commands
```bash
# Discovery
clever-searcher discover "your query" [options]

# System management
clever-searcher init          # Initialize database and directories
clever-searcher status        # Show system status
clever-searcher reset         # Reset database
clever-searcher dashboard     # Launch Streamlit dashboard
clever-searcher logs          # View recent logs
<img width="1288" height="642" alt="Screenshot from 2025-08-09 14-48-07" src="https://github.com/user-attachments/assets/fb5f98f1-9ce5-4f47-8938-c7c0d4424fdc" />

# Preference learning
clever-searcher preferences stats     # Dataset statistics
clever-searcher preferences pending   # Sessions needing feedback
clever-searcher preferences review    # Review and provide feedback
clever-searcher preferences export    # Export GRPO training data
```

### Discovery Options
```bash
--max-pages N           # Maximum pages to fetch (default: 50)
--max-queries N         # Maximum search queries to generate (default: 6)
--sites "site1,site2"   # Preferred sites to search
--search-engine ENGINE  # duckduckgo or tavily
--output-format FORMAT  # markdown, json, or csv
--dry-run              # Show plan without executing
```

## Configuration

The system works with sensible defaults. Customize via `.env`:

```env
# LLM Configuration
OPENAI_API_KEY=your_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # or http://localhost:11434/v1 for Ollama
MODEL_PLANNER=gpt-4o-mini                  # or llama3.2:3b for Ollama
MODEL_SUMMARY=gpt-4o-mini                  # or llama3.2:3b for Ollama

# Search Configuration
TAVILY_API_KEY=your_tavily_key            # Optional: for Tavily search
DEFAULT_SEARCH_ENGINE=duckduckgo          # duckduckgo or tavily
MAX_PAGES_PER_RUN=50
REQUEST_DELAY=1.0

# Content Processing
MIN_CONTENT_LENGTH=200
SIMILARITY_THRESHOLD=0.85

# Output
OUTPUT_DIR=output
DIGEST_FORMAT=markdown
```

## Architecture

```
Query Input
    ↓
🧠 LLM Planner → Generates search queries + strategy
    ↓
🔍 Multi-Searcher → DuckDuckGo/Tavily search
    ↓
📥 Smart Fetcher → httpx + Playwright fallback
    ↓
🔄 Deduplicator → URL canonicalization + content hashing
    ↓
📝 LLM Summarizer → Structured summaries with metadata
    ↓
⭐ Personalization → Embedding-based scoring + user preferences
    ↓
💾 Storage → SQLite database + file exports
    ↓
📊 Digest Generator → Markdown/JSON/CSV outputs
    ↓
🎯 Preference Collector → GRPO training data
```

## Project Structure

```
clever_searcher/
├── agent.py              # Main orchestrator
├── cli.py                # Command-line interface
├── core/                 # Core processing components
│   ├── planner.py        # LLM-based query planning
│   ├── searcher.py       # Multi-source search (DuckDuckGo/Tavily)
│   ├── fetcher.py        # Content fetching (httpx/Playwright)
│   ├── deduper.py        # Deduplication and canonicalization
│   ├── summarizer.py     # LLM-powered summarization
│   ├── scorer.py         # Embedding-based personalization
│   └── simple_scorer.py  # Fallback scoring without embeddings
├── storage/              # Data persistence
│   ├── models.py         # SQLAlchemy models
│   └── database.py       # Database management
├── output/               # Output generation
│   └── digest.py         # Multi-format digest generation
├── logging/              # Advanced logging
│   ├── operation_logger.py    # Complete operation tracking
│   └── preference_collector.py # GRPO dataset collection
└── utils/                # Configuration and utilities
    └── config.py         # Settings management
```

## Use Cases

### Research & Discovery
```bash
# Academic research
clever-searcher discover "machine learning interpretability papers" --sites "arxiv.org,paperswithcode.com"

# Market research
clever-searcher discover "fintech startup funding 2024" --max-pages 30

# Technology trends
clever-searcher discover "rust programming language adoption" --max-queries 8
```

### News & Monitoring
```bash
# Industry news
clever-searcher discover "AI regulation updates" --sites "techcrunch.com,arstechnica.com"

# Company monitoring
clever-searcher discover "OpenAI latest developments" --max-pages 25
```

### Learning & Tutorials
```bash
# Technical tutorials
clever-searcher discover "advanced python asyncio patterns" --sites "realpython.com,python.org"

# Best practices
clever-searcher discover "microservices architecture patterns" --output-format json
```

## Advanced Features

### Preference Learning
The system collects preference data for GRPO (Generalized Preference Optimization) training:

```bash
# View collected data
clever-searcher preferences stats

# Review and provide feedback
clever-searcher preferences review session_id

# Export training dataset
clever-searcher preferences export --output training_data.jsonl
```
<img width="1285" height="458" alt="Screenshot from 2025-08-09 14-47-27" src="https://github.com/user-attachments/assets/9d3eb230-e5e6-43b1-8294-93109159e0cc" />

### Personalization
- Learns from your interactions and feedback
- Uses embeddings to understand content preferences
- Adapts scoring based on your interests
- Falls back to simple scoring if embeddings fail

### Smart Fallbacks
- Playwright fallback for JavaScript-heavy sites
- Simple scoring fallback if embeddings fail
- Multiple search engines with automatic failover
- Graceful degradation for all components

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Code formatting
black .
isort .

# Type checking
mypy clever_searcher
```

## Troubleshooting

### Common Issues

1. **LLM Connection Errors**
   - For Ollama: Ensure it's running (`ollama serve`)
   - For OpenAI: Check API key in `.env`

2. **Search Failures**
   - DuckDuckGo may rate-limit; reduce `--max-pages`
   - Try switching to Tavily with `--search-engine tavily`

3. **Content Fetching Issues**
   - Some sites block automated requests
   - Playwright fallback handles most JS-heavy sites

4. **Embedding Errors**
   - System automatically falls back to simple scoring
   - Check internet connection for model downloads

### Getting Help

```bash
clever-searcher --help
clever-searcher status
clever-searcher logs
```

## Why Clever Searcher?

- **🎯 Intelligent**: LLM-powered planning creates better search strategies
- **🔄 Robust**: Multiple fallback mechanisms ensure reliability
- **📊 Comprehensive**: Complete pipeline from search to structured output
- **🎓 Learning**: Builds preference datasets for continuous improvement
- **🆓 Accessible**: Works with free tools (DuckDuckGo + Ollama)
- **🔧 Extensible**: Modular architecture for easy customization

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Author

**Azzedine** - [Website](https://azzedde.github.io/) | [GitHub](https://github.com/Azzedde)
