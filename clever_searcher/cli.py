"""Command-line interface for Clever Searcher"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler

from .utils.config import settings, get_data_dir, get_output_dir
from .storage.database import init_database, reset_database
from .agent import default_agent

console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Setup logging with rich handler"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--config-file", help="Path to config file")
def main(log_level: str, config_file: Optional[str]) -> None:
    """Clever Searcher - Autonomous web discovery and digest agent"""
    setup_logging(level=log_level)
    
    if config_file:
        # TODO: Load custom config file
        pass
    
    console.print("🔍 [bold blue]Clever Searcher[/bold blue] - Autonomous Web Discovery Agent")


@main.command()
def init() -> None:
    """Initialize the database and create necessary directories"""
    console.print("🚀 Initializing Clever Searcher...")
    
    try:
        # Create data directory
        data_dir = get_data_dir()
        console.print(f"📁 Data directory: {data_dir}")
        
        # Create output directory
        output_dir = get_output_dir()
        console.print(f"📁 Output directory: {output_dir}")
        
        # Initialize database
        init_database()
        console.print("✅ Database initialized successfully")
        
        # Create example .env file
        env_example = Path(".env.example")
        if not env_example.exists():
            env_content = """# Clever Searcher Configuration

# LLM Configuration (for Ollama)
OPENAI_BASE_URL=http://localhost:11434/v1
MODEL_PLANNER=llama3.1
MODEL_SUMMARY=llama3.1

# Or use OpenAI
# OPENAI_API_KEY=your_openai_key_here
# OPENAI_BASE_URL=https://api.openai.com/v1
# MODEL_PLANNER=gpt-4
# MODEL_SUMMARY=gpt-3.5-turbo

# Search APIs
TAVILY_API_KEY=your_tavily_api_key_here
DEFAULT_SEARCH_ENGINE=duckduckgo  # duckduckgo or tavily

# Database
DATABASE_URL=sqlite:///clever_searcher.db

# Search Configuration
MAX_PAGES_PER_RUN=50
MAX_PAGES_PER_DOMAIN=10
REQUEST_TIMEOUT=30
REQUEST_DELAY=1.0

# Content Processing
MIN_CONTENT_LENGTH=200
MAX_CONTENT_LENGTH=50000
SIMILARITY_THRESHOLD=0.85

# Personalization
EMBEDDING_MODEL=all-MiniLM-L6-v2
PERSONALIZATION_THRESHOLD=0.3

# Output
OUTPUT_DIR=output
DIGEST_FORMAT=markdown

# Logging
LOG_LEVEL=INFO
"""
            env_example.write_text(env_content)
            console.print(f"📝 Created example config file: {env_example}")
        
        console.print("\n✨ [bold green]Initialization complete![/bold green]")
        console.print("\n📋 Next steps:")
        console.print("1. Copy .env.example to .env and configure your settings")
        console.print("2. Start Ollama or configure OpenAI API key")
        console.print("3. Run: clever-searcher discover 'your topic' --max-pages 20")
        
    except Exception as e:
        console.print(f"❌ [bold red]Initialization failed:[/bold red] {e}")
        sys.exit(1)


@main.command()
@click.argument("category")
@click.option("--query", "-q", help="Additional search query")
@click.option("--max-pages", "-n", default=50, help="Maximum pages to discover")
@click.option("--max-queries", default=6, help="Maximum search queries to generate")
@click.option("--sites", help="Comma-separated list of preferred sites")
@click.option("--search-engine", default=None, help="Search engine to use (duckduckgo, tavily)")
@click.option("--llm-provider", default=None, help="LLM provider to use (openai, ollama)")
@click.option("--output-format", default="markdown", help="Output format (markdown, json, csv)")
@click.option("--dry-run", is_flag=True, help="Show plan without executing")
def discover(
    category: str,
    query: Optional[str],
    max_pages: int,
    max_queries: int,
    sites: Optional[str],
    search_engine: Optional[str],
    llm_provider: Optional[str],
    output_format: str,
    dry_run: bool,
) -> None:
    """Discover and analyze content for a given category"""
    
    async def run_discovery():
        console.print(f"🔍 Starting discovery for category: [bold]{category}[/bold]")
        
        # Parse sites
        custom_sites = []
        if sites:
            custom_sites = [site.strip() for site in sites.split(",")]
        
        if dry_run:
            # Just create and show the plan
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Creating crawl plan...", total=None)
                
                plan = await default_agent.planner.create_plan(
                    category=category,
                    user_query=query or "",
                    max_queries=max_queries,
                    max_pages=max_pages,
                    custom_sites=custom_sites,
                )
                
                progress.update(task, description="✅ Crawl plan created")
            
            # Display plan
            console.print("\n📋 [bold]Crawl Plan:[/bold]")
            plan_table = Table(show_header=True, header_style="bold magenta")
            plan_table.add_column("Property", style="cyan")
            plan_table.add_column("Value", style="white")
            
            plan_table.add_row("Category", plan.category)
            plan_table.add_row("Queries", ", ".join(plan.queries))
            plan_table.add_row("Max Pages", str(plan.max_pages))
            plan_table.add_row("Preferred Sites", ", ".join(plan.preferred_sites))
            plan_table.add_row("Include News", "Yes" if plan.include_news else "No")
            plan_table.add_row("Freshness", f"{plan.freshness_days} days")
            plan_table.add_row("Estimated Duration", f"{plan.estimated_duration_minutes} minutes")
            
            console.print(plan_table)
            console.print("\n🏃 [bold yellow]Dry run complete - no content fetched[/bold yellow]")
            return
        
        # Initialize agent with search engine preference
        from clever_searcher.agent import CleverSearcherAgent
        
        # Determine search engine
        engine = search_engine or settings.default_search_engine
        if engine not in ["duckduckgo", "tavily"]:
            engine = "duckduckgo"  # Default fallback
        
        # Override LLM provider if specified
        if llm_provider:
            if llm_provider not in ["openai", "ollama"]:
                console.print(f"❌ Invalid LLM provider: {llm_provider}. Using default: {settings.llm_provider}")
            else:
                settings.llm_provider = llm_provider
                console.print(f"🤖 Using LLM provider: [bold]{llm_provider}[/bold]")
        
        console.print(f"🔍 Using search engine: [bold]{engine}[/bold]")
        console.print(f"🤖 Using LLM provider: [bold]{settings.llm_provider}[/bold]")
        console.print(f"📝 Using models: [bold]{settings.model_planner}[/bold] (planner), [bold]{settings.model_summary}[/bold] (summary)")
        
        # Create agent with specified search engine
        agent = CleverSearcherAgent(search_engine=engine)
        
        # Run full discovery using the agent
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running discovery pipeline...", total=None)
            
            result = await agent.discover(
                category=category,
                user_query=query or "",
                max_pages=max_pages,
                max_queries=max_queries,
                custom_sites=custom_sites,
                output_format=output_format,
                save_to_db=True,
            )
            
            progress.update(task, description="✅ Discovery completed")
        
        # Display results
        console.print(f"\n📊 [bold green]Discovery Results:[/bold green]")
        
        # Show statistics
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Duration", f"{result.stats['duration_seconds']:.1f} seconds")
        stats_table.add_row("Search Results", str(result.stats['search_results_found']))
        stats_table.add_row("Documents Fetched", str(result.stats['documents_fetched']))
        stats_table.add_row("Unique Documents", str(result.stats['unique_documents']))
        stats_table.add_row("Final Results", str(result.stats['final_results']))
        stats_table.add_row("Average Score", f"{result.stats['avg_score']:.3f}")
        stats_table.add_row("Top Domains", ", ".join(result.stats['top_domains'][:3]))
        
        console.print(stats_table)
        
        # Show top results
        if result.documents:
            console.print(f"\n📋 [bold]Top Results:[/bold]")
            results_table = Table(show_header=True, header_style="bold magenta")
            results_table.add_column("Rank", style="cyan", width=6)
            results_table.add_column("Title", style="white", max_width=50)
            results_table.add_column("Domain", style="yellow", width=20)
            results_table.add_column("Score", style="green", width=8)
            
            for i, (doc, score) in enumerate(zip(result.documents[:10], result.scores[:10]), 1):
                results_table.add_row(
                    str(i),
                    doc.title[:47] + "..." if len(doc.title) > 50 else doc.title,
                    doc.domain,
                    f"{score:.3f}"
                )
            
            console.print(results_table)
            
            if len(result.documents) > 10:
                console.print(f"... and {len(result.documents) - 10} more results")
            
            # Show digest path
            if result.digest_path:
                console.print(f"\n💾 [bold green]Digest saved to:[/bold green] {result.digest_path}")
        
        else:
            console.print("❌ [bold red]No valid content found[/bold red]")
    
    try:
        asyncio.run(run_discovery())
    except KeyboardInterrupt:
        console.print("\n⏹️  Discovery interrupted by user")
    except Exception as e:
        console.print(f"❌ [bold red]Discovery failed:[/bold red] {e}")
        sys.exit(1)


@main.command()
def reset() -> None:
    """Reset the database (WARNING: This will delete all data!)"""
    if click.confirm("⚠️  This will delete all data. Are you sure?"):
        try:
            reset_database()
            console.print("✅ [bold green]Database reset successfully[/bold green]")
        except Exception as e:
            console.print(f"❌ [bold red]Reset failed:[/bold red] {e}")
            sys.exit(1)


@main.command()
def status() -> None:
    """Show system status and configuration"""
    console.print("📊 [bold]Clever Searcher Status[/bold]\n")
    
    # Configuration
    config_table = Table(show_header=True, header_style="bold magenta")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")
    
    config_table.add_row("Data Directory", str(get_data_dir()))
    config_table.add_row("Output Directory", str(get_output_dir()))
    config_table.add_row("Database URL", settings.database_url)
    config_table.add_row("LLM Provider", settings.llm_provider)
    config_table.add_row("LLM Model (Planner)", settings.model_planner)
    config_table.add_row("LLM Model (Summary)", settings.model_summary)
    config_table.add_row("LLM Base URL", settings.openai_base_url)
    config_table.add_row("Default Search Engine", settings.default_search_engine)
    config_table.add_row("Max Pages per Run", str(settings.max_pages_per_run))
    config_table.add_row("Request Delay", f"{settings.request_delay}s")
    
    console.print(config_table)


@main.command()
def dashboard() -> None:
    """Launch the Streamlit dashboard"""
    console.print("🚀 Launching Streamlit dashboard...")
    console.print("📝 Note: Make sure you have streamlit installed")
    
    try:
        import subprocess
        import sys
        
        # Try to find the dashboard module
        dashboard_path = Path(__file__).parent / "output" / "dashboard.py"
        
        if not dashboard_path.exists():
            console.print("❌ Dashboard not found. Creating basic dashboard...")
            # TODO: Create basic dashboard
            console.print("📝 Dashboard creation not implemented yet")
            return
        
        # Launch streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
        
    except ImportError:
        console.print("❌ Streamlit not installed. Install with: pip install streamlit")
    except Exception as e:
        console.print(f"❌ Failed to launch dashboard: {e}")


if __name__ == "__main__":
    main()