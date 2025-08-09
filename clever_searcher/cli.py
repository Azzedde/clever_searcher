"""Command-line interface for Clever Searcher"""

import asyncio
import logging
import sys
from datetime import datetime
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
from .logging.operation_logger import operation_logger
from .logging.preference_collector import preference_collector

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
@click.argument("query")
@click.option("--max-pages", "-n", default=50, help="Maximum pages to discover")
@click.option("--max-queries", default=6, help="Maximum search queries to generate")
@click.option("--sites", help="Comma-separated list of preferred sites")
@click.option("--search-engine", default=None, help="Search engine to use (duckduckgo, tavily)")
@click.option("--llm-provider", default=None, help="LLM provider to use (openai, ollama)")
@click.option("--output-format", default="markdown", help="Output format (markdown, json, csv)")
@click.option("--dry-run", is_flag=True, help="Show plan without executing")
def discover(
    query: str,
    max_pages: int,
    max_queries: int,
    sites: Optional[str],
    search_engine: Optional[str],
    llm_provider: Optional[str],
    output_format: str,
    dry_run: bool,
) -> None:
    """Discover and analyze content for a given query"""
    
    async def run_discovery():
        console.print(f"🔍 Starting discovery for query: [bold]{query}[/bold]")
        
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
                    query=query,
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
                query=query,
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


@main.command()
def logs() -> None:
    """Show recent operation logs and statistics"""
    console.print("📊 [bold]Operation Logs & Statistics[/bold]\n")
    
    # Show operation statistics
    stats = operation_logger.get_operation_stats()
    
    stats_table = Table(show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")
    
    stats_table.add_row("Total Operations", str(stats.get("total_operations", 0)))
    stats_table.add_row("Completed", str(stats.get("completed", 0)))
    stats_table.add_row("Failed", str(stats.get("failed", 0)))
    stats_table.add_row("Avg Duration", f"{stats.get('avg_duration_seconds', 0):.1f}s")
    stats_table.add_row("Total Log Entries", str(stats.get("total_entries", 0)))
    stats_table.add_row("Avg Entries/Operation", f"{stats.get('avg_entries_per_operation', 0):.1f}")
    
    console.print(stats_table)
    
    # Show recent operations
    recent_logs = operation_logger.get_recent_logs(limit=5)
    if recent_logs:
        console.print(f"\n📋 [bold]Recent Operations:[/bold]")
        
        logs_table = Table(show_header=True, header_style="bold magenta")
        logs_table.add_column("Time", style="cyan", width=20)
        logs_table.add_column("Query", style="white", max_width=40)
        logs_table.add_column("Status", style="green", width=12)
        logs_table.add_column("Duration", style="yellow", width=10)
        logs_table.add_column("Results", style="blue", width=8)
        
        for log in recent_logs:
            started_at = datetime.fromisoformat(log["started_at"]).strftime("%Y-%m-%d %H:%M")
            query = log["user_query"][:37] + "..." if len(log["user_query"]) > 40 else log["user_query"]
            status = log["status"]
            duration = f"{log.get('metadata', {}).get('total_duration_seconds', 0):.1f}s"
            results = str(log.get('metadata', {}).get('stats', {}).get('final_results', 0))
            
            logs_table.add_row(started_at, query, status, duration, results)
        
        console.print(logs_table)
    else:
        console.print("\n📝 No operation logs found")


@main.group()
def preferences() -> None:
    """Manage preference data for GRPO training"""
    pass


@preferences.command()
def stats() -> None:
    """Show preference dataset statistics"""
    console.print("📊 [bold]Preference Dataset Statistics[/bold]\n")
    
    stats = preference_collector.get_dataset_stats()
    
    stats_table = Table(show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")
    
    stats_table.add_row("Total Sessions", str(stats.get("total_sessions", 0)))
    stats_table.add_row("Sessions with Feedback", str(stats.get("sessions_with_feedback", 0)))
    stats_table.add_row("Pending Feedback", str(stats.get("pending_feedback", 0)))
    stats_table.add_row("Avg Satisfaction", f"{stats.get('avg_satisfaction', 0):.2f}/5.0")
    
    console.print(stats_table)
    
    # Show common categories
    common_categories = stats.get("common_categories", {})
    if common_categories:
        console.print(f"\n📋 [bold]Common Categories:[/bold]")
        cat_table = Table(show_header=True, header_style="bold magenta")
        cat_table.add_column("Category", style="cyan")
        cat_table.add_column("Count", style="white")
        
        for category, count in list(common_categories.items())[:5]:
            cat_table.add_row(category, str(count))
        
        console.print(cat_table)
    
    # Show common domains
    common_domains = stats.get("common_domains", {})
    if common_domains:
        console.print(f"\n🌐 [bold]Common Domains:[/bold]")
        domain_table = Table(show_header=True, header_style="bold magenta")
        domain_table.add_column("Domain", style="cyan")
        domain_table.add_column("Count", style="white")
        
        for domain, count in list(common_domains.items())[:5]:
            domain_table.add_row(domain, str(count))
        
        console.print(domain_table)


@preferences.command()
def pending() -> None:
    """Show sessions pending human feedback"""
    console.print("⏳ [bold]Sessions Pending Feedback[/bold]\n")
    
    pending_sessions = preference_collector.get_pending_feedback_sessions()
    
    if not pending_sessions:
        console.print("✅ No sessions pending feedback")
        return
    
    pending_table = Table(show_header=True, header_style="bold magenta")
    pending_table.add_column("Session ID", style="cyan", width=12)
    pending_table.add_column("Time", style="yellow", width=16)
    pending_table.add_column("Query", style="white", max_width=30)
    pending_table.add_column("LLM Category", style="green", width=15)
    pending_table.add_column("Results", style="blue", width=8)
    
    for session in pending_sessions[:10]:  # Show top 10
        session_id = session["session_id"][:8] + "..."
        time_str = datetime.fromisoformat(session["timestamp"]).strftime("%m-%d %H:%M")
        query = session["original_query"][:27] + "..." if len(session["original_query"]) > 30 else session["original_query"]
        category = session["llm_category"]
        results = str(session["results_count"])
        
        pending_table.add_row(session_id, time_str, query, category, results)
    
    console.print(pending_table)
    
    if len(pending_sessions) > 10:
        console.print(f"\n... and {len(pending_sessions) - 10} more sessions")
    
    console.print(f"\n💡 [bold yellow]Tip:[/bold yellow] Use 'clever-searcher preferences feedback <session_id>' to add feedback")


@preferences.command()
@click.argument("session_id")
def review(session_id: str) -> None:
    """Interactive review session - see LLM data and approve/modify all in one place"""
    
    # Load session data
    session_data = preference_collector._load_session_data(session_id)
    if not session_data:
        console.print(f"❌ [bold red]Session not found: {session_id}[/bold red]")
        return
    
    console.print(f"📋 [bold]Interactive Review Session: {session_id[:8]}...[/bold]\n")
    
    # Show original query
    console.print(f"🔍 [bold cyan]Original Query:[/bold cyan] {session_data.original_query}\n")
    
    # Show LLM decisions
    llm_table = Table(show_header=True, header_style="bold magenta", title="🤖 LLM Generated Plan")
    llm_table.add_column("Field", style="cyan", width=20)
    llm_table.add_column("LLM Decision", style="white")
    
    llm_table.add_row("Category", session_data.llm_category)
    llm_table.add_row("Search Queries", ", ".join(session_data.llm_queries))
    llm_table.add_row("Preferred Sites", ", ".join(session_data.llm_preferred_sites))
    llm_table.add_row("Keywords", ", ".join(session_data.llm_keywords))
    llm_table.add_row("Avoid Keywords", ", ".join(session_data.llm_avoid_keywords))
    if session_data.llm_reasoning:
        llm_table.add_row("Reasoning", session_data.llm_reasoning)
    
    console.print(llm_table)
    
    # Show execution results
    console.print(f"\n📊 [bold]Execution Results:[/bold]")
    results_table = Table(show_header=True, header_style="bold magenta")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="white")
    
    results_table.add_row("Domains Searched", ", ".join(session_data.actual_domains_searched))
    results_table.add_row("Search Results", str(session_data.search_results_count))
    results_table.add_row("Unique Documents", str(session_data.unique_documents_found))
    results_table.add_row("Top Scoring Domains", ", ".join(session_data.top_scoring_domains))
    
    console.print(results_table)
    
    # Interactive feedback collection
    console.print(f"\n🎯 [bold yellow]Now provide your feedback:[/bold yellow]")
    
    # Ask for each field
    console.print(f"\n1️⃣ [bold]Category:[/bold] LLM chose '{session_data.llm_category}'")
    user_category = click.prompt("Your preferred category (press Enter to approve LLM choice)", default="", show_default=False)
    final_category = user_category.strip() if user_category.strip() else session_data.llm_category
    
    console.print(f"\n2️⃣ [bold]Search Queries:[/bold] LLM generated {len(session_data.llm_queries)} queries")
    console.print("Review each query individually:")
    
    final_queries = []
    for i, llm_query in enumerate(session_data.llm_queries, 1):
        console.print(f"\n   Query {i}: [yellow]'{llm_query}'[/yellow]")
        user_choice = click.prompt(
            f"   [A]pprove, [M]odify, or [S]kip this query?",
            type=click.Choice(['A', 'M', 'S', 'a', 'm', 's'], case_sensitive=False),
            default='A'
        ).upper()
        
        if user_choice == 'A':
            final_queries.append(llm_query)
            console.print(f"   ✅ Approved: '{llm_query}'")
        elif user_choice == 'M':
            modified_query = click.prompt(f"   Enter your modified query", default=llm_query)
            final_queries.append(modified_query.strip())
            console.print(f"   🔧 Modified: '{modified_query.strip()}'")
        else:  # Skip
            console.print(f"   ⏭️  Skipped: '{llm_query}'")
    
    # Option to add new queries
    while True:
        add_more = click.confirm(f"\nAdd additional query? (Current: {len(final_queries)} queries)", default=False)
        if not add_more:
            break
        new_query = click.prompt("Enter new query")
        if new_query.strip():
            final_queries.append(new_query.strip())
            console.print(f"   ➕ Added: '{new_query.strip()}'")
    
    console.print(f"\n   📝 Final queries ({len(final_queries)}): {final_queries}")
    
    console.print(f"\n3️⃣ [bold]Preferred Domains:[/bold] LLM chose {len(session_data.llm_preferred_sites)} domains")
    console.print("Review each domain individually:")
    
    final_domains = []
    for i, llm_domain in enumerate(session_data.llm_preferred_sites, 1):
        console.print(f"\n   Domain {i}: [yellow]'{llm_domain}'[/yellow]")
        user_choice = click.prompt(
            f"   [A]pprove, [M]odify, or [S]kip this domain?",
            type=click.Choice(['A', 'M', 'S', 'a', 'm', 's'], case_sensitive=False),
            default='A'
        ).upper()
        
        if user_choice == 'A':
            final_domains.append(llm_domain)
            console.print(f"   ✅ Approved: '{llm_domain}'")
        elif user_choice == 'M':
            modified_domain = click.prompt(f"   Enter your modified domain", default=llm_domain)
            final_domains.append(modified_domain.strip())
            console.print(f"   🔧 Modified: '{modified_domain.strip()}'")
        else:  # Skip
            console.print(f"   ⏭️  Skipped: '{llm_domain}'")
    
    # Option to add new domains
    while True:
        add_more = click.confirm(f"\nAdd additional domain(s)? (Current: {len(final_domains)} domains)", default=False)
        if not add_more:
            break
        new_domains_input = click.prompt("Enter new domain(s) (comma-separated for multiple)")
        if new_domains_input.strip():
            # Split by comma and clean up
            new_domains = [d.strip() for d in new_domains_input.split(",") if d.strip()]
            final_domains.extend(new_domains)
            for domain in new_domains:
                console.print(f"   ➕ Added: '{domain}'")
    
    console.print(f"\n   📝 Final domains ({len(final_domains)}): {final_domains}")
    
    console.print(f"\n4️⃣ [bold]Keywords to Include:[/bold] LLM chose {len(session_data.llm_keywords)} keywords")
    console.print("Review each keyword individually:")
    
    final_keywords = []
    for i, llm_keyword in enumerate(session_data.llm_keywords, 1):
        console.print(f"\n   Keyword {i}: [yellow]'{llm_keyword}'[/yellow]")
        user_choice = click.prompt(
            f"   [A]pprove, [M]odify, or [S]kip this keyword?",
            type=click.Choice(['A', 'M', 'S', 'a', 'm', 's'], case_sensitive=False),
            default='A'
        ).upper()
        
        if user_choice == 'A':
            final_keywords.append(llm_keyword)
            console.print(f"   ✅ Approved: '{llm_keyword}'")
        elif user_choice == 'M':
            modified_keyword = click.prompt(f"   Enter your modified keyword", default=llm_keyword)
            final_keywords.append(modified_keyword.strip())
            console.print(f"   🔧 Modified: '{modified_keyword.strip()}'")
        else:  # Skip
            console.print(f"   ⏭️  Skipped: '{llm_keyword}'")
    
    # Option to add new keywords
    while True:
        add_more = click.confirm(f"\nAdd additional keyword(s)? (Current: {len(final_keywords)} keywords)", default=False)
        if not add_more:
            break
        new_keywords_input = click.prompt("Enter new keyword(s) (comma-separated for multiple)")
        if new_keywords_input.strip():
            new_keywords = [k.strip() for k in new_keywords_input.split(",") if k.strip()]
            final_keywords.extend(new_keywords)
            for keyword in new_keywords:
                console.print(f"   ➕ Added: '{keyword}'")
    
    console.print(f"\n   📝 Final keywords ({len(final_keywords)}): {final_keywords}")

    console.print(f"\n5️⃣ [bold]Keywords to Avoid:[/bold] LLM chose {len(session_data.llm_avoid_keywords)} avoid keywords")
    console.print("Review each avoid keyword individually:")
    
    final_avoid_keywords = []
    for i, llm_avoid_keyword in enumerate(session_data.llm_avoid_keywords, 1):
        console.print(f"\n   Avoid Keyword {i}: [yellow]'{llm_avoid_keyword}'[/yellow]")
        user_choice = click.prompt(
            f"   [A]pprove, [M]odify, or [S]kip this avoid keyword?",
            type=click.Choice(['A', 'M', 'S', 'a', 'm', 's'], case_sensitive=False),
            default='A'
        ).upper()
        
        if user_choice == 'A':
            final_avoid_keywords.append(llm_avoid_keyword)
            console.print(f"   ✅ Approved: '{llm_avoid_keyword}'")
        elif user_choice == 'M':
            modified_avoid_keyword = click.prompt(f"   Enter your modified avoid keyword", default=llm_avoid_keyword)
            final_avoid_keywords.append(modified_avoid_keyword.strip())
            console.print(f"   🔧 Modified: '{modified_avoid_keyword.strip()}'")
        else:  # Skip
            console.print(f"   ⏭️  Skipped: '{llm_avoid_keyword}'")
    
    # Option to add new avoid keywords
    while True:
        add_more = click.confirm(f"\nAdd additional avoid keyword(s)? (Current: {len(final_avoid_keywords)} avoid keywords)", default=False)
        if not add_more:
            break
        new_avoid_keywords_input = click.prompt("Enter new avoid keyword(s) (comma-separated for multiple)")
        if new_avoid_keywords_input.strip():
            new_avoid_keywords = [k.strip() for k in new_avoid_keywords_input.split(",") if k.strip()]
            final_avoid_keywords.extend(new_avoid_keywords)
            for keyword in new_avoid_keywords:
                console.print(f"   ➕ Added: '{keyword}'")
    
    console.print(f"\n   📝 Final avoid keywords ({len(final_avoid_keywords)}): {final_avoid_keywords}")

    # Additional LLM plan preferences
    console.print(f"\n6️⃣ [bold]Search Settings:[/bold]")
    
    # Max pages preference
    llm_max_pages = getattr(session_data, 'llm_max_pages', None) or session_data.execution_metadata.get('max_pages', 'Unknown') if session_data.execution_metadata else 'Unknown'
    console.print(f"   Max Pages: LLM/System chose '{llm_max_pages}'")
    user_max_pages_input = click.prompt("Your preferred max pages (press Enter to approve)", default="", show_default=False)
    final_max_pages = int(user_max_pages_input) if user_max_pages_input.strip().isdigit() else llm_max_pages
    
    # Freshness preference
    llm_freshness = getattr(session_data, 'llm_freshness_days', None) or 'Unknown'
    console.print(f"   Freshness Days: LLM chose '{llm_freshness}'")
    user_freshness_input = click.prompt("Your preferred freshness days (press Enter to approve)", default="", show_default=False)
    final_freshness = int(user_freshness_input) if user_freshness_input.strip().isdigit() else llm_freshness
    
    # Include news preference
    llm_include_news = getattr(session_data, 'llm_include_news', None)
    if llm_include_news is not None:
        console.print(f"   Include News: LLM chose '{llm_include_news}'")
        final_include_news = click.confirm("Include news sources?", default=llm_include_news)
    else:
        final_include_news = None

    console.print(f"\n7️⃣ [bold]Overall Satisfaction:[/bold]")
    satisfaction = click.prompt("Rate overall satisfaction (1-5)", type=float, default=5.0)
    
    console.print(f"\n8️⃣ [bold]Notes (optional):[/bold]")
    notes = click.prompt("Any additional notes", default="", show_default=False)
    
    # Show summary of choices
    console.print(f"\n📝 [bold]Summary of Your Choices:[/bold]")
    summary_table = Table(show_header=True, header_style="bold green")
    summary_table.add_column("Field", style="cyan", width=15)
    summary_table.add_column("LLM Original", style="red", max_width=30)
    summary_table.add_column("Your Choice", style="green", max_width=30)
    summary_table.add_column("Status", style="yellow", width=12)
    
    summary_table.add_row(
        "Category",
        session_data.llm_category,
        final_category,
        "✅ Approved" if final_category == session_data.llm_category else "🔧 Modified"
    )
    summary_table.add_row(
        "Queries",
        f"{len(session_data.llm_queries)} queries",
        f"{len(final_queries)} queries",
        "✅ Approved" if final_queries == session_data.llm_queries else "🔧 Modified"
    )
    summary_table.add_row(
        "Domains",
        f"{len(session_data.llm_preferred_sites)} domains",
        f"{len(final_domains)} domains",
        "✅ Approved" if final_domains == session_data.llm_preferred_sites else "🔧 Modified"
    )
    summary_table.add_row(
        "Keywords",
        f"{len(session_data.llm_keywords)} keywords",
        f"{len(final_keywords)} keywords",
        "✅ Approved" if final_keywords == session_data.llm_keywords else "🔧 Modified"
    )
    summary_table.add_row(
        "Avoid Keywords",
        f"{len(session_data.llm_avoid_keywords)} avoid",
        f"{len(final_avoid_keywords)} avoid",
        "✅ Approved" if final_avoid_keywords == session_data.llm_avoid_keywords else "🔧 Modified"
    )
    summary_table.add_row("Max Pages", str(llm_max_pages), str(final_max_pages), "✅ Approved" if final_max_pages == llm_max_pages else "🔧 Modified")
    summary_table.add_row("Freshness", str(llm_freshness), str(final_freshness), "✅ Approved" if final_freshness == llm_freshness else "🔧 Modified")
    if final_include_news is not None:
        summary_table.add_row("Include News", str(llm_include_news), str(final_include_news), "✅ Approved" if final_include_news == llm_include_news else "🔧 Modified")
    summary_table.add_row("Satisfaction", "N/A", f"{satisfaction}/5.0", "📊 Rated")
    
    console.print(summary_table)
    
    # Confirm and save
    if click.confirm("\n💾 Save these preferences?", default=True):
        success = preference_collector.add_human_feedback(
            session_id=session_id,
            preferred_category=final_category,
            preferred_queries=final_queries,
            preferred_domains=final_domains,
            preferred_keywords=final_keywords,
            preferred_avoid_keywords=final_avoid_keywords,
            preferred_max_pages=final_max_pages if isinstance(final_max_pages, int) else None,
            preferred_freshness_days=final_freshness if isinstance(final_freshness, int) else None,
            preferred_include_news=final_include_news,
            overall_satisfaction=satisfaction,
            notes=notes.strip() if notes.strip() else None,
        )
        
        if success:
            console.print(f"\n✅ [bold green]Preferences saved successfully![/bold green]")
            console.print(f"📊 This session is now ready for GRPO training dataset")
        else:
            console.print(f"\n❌ [bold red]Failed to save preferences[/bold red]")
    else:
        console.print(f"\n❌ [bold yellow]Preferences not saved[/bold yellow]")


@preferences.command()
@click.option("--output", help="Output file path")
def export(output: Optional[str]) -> None:
    """Export training dataset for GRPO"""
    
    try:
        output_path = Path(output) if output else None
        dataset_path = preference_collector.export_training_dataset(output_path)
        
        console.print(f"✅ [bold green]Training dataset exported to:[/bold green] {dataset_path}")
        
        # Show some stats about the export
        with open(dataset_path, 'r') as f:
            line_count = sum(1 for _ in f)
        
        console.print(f"📊 Exported {line_count} training examples")
        
    except Exception as e:
        console.print(f"❌ [bold red]Export failed:[/bold red] {e}")


if __name__ == "__main__":
    main()