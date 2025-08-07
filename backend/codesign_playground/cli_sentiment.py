"""
Sentiment Analysis CLI Module

Command-line interface for sentiment analysis functionality.
"""

import typer
import json
import sys
from pathlib import Path
from typing import List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.panel import Panel

from .sentiment_analyzer import SentimentAnalyzerAPI, SentimentResult
from .utils.logging import setup_logging

console = Console()


sentiment = typer.Typer(help="Sentiment Analysis CLI commands")

def setup_sentiment_logging():
    \"\"\"Setup logging for sentiment analysis CLI.\"\"\"
    setup_logging()


@sentiment.command()
def analyze(
    text: str = typer.Argument(..., help="Text to analyze"),
    format: str = typer.Option('simple', '--format', '-f', help='Output format'),
    confidence_threshold: float = typer.Option(0.0, '--confidence-threshold', '-c', help='Minimum confidence threshold')
):
    """Analyze sentiment of a single text."""
    try:
        api = SentimentAnalyzerAPI()
        result = api.analyze_text(text)
        
        if result.confidence < confidence_threshold:
            console.print(f"[yellow]Warning: Confidence {result.confidence:.3f} is below threshold {confidence_threshold}[/yellow]")
        
        _display_result(result, format)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@sentiment.command()
def batch(
    file_path: str = typer.Argument(..., help="Path to input file"),
    output: Optional[str] = typer.Option(None, '--output', '-o', help='Output file path'),
    format: str = typer.Option('json', '--format', '-f', help='Output format'),
    batch_size: int = typer.Option(100, '--batch-size', '-b', help='Batch size for processing')
):
    """Analyze sentiment for texts in a file (one per line)."""
    try:
        api = SentimentAnalyzerAPI()
        
        # Read input file
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        if not texts:
            console.print("[yellow]No texts found in input file[/yellow]")
            return
        
        # Process in batches with progress bar
        results = []
        with Progress() as progress:
            task = progress.add_task("Analyzing...", total=len(texts))
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_results = api.analyze_batch(batch_texts)
                results.extend(batch_results)
                progress.update(task, advance=len(batch_texts))
        
        # Output results
        if output:
            _save_results(results, output, format)
            console.print(f"[green]Results saved to {output}[/green]")
        else:
            _display_batch_results(results, format)
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@sentiment.command()
def stats():
    """Display sentiment analyzer statistics."""
    try:
        api = SentimentAnalyzerAPI()
        stats = api.get_stats()
        
        table = Table(title="Sentiment Analyzer Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@sentiment.command()
def demo(
    interactive: bool = typer.Option(True, '--interactive/--no-interactive', help='Interactive mode')
):
    """Run sentiment analysis demo."""
    api = SentimentAnalyzerAPI()
    
    demo_texts = [
        "I absolutely love this product! It's amazing!",
        "This is the worst thing I've ever bought. Terrible quality.",
        "The weather is okay today.",
        "Not bad, but could be better.",
        "Extremely disappointed with the service. Very poor experience."
    ]
    
    console.print(Panel("[bold blue]Sentiment Analysis Demo[/bold blue]", expand=False))
    
    if interactive:
        while True:
            text = console.input("\n[bold]Enter text to analyze (or 'quit' to exit): [/bold]")
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if text.strip():
                result = api.analyze_text(text)
                _display_result(result, 'simple')
    else:
        console.print("\n[bold]Analyzing demo texts...[/bold]\n")
        for text in demo_texts:
            result = api.analyze_text(text)
            console.print(f"[dim]Text:[/dim] {text[:60]}{'...' if len(text) > 60 else ''}")
            _display_result(result, 'simple')
            console.print()


def _display_result(result: SentimentResult, format: str):
    """Display a single sentiment result."""
    if format == 'json':
        console.print(json.dumps(result.to_dict(), indent=2))
    elif format == 'table':
        table = Table()
        table.add_column("Attribute", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Text", result.text[:100] + ('...' if len(result.text) > 100 else ''))
        table.add_row("Label", result.label.value.upper())
        table.add_row("Confidence", f"{result.confidence:.3f}")
        table.add_row("Positive Score", f"{result.scores['positive']:.3f}")
        table.add_row("Negative Score", f"{result.scores['negative']:.3f}")
        table.add_row("Neutral Score", f"{result.scores['neutral']:.3f}")
        table.add_row("Processing Time", f"{result.processing_time_ms:.2f}ms")
        
        console.print(table)
    else:  # simple
        label_color = {
            'positive': 'green',
            'negative': 'red',
            'neutral': 'yellow'
        }.get(result.label.value, 'white')
        
        console.print(f"[bold {label_color}]{result.label.value.upper()}[/bold {label_color}] "
                     f"(confidence: {result.confidence:.3f}, "
                     f"time: {result.processing_time_ms:.2f}ms)")


def _display_batch_results(results: List[SentimentResult], format: str):
    """Display batch sentiment results."""
    if format == 'json':
        console.print(json.dumps([r.to_dict() for r in results], indent=2))
    elif format == 'csv':
        # Print CSV header
        console.print("text,label,confidence,positive_score,negative_score,neutral_score,processing_time_ms")
        for result in results:
            text = result.text.replace('"', '""')  # Escape quotes
            console.print(f'"{text}",{result.label.value},{result.confidence:.3f},' +
                         f'{result.scores["positive"]:.3f},{result.scores["negative"]:.3f},' +
                         f'{result.scores["neutral"]:.3f},{result.processing_time_ms:.2f}')
    else:  # table
        table = Table(title=f"Sentiment Analysis Results ({len(results)} texts)")
        table.add_column("#", style="dim")
        table.add_column("Text", max_width=50)
        table.add_column("Label", justify="center")
        table.add_column("Confidence", justify="right")
        table.add_column("Time (ms)", justify="right")
        
        for i, result in enumerate(results, 1):
            label_color = {
                'positive': 'green',
                'negative': 'red', 
                'neutral': 'yellow'
            }.get(result.label.value, 'white')
            
            text_preview = result.text[:47] + ('...' if len(result.text) > 47 else '')
            
            table.add_row(
                str(i),
                text_preview,
                f"[{label_color}]{result.label.value.upper()}[/{label_color}]",
                f"{result.confidence:.3f}",
                f"{result.processing_time_ms:.2f}"
            )
        
        console.print(table)
        
        # Summary statistics
        positive_count = sum(1 for r in results if r.label.value == 'positive')
        negative_count = sum(1 for r in results if r.label.value == 'negative')
        neutral_count = sum(1 for r in results if r.label.value == 'neutral')
        avg_confidence = sum(r.confidence for r in results) / len(results)
        avg_time = sum(r.processing_time_ms for r in results) / len(results)
        
        summary_table = Table(title="Summary Statistics")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Texts", str(len(results)))
        summary_table.add_row("Positive", f"{positive_count} ({positive_count/len(results)*100:.1f}%)")
        summary_table.add_row("Negative", f"{negative_count} ({negative_count/len(results)*100:.1f}%)")
        summary_table.add_row("Neutral", f"{neutral_count} ({neutral_count/len(results)*100:.1f}%)")
        summary_table.add_row("Avg Confidence", f"{avg_confidence:.3f}")
        summary_table.add_row("Avg Processing Time", f"{avg_time:.2f}ms")
        
        console.print("\n")
        console.print(summary_table)


def _save_results(results: List[SentimentResult], output_path: str, format: str):
    """Save results to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        if format == 'json':
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        elif format == 'csv':
            f.write("text,label,confidence,positive_score,negative_score,neutral_score,processing_time_ms\n")
            for result in results:
                text = result.text.replace('"', '""')  # Escape quotes
                f.write(f'"{text}",{result.label.value},{result.confidence:.3f},' +
                       f'{result.scores["positive"]:.3f},{result.scores["negative"]:.3f},' +
                       f'{result.scores["neutral"]:.3f},{result.processing_time_ms:.2f}\n')
        else:
            # Default to JSON format
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    sentiment()
