#!/usr/bin/env python3
# /// script
# dependencies = ["internetarchive", "typer", "rich", "psutil", "static-ffmpeg"]
# ///
"""Download public-domain audio samples from the Internet Archive."""

import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

import internetarchive as ia
import psutil
import static_ffmpeg
import typer
from internetarchive import File as IAFile
from internetarchive import Item as IAItem
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Column, Table

console = Console()

KNOWN_GENRES = [
    "jazz",
    "classical",
    "blues",
    "folk",
    "electronic",
    "country",
    "ragtime",
    "gospel",
]

AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".ogg"}

_DESC_WIDTH = 40


def _count_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description}",
            table_column=Column(width=_DESC_WIDTH, no_wrap=True),
        ),
        BarColumn(complete_style="green", finished_style="bright_green"),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def _file_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description}",
            table_column=Column(width=_DESC_WIDTH, no_wrap=True),
        ),
        BarColumn(complete_style="green", finished_style="bright_green"),
        DownloadColumn(binary_units=True),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


def build_query(genre: str) -> str:
    return f"subject:{genre} AND mediatype:audio AND licenseurl:*publicdomain*"


def find_audio_file(item: IAItem) -> IAFile | None:
    for f in item.get_files():
        if Path(f.name).suffix.lower() in AUDIO_EXTENSIONS:
            return f
    return None


def clean_partials(output: Path) -> int:
    partials = list(output.glob("*.part"))
    for p in partials:
        p.unlink()
    return len(partials)


def get_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (ValueError, OSError):
        return None


def download_file(
    ia_file: IAFile,
    dest: Path,
    progress: Progress,
    task_id: TaskID,
    max_bytes: int,
) -> bool:
    encoded_name = quote(ia_file.name, safe="/")
    url = f"https://archive.org/download/{ia_file.identifier}/{encoded_name}"
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))
            if total > max_bytes:
                return False
            progress.update(task_id, total=total)
            chunk_size = 65536
            with open(part, "wb") as out:
                while chunk := response.read(chunk_size):
                    out.write(chunk)
                    progress.advance(task_id, len(chunk))
        part.rename(dest)
        return True
    except Exception as e:
        console.print(f"  [red]Failed:[/red] {ia_file.name}: {e}")
        if part.exists():
            part.unlink()
        return False


def fetch_genre(
    genre: str,
    target: int,
    output: Path,
    total_prog: Progress,
    total_task: TaskID,
    genre_prog: Progress,
    genre_task: TaskID,
    file_prog: Progress,
    workers: int,
    min_dur: float,
    max_dur: float,
    max_bytes: int,
) -> int:
    results = ia.search_items(
        build_query(genre),
        fields=["identifier"],
        params={"rows": target * 4},
    )
    identifiers = [r["identifier"] for r in results]

    if not identifiers:
        console.print(f"  [yellow]No results found for genre '{genre}'[/yellow]")
        return 0

    downloaded = 0

    def process(identifier: str) -> bool:
        try:
            item = ia.get_item(identifier)
            audio_file = find_audio_file(item)
            if not audio_file:
                return False

            dest = output / f"{identifier}__{audio_file.name}"
            if dest.exists():
                dur = get_duration(dest)
                if dur is not None and not (min_dur <= dur <= max_dur):
                    dest.unlink()
                    return False
                return True

            name = (
                audio_file.name[: _DESC_WIDTH - 7] + "..."
                if len(audio_file.name) > _DESC_WIDTH - 7
                else audio_file.name
            )
            dl_task = file_prog.add_task(f"    [dim]{name}[/dim]", total=None)
            ok = download_file(audio_file, dest, file_prog, dl_task, max_bytes)

            file_prog.remove_task(dl_task)

            if ok:
                dur = get_duration(dest)
                if dur is None or not (min_dur <= dur <= max_dur):
                    dest.unlink()
                    return False

            return ok
        except Exception as e:
            console.print(f"  [red]Error fetching {identifier}:[/red] {e}")
            return False

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, ident): ident for ident in identifiers}
        for future in as_completed(futures):
            if future.result():
                downloaded += 1
                genre_prog.advance(genre_task, 1)
                total_prog.advance(total_task, 1)
            if downloaded >= target:
                for pending in futures:
                    pending.cancel()
                break

    return downloaded


app = typer.Typer(help="Download public-domain audio samples from Internet Archive.")


@app.command()
def main(
    songs: Annotated[
        int,
        typer.Option("--songs", "-x", help="Total songs to download"),
    ] = 50,
    genres: Annotated[
        int,
        typer.Option("--genres", "-y", help="Number of genres to sample from"),
    ] = 5,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("audio_raw"),
    genre_list: Annotated[
        str | None,
        typer.Option("--genre-list", help="Comma-separated genres to use"),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Parallel download threads"),
    ] = max(1, (psutil.cpu_count(logical=False) or 2) // 2),
    min_length: Annotated[
        float,
        typer.Option("--min-length", help="Minimum track duration in seconds"),
    ] = 30.0,
    max_length: Annotated[
        float,
        typer.Option("--max-length", help="Maximum track duration in seconds"),
    ] = 300.0,
    max_size: Annotated[
        int,
        typer.Option("--max-size", help="Maximum file size in bytes"),
    ] = 100 * 1024 * 1024,
) -> None:
    """Download X songs spread across Y genres from the Internet Archive."""
    static_ffmpeg.add_paths()
    output.mkdir(parents=True, exist_ok=True)

    if n := clean_partials(output):
        console.print(
            f"[yellow]Cleaned {n} partial download(s) from a previous run.[/yellow]\n"
        )

    selected_genres: list[str]
    if genre_list:
        selected_genres = [g.strip() for g in genre_list.split(",")][:genres]
    else:
        selected_genres = KNOWN_GENRES[:genres]

    per_genre = max(1, songs // len(selected_genres))
    remainder = songs - per_genre * len(selected_genres)
    targets = [
        per_genre + (1 if i < remainder else 0) for i in range(len(selected_genres))
    ]

    table = Table(title="Download Plan", show_header=True)
    table.add_column("Genre")
    table.add_column("Target", justify="right")
    for g, t in zip(selected_genres, targets):
        table.add_row(g, str(t))
    console.print(table)
    console.print(
        f"  Workers: [cyan]{workers}[/cyan]  "
        f"Duration: [cyan]{min_length:.0f}s[/cyan] - [cyan]{max_length:.0f}s[/cyan]  "
        f"Max size: [cyan]{max_size // (1024 * 1024)}MiB[/cyan]\n"
    )

    total_prog = _count_progress()
    genre_prog = _count_progress()
    file_prog = _file_progress()

    with Live(
        Group(total_prog, genre_prog, file_prog), console=console, refresh_per_second=15
    ):
        total_task = total_prog.add_task("[bold]Total[/bold]", total=songs)

        for genre, target in zip(selected_genres, targets):
            genre_task = genre_prog.add_task(f"  [cyan]{genre}[/cyan]", total=target)
            fetch_genre(
                genre,
                target,
                output,
                total_prog,
                total_task,
                genre_prog,
                genre_task,
                file_prog,
                workers,
                min_length,
                max_length,
                max_size,
            )

    if n := clean_partials(output):
        console.print(f"[yellow]Cleaned {n} partial download(s).[/yellow]")

    console.print(
        f"\n[bold green]Done.[/bold green] Files saved to [cyan]{output}[/cyan]"
    )


if __name__ == "__main__":
    app()
