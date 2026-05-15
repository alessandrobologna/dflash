from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_DRAFT_MODEL = "z-lab/Qwen3.5-4B-DFlash"
THINKING_START_TAGS = ("<think>", "<thinking>")
THINKING_END_TAGS = ("</think>", "</thinking>")
STOP_TEXT_MARKERS = ("<|im_end|>", "<|endoftext|>", "</s>")
PLAIN_THINKING_PREFIXES = (
    "here's a thinking process",
    "here is a thinking process",
    "thinking process:",
    "let's think",
)
PLAIN_FINAL_MARKERS = (
    "\nfinal answer:",
    "\nfinal:",
    "\nanswer:",
    "\nstory:",
    "\nthe story:",
)


@dataclass
class RunStats:
    name: str
    text: str
    tokens: int
    elapsed_s: float
    generation_tps: float
    peak_memory_gb: float
    acceptance_lengths: list[int]

    @property
    def measured_tps(self) -> float:
        if self.elapsed_s <= 0:
            return 0.0
        return self.tokens / self.elapsed_s


class TerminalUI:
    def __init__(self) -> None:
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            self.rich = False
            self.out = None
            self.table_cls = None
            return

        self.rich = True
        self.out = Console()
        self.table_cls = Table

    def header(self, *, args: argparse.Namespace, run_dflash: bool) -> None:
        stream_path = "dflash" if args.dflash else "target-only"
        compare_path = "off"
        if not args.no_compare:
            compare_path = "target-only" if args.dflash else "dflash"

        rows = [
            ("target", args.model),
            ("draft", args.draft_model if run_dflash else "not loaded"),
            ("stream", stream_path),
            ("compare", compare_path),
            ("tokens", str(args.max_tokens)),
            ("block", str(args.block_size)),
            ("temp", f"{args.temperature:g}"),
        ]
        if not self.rich:
            print("DFlash prompt demo")
            for key, value in rows:
                print(f"  {key:<8} {value}")
            sys.stdout.flush()
            return

        self.out.print("[magenta]DFlash[/magenta] prompt demo")
        grid = self.table_cls.grid(padding=(0, 2))
        grid.add_column(style="dim", no_wrap=True)
        grid.add_column()
        for key, value in rows:
            grid.add_row(key, value)
        self.out.print(grid)
        self.out.file.flush()

    def status(self, label: str, detail: str | None = None) -> None:
        if not self.rich:
            suffix = f": {detail}" if detail else ""
            print(f"* {label}{suffix}", flush=True)
            return
        self.out.print("* ", end="", style="cyan")
        self.out.print(label, end="")
        if detail:
            self.out.print(f" {detail}", style="dim")
        else:
            self.out.print()

    def section(self, title: str, detail: str | None = None) -> None:
        if not self.rich:
            suffix = f" {detail}" if detail else ""
            print(f"\n[{title}]{suffix}")
            return
        self.out.print()
        self.out.print("> ", end="", style="magenta")
        self.out.print(title, end="")
        if detail:
            self.out.print(f"  {detail}", style="dim")
        else:
            self.out.print()

    def stream(self, text: str, *, style: str | None = None) -> None:
        if not text:
            return
        if not self.rich:
            sys.stdout.write(text)
            sys.stdout.flush()
            return
        self.out.print(text, end="", style=style, markup=False, highlight=False)

    def stats(self, target: RunStats | None, dflash: RunStats | None, *, block_size: int, exact_match: bool) -> None:
        self.section("stats")
        if not self.rich:
            _print_plain_stats(target, dflash, block_size=block_size, exact_match=exact_match)
            return

        table = self.table_cls(show_header=True, box=None, padding=(0, 2))
        table.add_column("path", style="dim")
        table.add_column("tokens", justify="right")
        table.add_column("time", justify="right")
        table.add_column("tok/s", justify="right")
        table.add_column("peak", justify="right")
        for stats in [target, dflash]:
            if stats is None:
                continue
            table.add_row(
                stats.name,
                str(stats.tokens),
                f"{stats.elapsed_s:.3f}s",
                f"{stats.measured_tps:.2f}",
                f"{stats.peak_memory_gb:.2f} GB",
            )
        self.out.print(table)

        if target is not None and dflash is not None:
            speedup = dflash.measured_tps / max(target.measured_tps, 1e-9)
            speed_style = "green" if speedup >= 1.0 else "red"
            self.out.print("speedup   ", end="", style="dim")
            self.out.print(f"{speedup:.2f}x", style=speed_style)
            self.out.print("same text ", end="", style="dim")
            self.out.print("yes" if exact_match else "no", style="green" if exact_match else "red")

        if dflash is not None and dflash.acceptance_lengths:
            accepted = dflash.acceptance_lengths
            avg_accept = sum(accepted) / len(accepted)
            full_blocks = sum(1 for value in accepted if value >= block_size)
            self.out.print("avg accept ", end="", style="dim")
            self.out.print(f"{avg_accept:.2f} tokens/block")
            self.out.print("full block ", end="", style="dim")
            self.out.print(f"{full_blocks / len(accepted) * 100:.1f}%")


class StreamPrinter:
    def __init__(
        self,
        *,
        ui: TerminalUI,
        thinking_style: str,
        stop_markers: tuple[str, ...] = STOP_TEXT_MARKERS,
    ) -> None:
        self.ui = ui
        self.thinking_style = thinking_style
        self.stop_markers = stop_markers
        self.mode = "plain" if thinking_style == "plain" else "pending"
        self.buffer = ""
        self.stop_buffer = ""
        self.stop_hold = max((len(marker) for marker in stop_markers), default=1) - 1
        self.stopped = False
        self.plain_thinking = False

    def write(self, text: str) -> bool:
        if not text:
            return True
        if self.stopped:
            return False

        text = self.stop_buffer + text
        marker_match = self._find_any(text, self.stop_markers)
        if marker_match is not None:
            _, idx = marker_match
            self.stop_buffer = ""
            self.stopped = True
            self._write_content(text[:idx])
            return False

        if self.stop_hold > 0 and len(text) > self.stop_hold:
            safe_text = text[:-self.stop_hold]
            self.stop_buffer = text[-self.stop_hold:]
        else:
            safe_text = ""
            self.stop_buffer = text

        self._write_content(safe_text)
        return True

    def close(self) -> None:
        if not self.stopped and self.stop_buffer:
            self._write_content(self.stop_buffer)
            self.stop_buffer = ""
        if self.mode == "pending" and self.buffer:
            self._write(self.buffer)
            self.buffer = ""

    def _write_content(self, text: str) -> None:
        if self.mode == "plain":
            self._write(text)
            return
        if self.mode == "answer":
            self._write(text)
            return
        if self.mode == "pending":
            self._write_pending(text)
            return
        self._write_thinking(text)

    def _write_pending(self, text: str) -> None:
        self.buffer += text

        start = self._find_any(self.buffer.lower(), THINKING_START_TAGS)
        if start is not None:
            marker, idx = start
            before = self.buffer[:idx]
            after = self.buffer[idx + len(marker):]
            self.buffer = ""
            if before:
                self._write(before)
            self._start_thinking(plain=False)
            self._write_thinking(after)
            return

        stripped = self.buffer.lstrip().lower()
        if any(stripped.startswith(prefix) for prefix in PLAIN_THINKING_PREFIXES):
            buffered = self.buffer
            self.buffer = ""
            self._start_thinking(plain=True)
            self._write_thinking(buffered)
            return

        if len(self.buffer) >= 240:
            buffered = self.buffer
            self.buffer = ""
            self.mode = "answer"
            self._write(buffered)

    def _start_thinking(self, *, plain: bool) -> None:
        self.mode = "thinking"
        self.plain_thinking = plain
        detail = None
        if self.thinking_style == "hide":
            detail = "hidden"
        elif self.thinking_style == "dim":
            detail = "dimmed"
        self.ui.section("thinking", detail)

    def _write_thinking(self, text: str) -> None:
        lower = text.lower()
        end = self._find_any(lower, THINKING_END_TAGS)
        if end is None and self.plain_thinking:
            end = self._find_any(lower, PLAIN_FINAL_MARKERS)

        if end is None:
            self._write_visible_thinking(text)
            return

        marker, idx = end
        before = text[:idx]
        after_start = idx + len(marker)
        after = text[after_start:] if marker in THINKING_END_TAGS else text[idx:]
        self._write_visible_thinking(before)
        self._end_thinking()
        if after:
            self._write(after)

    def _write_visible_thinking(self, text: str) -> None:
        if self.thinking_style == "hide":
            return
        self._write(text)

    def _end_thinking(self) -> None:
        self.mode = "answer"
        self.ui.section("answer")

    @staticmethod
    def _find_any(text: str, markers: tuple[str, ...]) -> tuple[str, int] | None:
        found: tuple[str, int] | None = None
        for marker in markers:
            idx = text.find(marker)
            if idx >= 0 and (found is None or idx < found[1]):
                found = (marker, idx)
        return found

    def _write(self, text: str) -> None:
        style = "dim" if self.mode == "thinking" and self.thinking_style == "dim" else None
        self.ui.stream(text, style=style)


def _load_runtime() -> SimpleNamespace:
    import mlx.core as mx
    from mlx_lm import stream_generate as stream_generate_baseline
    from mlx_lm.sample_utils import make_sampler

    from dflash.model_mlx_clean import load, load_draft, stream_generate

    return SimpleNamespace(
        mx=mx,
        stream_generate_baseline=stream_generate_baseline,
        make_sampler=make_sampler,
        load=load,
        load_draft=load_draft,
        stream_generate=stream_generate,
    )


def _strip_stop_text(text: str, stop_markers: tuple[str, ...] = STOP_TEXT_MARKERS) -> str:
    found = StreamPrinter._find_any(text, stop_markers)
    if found is None:
        return text
    _, idx = found
    return text[:idx]


def _make_prompt(tokenizer: Any, user_prompt: str, *, enable_thinking: bool, raw_prompt: bool) -> str:
    if raw_prompt:
        return user_prompt

    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def _run_target(
    runtime: SimpleNamespace,
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    ui: TerminalUI,
    max_tokens: int,
    sampler: Any,
    stream: bool,
    thinking_style: str,
) -> RunStats:
    printer = StreamPrinter(ui=ui, thinking_style=thinking_style)
    if stream:
        ui.section("target-only")

    text_parts: list[str] = []
    tokens = 0
    last_tps = 0.0
    peak_memory = 0.0

    runtime.mx.synchronize()
    start = time.perf_counter()
    for response in runtime.stream_generate_baseline(
        model,
        tokenizer,
        prompt,
        max_tokens,
        sampler=sampler,
    ):
        segment = getattr(response, "text", "")
        if stream and segment:
            if not printer.write(segment):
                text_parts.append(segment)
                tokens += 1
                last_tps = float(getattr(response, "generation_tps", 0.0) or 0.0)
                peak_memory = max(peak_memory, float(getattr(response, "peak_memory", 0.0) or 0.0))
                break
        text_parts.append(segment)
        tokens += 1
        last_tps = float(getattr(response, "generation_tps", 0.0) or 0.0)
        peak_memory = max(peak_memory, float(getattr(response, "peak_memory", 0.0) or 0.0))
        if not stream and _strip_stop_text("".join(text_parts)) != "".join(text_parts):
            break
    runtime.mx.synchronize()
    elapsed = time.perf_counter() - start

    if stream:
        printer.close()
        print(flush=True)

    return RunStats(
        name="target-only",
        text=_strip_stop_text("".join(text_parts)),
        tokens=tokens,
        elapsed_s=elapsed,
        generation_tps=last_tps,
        peak_memory_gb=peak_memory,
        acceptance_lengths=[],
    )


def _run_dflash(
    runtime: SimpleNamespace,
    model: Any,
    draft: Any,
    tokenizer: Any,
    prompt: str,
    *,
    ui: TerminalUI,
    block_size: int,
    max_tokens: int,
    sampler: Any,
    stream: bool,
    thinking_style: str,
) -> RunStats:
    printer = StreamPrinter(ui=ui, thinking_style=thinking_style)
    if stream:
        ui.section("dflash")

    text_parts: list[str] = []
    tokens = 0
    last_tps = 0.0
    peak_memory = 0.0
    acceptance_lengths: list[int] = []

    runtime.mx.synchronize()
    start = time.perf_counter()
    for response in runtime.stream_generate(
        model,
        draft,
        tokenizer,
        prompt,
        block_size=block_size,
        max_tokens=max_tokens,
        sampler=sampler,
    ):
        segment = response.text
        if stream and segment:
            if not printer.write(segment):
                text_parts.append(segment)
                emitted = len(response.tokens)
                tokens += emitted
                if emitted:
                    acceptance_lengths.append(response.accepted)
                last_tps = float(response.generation_tps)
                peak_memory = max(peak_memory, float(response.peak_memory))
                break
        text_parts.append(segment)
        emitted = len(response.tokens)
        tokens += emitted
        if emitted:
            acceptance_lengths.append(response.accepted)
        last_tps = float(response.generation_tps)
        peak_memory = max(peak_memory, float(response.peak_memory))
        if not stream and _strip_stop_text("".join(text_parts)) != "".join(text_parts):
            break
    runtime.mx.synchronize()
    elapsed = time.perf_counter() - start

    if stream:
        printer.close()
        print(flush=True)

    return RunStats(
        name="dflash",
        text=_strip_stop_text("".join(text_parts)),
        tokens=tokens,
        elapsed_s=elapsed,
        generation_tps=last_tps,
        peak_memory_gb=peak_memory,
        acceptance_lengths=acceptance_lengths,
    )


def _warmup(
    runtime: SimpleNamespace,
    model: Any,
    draft: Any | None,
    tokenizer: Any,
    sampler: Any,
    block_size: int,
    warmup_tokens: int,
    *,
    include_target: bool,
    include_dflash: bool,
) -> None:
    if warmup_tokens <= 0:
        return
    warmup_prompt = tokenizer.encode("Hi")
    if include_target:
        list(runtime.stream_generate_baseline(model, tokenizer, warmup_prompt, warmup_tokens, sampler=sampler))
    if include_dflash:
        if draft is None:
            raise ValueError("DFlash warmup requires a loaded draft model")
        list(
            runtime.stream_generate(
                model,
                draft,
                tokenizer,
                warmup_prompt,
                block_size=block_size,
                max_tokens=warmup_tokens,
                sampler=sampler,
            )
        )
    runtime.mx.synchronize()


def _print_plain_stats(target: RunStats | None, dflash: RunStats | None, *, block_size: int, exact_match: bool) -> None:
    for stats in [target, dflash]:
        if stats is None:
            continue
        print(
            f"{stats.name:11} "
            f"tokens={stats.tokens:4d} "
            f"time={stats.elapsed_s:7.3f}s "
            f"tps={stats.measured_tps:7.2f} "
            f"peak={stats.peak_memory_gb:5.2f} GB"
        )

    if target is not None and dflash is not None:
        speedup = dflash.measured_tps / max(target.measured_tps, 1e-9)
        print(f"speedup     {speedup:7.2f}x")
        print(f"same text   {'yes' if exact_match else 'no'}")

    if dflash is not None and dflash.acceptance_lengths:
        accepted = dflash.acceptance_lengths
        avg_accept = sum(accepted) / len(accepted)
        full_blocks = sum(1 for value in accepted if value >= block_size)
        print(f"avg accept  {avg_accept:7.2f} tokens/block")
        print(f"full block  {full_blocks / len(accepted) * 100:7.1f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream one MLX prompt with target-only and DFlash timing.")
    parser.add_argument("--prompt", required=True, help="Prompt to send to the model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Target model (default: {DEFAULT_MODEL})")
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL, help=f"DFlash draft model (default: {DEFAULT_DRAFT_MODEL})")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--raw-prompt", action="store_true", help="Use --prompt directly without chat template")
    parser.add_argument("--dflash", action="store_true", help="Stream the DFlash run instead of the target-only run")
    parser.add_argument("--no-compare", action="store_true", help="Only run the streamed path; omit speedup comparison")
    parser.add_argument(
        "--thinking-style",
        choices=["dim", "plain", "hide"],
        default="dim",
        help="How to display detected thinking text while streaming",
    )
    parser.add_argument("--warmup-tokens", type=int, default=3, help="Warmup tokens per path before measuring")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ui = TerminalUI()
    runtime = _load_runtime()
    run_target = (not args.dflash) or (not args.no_compare)
    run_dflash = args.dflash or (not args.no_compare)

    ui.header(args=args, run_dflash=run_dflash)
    ui.status("Loading target", args.model)
    model, tokenizer = runtime.load(args.model)
    draft = None
    if run_dflash:
        ui.status("Loading draft", args.draft_model)
        draft = runtime.load_draft(args.draft_model)

    sampler = runtime.make_sampler(temp=args.temperature)
    prompt = _make_prompt(
        tokenizer,
        args.prompt,
        enable_thinking=args.enable_thinking,
        raw_prompt=args.raw_prompt,
    )

    if args.warmup_tokens > 0:
        ui.status("Warming up", f"{args.warmup_tokens} tokens per path")
    _warmup(
        runtime,
        model,
        draft,
        tokenizer,
        sampler,
        args.block_size,
        args.warmup_tokens,
        include_target=run_target,
        include_dflash=run_dflash,
    )

    target_stats: RunStats | None = None
    dflash_stats: RunStats | None = None

    if args.dflash:
        if draft is None:
            raise ValueError("--dflash requires a loaded draft model")
        dflash_stats = _run_dflash(
            runtime,
            model,
            draft,
            tokenizer,
            prompt,
            ui=ui,
            block_size=args.block_size,
            max_tokens=args.max_tokens,
            sampler=sampler,
            stream=True,
            thinking_style=args.thinking_style,
        )
        if not args.no_compare:
            ui.status("Comparing", "target-only")
            target_stats = _run_target(
                runtime,
                model,
                tokenizer,
                prompt,
                ui=ui,
                max_tokens=args.max_tokens,
                sampler=sampler,
                stream=False,
                thinking_style="plain",
            )
    else:
        target_stats = _run_target(
            runtime,
            model,
            tokenizer,
            prompt,
            ui=ui,
            max_tokens=args.max_tokens,
            sampler=sampler,
            stream=True,
            thinking_style=args.thinking_style,
        )
        if not args.no_compare:
            ui.status("Comparing", "dflash")
            if draft is None:
                raise ValueError("DFlash comparison requires a loaded draft model")
            dflash_stats = _run_dflash(
                runtime,
                model,
                draft,
                tokenizer,
                prompt,
                ui=ui,
                block_size=args.block_size,
                max_tokens=args.max_tokens,
                sampler=sampler,
                stream=False,
                thinking_style="plain",
            )

    exact_match = bool(target_stats and dflash_stats and target_stats.text == dflash_stats.text)
    ui.stats(target_stats, dflash_stats, block_size=args.block_size, exact_match=exact_match)


if __name__ == "__main__":
    main()
