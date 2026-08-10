from __future__ import annotations

import re
import unittest
from pathlib import Path


class DocumentationTests(unittest.TestCase):
    def test_local_markdown_links_and_line_anchors_resolve(self) -> None:
        root = Path(__file__).resolve().parents[1]
        markdown_files = [root / "README.md", root / "README_zh.md"]
        markdown_files.extend(sorted((root / "docs").rglob("*.md")))
        pattern = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
        failures: list[str] = []

        for markdown in markdown_files:
            text = markdown.read_text(encoding="utf-8")
            for label, target in pattern.findall(text):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                path_text, _, fragment = target.partition("#")
                resolved = (markdown.parent / path_text).resolve()
                if not resolved.is_file():
                    failures.append(f"{markdown.name}: missing {target}")
                    continue
                if fragment.startswith("L") and fragment[1:].isdigit():
                    line_number = int(fragment[1:])
                    lines = resolved.read_text(encoding="utf-8").splitlines()
                    line_count = len(lines)
                    if not 1 <= line_number <= line_count:
                        failures.append(f"{markdown.name}: {target} exceeds {line_count} lines")
                        continue
                    if resolved.suffix == ".py" and label.startswith("`") and label.endswith("`"):
                        symbol = label[1:-1]
                        terminal_symbol = symbol.rsplit(".", maxsplit=1)[-1]
                        if terminal_symbol not in lines[line_number - 1]:
                            failures.append(
                                f"{markdown.name}: {target} no longer starts at `{symbol}`"
                            )

        self.assertEqual(failures, [], "\n".join(failures))

    def test_detailed_walkthrough_ranges_cover_declared_source_files_once(self) -> None:
        root = Path(__file__).resolve().parents[1]
        walkthrough_files = sorted((root / "docs" / "walkthrough").glob("[0-9]*.md"))
        range_pattern = re.compile(r"\[L(\d+)[–-]L(\d+)\]\(([^)#]+)#L(\d+)\)")
        coverage: dict[Path, dict[int, int]] = {}
        failures: list[str] = []

        for markdown in walkthrough_files:
            text = markdown.read_text(encoding="utf-8")
            for start_text, end_text, target, anchor_text in range_pattern.findall(text):
                start, end, anchor = int(start_text), int(end_text), int(anchor_text)
                source = (markdown.parent / target).resolve()
                if start != anchor:
                    failures.append(f"{markdown.name}: range L{start} starts at anchor L{anchor}")
                    continue
                if not source.is_file():
                    failures.append(f"{markdown.name}: missing walkthrough source {target}")
                    continue
                line_count = len(source.read_text(encoding="utf-8").splitlines())
                if not 1 <= start <= end <= line_count:
                    failures.append(
                        f"{markdown.name}: invalid L{start}–L{end} for {source.name} ({line_count})"
                    )
                    continue
                counts = coverage.setdefault(source, {})
                for line_number in range(start, end + 1):
                    counts[line_number] = counts.get(line_number, 0) + 1

        declared_sources = set((root / "src" / "tinygpt_forge").rglob("*.py"))
        self.assertEqual(set(coverage), declared_sources)
        for source in sorted(declared_sources):
            line_count = len(source.read_text(encoding="utf-8").splitlines())
            counts = coverage[source]
            missing = [line for line in range(1, line_count + 1) if counts.get(line, 0) == 0]
            duplicated = [line for line, count in counts.items() if count != 1]
            if missing:
                failures.append(f"{source.name}: uncovered lines {missing}")
            if duplicated:
                failures.append(f"{source.name}: multiply covered lines {sorted(duplicated)}")

        self.assertEqual(failures, [], "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
