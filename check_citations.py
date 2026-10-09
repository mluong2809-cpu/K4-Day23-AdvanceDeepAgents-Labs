"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return citation and reference inconsistencies in a report."""
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    by_number, seen_urls = {}, set()
    for index, source in enumerate(sources, 1):
        if not isinstance(source, dict):
            problems.append(f"source entry {index} is not an object")
            continue
        number, url = source.get("n"), source.get("url")
        if type(number) is not int:
            problems.append(f"source entry {index} has no integer n")
        elif number in by_number:
            problems.append(f"duplicate source [{number}]")
        else:
            by_number[number] = source
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source entry {index} has an invalid URL")
        elif url in seen_urls:
            problems.append(f"duplicate URL: {url}")
        else:
            seen_urls.add(url)

    heading = re.search(r"^## References\s*$", report_text, re.M)
    if heading is None:
        problems.append("missing ## References")
        body, references = report_text, ""
    else:
        body, references = report_text[:heading.start()], report_text[heading.end():]

    # Ignore fenced/inline code and Markdown links before looking for citations.
    visible_lines, fence = [], None
    for line in body.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker and fence is None:
            fence = marker.group(1)[0]
            continue
        if marker and fence == marker.group(1)[0]:
            fence = None
            continue
        if fence is None:
            visible_lines.append(line)
    body = "\n".join(visible_lines)
    body = re.sub(r"`[^`\n]*`", "", body)
    body = re.sub(r"\[\d+\]\([^)]*\)", "", body)
    cited = set()
    for match in re.finditer(r"\[([0-9]+(?:\s*[,\-]\s*[0-9]+)*)\]", body):
        value = match.group(1)
        if "," in value:
            cited.update(int(part.strip()) for part in value.split(","))
        elif "-" in value:
            start, end = (int(part.strip()) for part in value.split("-", 1))
            if start <= end and end - start <= 1000:
                cited.update(range(start, end + 1))
            else:
                problems.append(f"invalid citation range [{value}]")
        else:
            cited.add(int(value))
    for number in sorted(cited - by_number.keys()):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(by_number.keys() - cited):
        problems.append(f"source [{number}] never cited")

    reference_counts = {}
    for line in references.splitlines():
        match = re.match(r"^\s*\[(\d+)\]\s+(.+)$", line)
        if not match:
            if line.strip():
                problems.append(f"invalid reference line: {line[:80]}")
            continue
        number = int(match.group(1))
        reference_counts[number] = reference_counts.get(number, 0) + 1
        urls = re.findall(r"https?://[^\s<>]+", match.group(2))
        urls = [url.rstrip(".,;)") for url in urls]
        if len(urls) != 1:
            problems.append(f"reference [{number}] must contain exactly one URL")
        elif number in by_number and urls[0] != by_number[number].get("url"):
            problems.append(f"reference [{number}] URL differs from sources.json")
        if number in by_number and by_number[number].get("title") and by_number[number]["title"] not in line:
            problems.append(f"reference [{number}] title differs from sources.json")
    for number in sorted(by_number.keys()):
        if reference_counts.get(number, 0) != 1:
            problems.append(f"source [{number}] needs exactly one reference line")
    for number in sorted(reference_counts.keys() - by_number.keys()):
        problems.append(f"reference [{number}] missing from sources.json")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
