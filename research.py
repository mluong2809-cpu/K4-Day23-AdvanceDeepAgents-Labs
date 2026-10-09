"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, upload

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    normalized = unicodedata.normalize("NFKD", topic.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")[:60].rstrip("-") or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return (f"Research topic: {topic}\nProduce a complete, evidence-based survey using the team workflow. "
            "Delegate at least three independent research questions. Use at least three source families and "
            "validate citations in the sandbox before finishing.")


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    calls = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if isinstance(call, dict) and call.get("name"):
                calls[call["name"]] += 1
        usage = getattr(message, "usage_metadata", None) or {}
        tokens["input"] += usage.get("input_tokens", 0) or 0
        tokens["output"] += usage.get("output_tokens", 0) or 0
    return {"model": model_name, "elapsed_s": round(elapsed, 1),
            "subagent_calls": calls["task"], "tool_calls": dict(calls), "tokens": tokens}


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes, source_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not source_bytes:
        raise RuntimeError("sandbox did not produce both report.md and sources.json")
    try:
        report = report_bytes.decode("utf-8")
        sources = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(f"invalid sandbox output: {exc}") from exc
    if not report.strip() or not isinstance(sources, list) or not sources:
        raise RuntimeError("sandbox output is empty or sources.json is invalid")
    problems = check(report, sources)
    if problems:
        raise RuntimeError("citation validation failed: " + "; ".join(problems[:5]))

    for source in sources:
        if not all(isinstance(source.get(field), str) and source[field].strip()
                   for field in ("id", "url", "title", "source")):
            raise RuntimeError("sources.json contains an incomplete source")
        family, url, identity = source["source"], source["url"], source["id"]
        if family == "arxiv" and url != f"https://arxiv.org/abs/{identity}":
            raise RuntimeError(f"arxiv URL does not match source id: {identity}")
        if family in {"hf-daily", "hf-search"} and url != f"https://huggingface.co/papers/{identity}":
            raise RuntimeError(f"Hugging Face URL does not match source id: {identity}")
        if family not in {"arxiv", "hf-daily", "hf-search", "web"}:
            raise RuntimeError(f"unknown source family: {family}")

    meta = {"topic": topic, **summarize(messages, elapsed, model_name),
            "n_sources": len(sources),
            "source_families": sorted({source["source"] for source in sources if isinstance(source, dict)
                                       and isinstance(source.get("source"), str)})}
    if meta["subagent_calls"] < 3:
        raise RuntimeError("fewer than three researcher delegations")
    if len({"arxiv", "hf-daily", "hf-search", "web"} & set(meta["source_families"])) < 3:
        raise RuntimeError("fewer than three source families")

    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(topic)
    outputs = {reports_dir / f"{stem}.md": report_bytes,
               reports_dir / f"{stem}.sources.json": source_bytes,
               reports_dir / f"{stem}.meta.json": (json.dumps(meta, ensure_ascii=False, indent=2) + "\n").encode()}
    staged = []
    try:
        for target, content in outputs.items():
            with NamedTemporaryFile(dir=reports_dir, prefix=f".{stem}-", delete=False) as temp:
                temp.write(content)
                staged.append((Path(temp.name), target))
        for temp_path, target in staged:
            os.replace(temp_path, target)
    finally:
        for temp_path, _ in staged:
            temp_path.unlink(missing_ok=True)
    return reports_dir / f"{stem}.md"


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    topic = topic.strip()
    if not topic:
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    started = time.monotonic()
    try:
        model = make_model()
        model_name = os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL") or str(model)
        with open_sandbox() as backend:
            setup = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if setup.exit_code != 0:
                raise RuntimeError(f"cannot prepare sandbox: {setup.output}")
            upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                             FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
            agent = build_lead_agent(backend, model)
            result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                  config={"recursion_limit": 1000})
            report_path = save_outputs(backend, topic, result.get("messages", []),
                                       time.monotonic() - started, model_name)
        print(f"Saved {report_path}")
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
