"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"
_arxiv_lock = threading.Lock()
_last_arxiv_call = 0.0


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- Retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            delay = exc.retry_after
            if delay is None:
                delay = base * 2 ** attempt + random.uniform(0, base)
            time.sleep(max(0.0, min(cap, delay)))


def _retry_after(value):
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError):
            return None


def _request(method, url, **kwargs):
    try:
        response = httpx.request(method, url, timeout=30, **kwargs)
        if response.status_code in (429, 500, 502, 503, 504):
            raise RetryableError(f"HTTP {response.status_code}", _retry_after(response.headers.get("Retry-After")))
        response.raise_for_status()
        return response
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc


def _error(exc, secret=""):
    message = f"ERROR: {type(exc).__name__}: {exc}"
    return message.replace(secret, "[REDACTED]") if secret else message


def _compact(text, limit=600):
    return " ".join(str(text or "").split())[:limit]


def _records_or_empty(records):
    return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"


# ---- arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    global _last_arxiv_call
    try:
        terms = [term for term in re.findall(r"[\w-]+", query, re.UNICODE)
                 if term.casefold() not in {"all", "and", "or", "not"}]
        if not terms:
            return "NO RESULTS"
        search = " AND ".join(f"all:{term}" for term in terms)

        def call():
            global _last_arxiv_call
            with _arxiv_lock:
                wait = 3.0 - (time.monotonic() - _last_arxiv_call)
                if wait > 0:
                    time.sleep(wait)
                _last_arxiv_call = time.monotonic()
                return _request("GET", ARXIV_URL, params={
                    "search_query": search, "sortBy": "submittedDate", "sortOrder": "descending",
                    "max_results": max(1, min(30, int(max_results))), "start": 0,
                })

        root = ET.fromstring(with_retry(call, attempts=6, cap=60).text)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("a:entry", ns):
            raw_id = re.sub(r"v\d+$", "", entry.findtext("a:id", "", ns).split("/abs/")[-1])
            if not raw_id:
                continue
            records.append({"id": raw_id, "url": f"https://arxiv.org/abs/{raw_id}",
                            "published": entry.findtext("a:published", "", ns)[:10],
                            "title": _compact(entry.findtext("a:title", "", ns)),
                            "summary": _compact(entry.findtext("a:summary", "", ns))})
        return _records_or_empty(records)
    except Exception as exc:
        return _error(exc)


def _hf_record(item, prefer_ai=False):
    paper = item.get("paper", item) if isinstance(item, dict) else {}
    if not isinstance(paper, dict) or not paper.get("id"):
        return None
    identity = str(paper["id"])
    return {"id": identity, "url": f"https://huggingface.co/papers/{identity}",
            "published": str(paper.get("publishedAt") or item.get("publishedAt") or "")[:10],
            "title": _compact(paper.get("title") or item.get("title")),
            "summary": _compact(((paper.get("ai_summary") or item.get("ai_summary")) if prefer_ai else None)
                                or paper.get("summary") or item.get("summary")),
            "upvotes": paper.get("upvotes") or item.get("upvotes") or 0,
            "github": paper.get("githubRepo") or item.get("githubRepo") or "",
            "stars": paper.get("githubStars") or item.get("githubStars") or 0}


# ---- Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    try:
        params = {"limit": max(1, min(100, int(limit)))}
        if date:
            params["date"] = date
        data = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params).json())
        records = [_hf_record(item) for item in data]
        records = [record for record in records if record]
        if keyword:
            words = keyword.casefold()
            records = [record for record in records
                       if words in (record["title"] + " " + record["summary"]).casefold()]
        return _records_or_empty(sorted(records, key=lambda record: record["upvotes"], reverse=True))
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    try:
        if not query.strip():
            return "NO RESULTS"
        data = with_retry(lambda: _request("GET", HF_SEARCH_URL,
                             params={"q": query, "limit": max(1, min(50, int(limit)))}).json())
        return _records_or_empty([record for item in data if (record := _hf_record(item, prefer_ai=True))])
    except Exception as exc:
        return _error(exc)


def _exa_call(name, arguments):
    secret = os.getenv("EXA_API_KEY", "")
    params = {"exaApiKey": secret} if secret else None
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": name, "arguments": arguments}}

    def call():
        response = _request("POST", EXA_URL, params=params, json=request,
                            headers={"Accept": "application/json, text/event-stream"})
        payloads = []
        if "text/event-stream" in response.headers.get("content-type", ""):
            payloads = [json.loads(line[5:].strip()) for line in response.text.splitlines()
                        if line.startswith("data:") and line[5:].strip() != "[DONE]"]
        else:
            payloads = [response.json()]
        if not payloads:
            raise ValueError("empty Exa response")
        payload = payloads[-1]
        if payload.get("error"):
            error = payload["error"]
            message = str(error.get("message", error)) if isinstance(error, dict) else str(error)
            if "rate limit" in message.lower() or "too many requests" in message.lower():
                raise RetryableError("Exa rate limited", 20)
            raise ValueError(message)
        result = payload.get("result", {})
        meta = result.get("_meta") or {}
        content = "\n".join(str(block.get("text", "")) for block in result.get("content", [])
                            if block.get("type") == "text")
        marker = (str(meta) + " " + content).lower()
        rate_flag = any(key.lower().replace("_", "") in
                        {"ratelimited", "isratelimited", "ratelimitexceeded"} and bool(value)
                        for key, value in meta.items()) if isinstance(meta, dict) else False
        if ("rate limit" in marker or "rate_limit" in marker or "too many requests" in marker
                or "rate-limited" in marker or rate_flag):
            raise RetryableError("Exa rate limited", 20)
        if result.get("isError"):
            raise ValueError(content or "Exa tool error")
        return content.strip()

    try:
        return with_retry(call, attempts=6, cap=60) or "NO RESULTS"
    except Exception as exc:
        return _error(exc, secret)


# ---- Web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    try:
        if not query.strip():
            return "NO RESULTS"
        return _exa_call("web_search_exa", {"query": query,
                         "objective": objective or f"Find reliable sources about {query}",
                         "numResults": max(1, min(10, int(num_results)))})
    except Exception as exc:
        return _error(exc, os.getenv("EXA_API_KEY", ""))


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    # PSEUDO-CODE: MCP tool "web_fetch_exa" with arguments {"urls": [url]}; truncate the text to ~12000 chars
    try:
        if not url.startswith(("http://", "https://")):
            return "ERROR: invalid URL"
        return _exa_call("web_fetch_exa", {"urls": [url]})[:12000]
    except Exception as exc:
        return _error(exc, os.getenv("EXA_API_KEY", ""))


# ---- Registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n")
