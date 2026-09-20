"""Loading papers and patents into :class:`~tekne.schema.Document`.

Three sources, chosen so the demonstration corpus is reproducible from public
endpoints with no credentials:

``arxiv``
    Metadata API. Gives title, abstract, categories and a submission date. We
    use abstracts rather than full text throughout: they are the densest
    technology-bearing region of a paper, they are uniformly available, and
    restricting to them removes PDF extraction as a confound.

``google_patents``
    Per-publication HTML with ``itemprop`` sections for abstract, claims and
    description. Sampling is driven by the HUPD metadata table below rather than
    by search, so the sample is defined by a published frame and not by what a
    search engine chose to return.

``hupd``
    The Harvard USPTO Patent Dataset metadata feather (patentdataset.org). Used
    as a sampling frame: publication numbers, filing dates and CPC labels.

Every loader records where a document came from in ``DocMetadata`` and stores the
sha256 of the raw text, so a rerun that silently picks up different content is
detectable.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schema import DocMetadata, Document, Genre
from .normalize import normalize
from .segment import segment_paper, segment_patent

USER_AGENT = "tekne/0.3 (research prototype; technology extraction study)"
ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"
GOOGLE_PATENTS = "https://patents.google.com/patent/{pub}/en"

_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivThrottled(RuntimeError):
    """Raised when the arXiv API keeps refusing a query after backing off."""


@dataclass
class FetchStats:
    requested: int = 0
    fetched: int = 0
    failed: int = 0
    skipped: int = 0


# --- arXiv -----------------------------------------------------------------


def fetch_arxiv(
    query: str,
    *,
    max_results: int = 50,
    start: int = 0,
    sort_by: str = "submittedDate",
    delay: float = 3.0,
    retries: int = 6,
    client: Any = None,
) -> list[dict[str, Any]]:
    """Query the arXiv API and return raw records.

    The API asks for a three-second courtesy delay between requests; the default
    honours it, which is why bulk fetches here are slow by design.
    """
    from urllib.parse import urlencode

    import httpx

    # Colons stay unencoded (the endpoint accepts either form) but brackets must
    # not: a date-range query with a literal "[" returns an empty body with no
    # error, which looks exactly like a query that matched nothing.
    qs = urlencode(
        {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        },
        safe=":+",
    )
    url = f"{ARXIV_ENDPOINT}?{qs}"

    # The endpoint throttles with 406 rather than 429, in bursts: a query that
    # fails four times in a row will serve normally after a longer pause, and the
    # same query succeeds immediately from a cold client. So back off
    # exponentially rather than retrying on a fixed delay, and open a fresh
    # connection each time -- we sleep between requests anyway, so pooling buys
    # nothing. The `client` argument stays for tests that inject a transport.
    last_status: int | None = None
    for attempt in range(retries):
        owned = client is None
        conn = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60.0)
        try:
            response = conn.get(url, follow_redirects=True)
            last_status = response.status_code
            if response.status_code == 200:
                records = _parse_arxiv_feed(response.text)
                time.sleep(delay)
                return records
            if response.status_code not in (406, 429, 500, 502, 503):
                response.raise_for_status()
        finally:
            if owned:
                conn.close()
        if attempt < retries - 1:
            time.sleep(min(delay * (2**attempt), 60.0))

    raise ArxivThrottled(f"arXiv returned {last_status} for {url} after {retries} attempts")


def _parse_arxiv_feed(xml: str) -> list[dict[str, Any]]:
    from lxml import etree

    root = etree.fromstring(xml.encode("utf-8"))
    out: list[dict[str, Any]] = []
    for entry in root.findall("a:entry", _ATOM):
        raw_id = _text(entry, "a:id")
        if not raw_id:
            continue
        arxiv_id = raw_id.rsplit("/", 1)[-1]
        categories = [
            c.get("term") for c in entry.findall("a:category", _ATOM) if c.get("term")
        ]
        out.append(
            {
                "id": arxiv_id,
                "title": _clean(_text(entry, "a:title")),
                "abstract": _clean(_text(entry, "a:summary")),
                "published": _text(entry, "a:published"),
                "updated": _text(entry, "a:updated"),
                "categories": categories,
                "primary_category": categories[0] if categories else None,
                "url": raw_id,
            }
        )
    return out


def arxiv_to_document(record: dict[str, Any]) -> Document:
    title = record["title"]
    abstract = record["abstract"]
    raw = f"{title}\n\n{abstract}"
    norm = normalize(raw)
    title_len = len(normalize(title).text)
    sections = segment_paper(
        norm.text, title_len=title_len, abstract_len=len(norm.text) - title_len
    )
    return Document(
        doc_id=f"arxiv:{record['id']}",
        genre=Genre.PAPER,
        text=norm.text,
        sections=sections,
        metadata=DocMetadata(
            source="arxiv",
            source_id=record["id"],
            url=record.get("url"),
            title=title,
            date=(record.get("published") or "")[:10] or None,
            categories=record.get("categories", []),
            extra={"primary_category": record.get("primary_category")},
        ),
        raw_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


# --- patents ---------------------------------------------------------------

_PATENT_FIELDS = ("abstract", "claims", "description")
_LEADING_LABEL = re.compile(r"^(?:Abstract|Claims\s*\(\d+\)|Description)\s*", re.I)
_CLAIM_PREAMBLE = re.compile(r"^(?:What is claimed is|We claim|I claim|The invention claimed is)\s*:?\s*", re.I)


def fetch_google_patent(
    publication: str, *, delay: float = 1.5, client: Any = None
) -> dict[str, Any] | None:
    """Fetch one publication's full text from Google Patents."""
    import httpx
    from lxml import html as lxml_html

    owns_client = client is None
    client = client or httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=60.0, follow_redirects=True
    )
    try:
        response = client.get(GOOGLE_PATENTS.format(pub=publication))
        if response.status_code != 200:
            return None
        tree = lxml_html.fromstring(response.content)
        record: dict[str, Any] = {"publication_number": publication}
        for field in _PATENT_FIELDS:
            nodes = tree.xpath(f'//section[@itemprop="{field}"]')
            text = " ".join(nodes[0].text_content().split()) if nodes else ""
            record[field] = _LEADING_LABEL.sub("", text).strip()
        titles = tree.xpath('//meta[@name="DC.title"]/@content')
        record["title"] = " ".join(titles[0].split()) if titles else publication
        dates = tree.xpath('//meta[@name="DC.date"]/@content')
        record["dates"] = list(dates)
        time.sleep(delay)
        return record if record.get("abstract") or record.get("claims") else None
    finally:
        if owns_client:
            client.close()


def patent_to_document(
    record: dict[str, Any], *, filing_date: str | None = None, cpc: Sequence[str] = ()
) -> Document:
    title = record.get("title") or record["publication_number"]
    abstract = record.get("abstract") or ""
    claims = _CLAIM_PREAMBLE.sub("", record.get("claims") or "")
    description = record.get("description") or ""

    parts = [title, abstract, claims, description]
    raw = "\n\n".join(p for p in parts if p)
    norm = normalize(raw)

    # Re-derive the field boundaries in normalised space by normalising each
    # part separately; lengths compose because normalisation is per-character
    # and the joiner is whitespace, which collapses to a single space.
    lengths = [len(normalize(p).text) for p in parts]
    cursor = 0
    bounds: list[tuple[int, int]] = []
    for length in lengths:
        if length == 0:
            bounds.append((cursor, cursor))
            continue
        bounds.append((cursor, cursor + length))
        cursor += length + 1  # the single space the joiner collapsed to

    title_len = lengths[0]
    abstract_len = lengths[1]
    sections = segment_patent(
        norm.text,
        title_len=title_len,
        abstract_len=abstract_len,
        claims_range=bounds[2] if lengths[2] else None,
        description_range=bounds[3] if lengths[3] else None,
    )
    date = filing_date or (record.get("dates") or [None])[0]
    return Document(
        doc_id=f"patent:{record['publication_number']}",
        genre=Genre.PATENT,
        text=norm.text,
        sections=sections,
        metadata=DocMetadata(
            source="google_patents",
            source_id=record["publication_number"],
            url=GOOGLE_PATENTS.format(pub=record["publication_number"]),
            title=title,
            date=date,
            categories=list(cpc),
            extra={"has_claims": bool(claims), "has_description": bool(description)},
        ),
        raw_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )


def load_hupd_frame(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Read the HUPD metadata feather as a patent sampling frame."""
    import pyarrow.feather as feather

    table = feather.read_table(
        str(path),
        columns=[
            "application_number",
            "filing_date",
            "publication_number",
            "earliest_pgpub_number",
            "invention_title",
            "main_cpc_label",
            "cpc_labels",
        ],
    )
    frame = table.to_pandas()
    frame = frame[frame["earliest_pgpub_number"].notna()]
    if limit:
        frame = frame.head(limit)
    return frame.to_dict("records")


# --- serialisation ---------------------------------------------------------


def write_jsonl(documents: Sequence[Document], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(doc.model_dump_json() + "\n")


def read_jsonl(path: str | Path) -> list[Document]:
    out: list[Document] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Document.model_validate_json(line))
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _text(node: Any, path: str) -> str:
    found = node.find(path, _ATOM)
    return (found.text or "") if found is not None else ""


def _clean(text: str) -> str:
    return " ".join(text.split())
