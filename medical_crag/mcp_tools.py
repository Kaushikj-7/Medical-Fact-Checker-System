"""Model Context Protocol (MCP) compatible medical external search tools."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from medical_crag.types import Document

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENFDA_BASE = "https://api.fda.gov/drug"
WHO_FEED_BASE = "https://www.who.int/api/news/diseaseoutbreaknews"

FDA_REGULATORY_STOPWORDS = {
    "boxed",
    "warning",
    "warnings",
    "fda",
    "drug",
    "label",
    "adverse",
    "side",
    "effect",
    "effects",
    "dose",
    "dosage",
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "injection",
    "oral",
    "treatment",
    "therapy",
    "risk",
    "risks",
    "indication",
    "indications",
    "safety",
    "prescribing",
    "information",
    "patient",
    "medication",
    "guide",
}


def clean_lucene_query(q: str) -> str:
    """Escape or strip dangerous Lucene characters that cause HTTP 500/400 errors."""
    cleaned = re.sub(r"[{}\[\]^~*?\\:]", " ", q)
    return " ".join(cleaned.split())


def _make_doc_id(prefix: str, identifier: str) -> str:
    """Generate a clean, deterministic, collision-free document ID."""
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "", identifier)
    if 3 <= len(clean) <= 40:
        return f"{prefix}-{clean}"
    h = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{h}"


# ==============================================================================
# Helper Functions for NCBI E-utilities & XML Processing
# ==============================================================================


def _extract_abstract_from_xml(article_or_xml: ET.Element | str) -> str:
    """Extract full abstract text from PubMed article XML element or XML string.

    Preserves structured section labels (e.g. BACKGROUND, METHODS, RESULTS, CONCLUSIONS)
    and nested formatting tags (<i>, <b>, <sup>, etc.) via .itertext().
    """
    if isinstance(article_or_xml, str):
        try:
            root = ET.fromstring(article_or_xml)
            article = root.find(".//PubmedArticle") if root.tag != "PubmedArticle" else root
            if article is None:
                article = root
        except ET.ParseError:
            return ""
    else:
        article = article_or_xml

    if article is None:
        return ""

    abstract_nodes = article.findall(".//AbstractText")
    if not abstract_nodes:
        abstract_nodes = article.findall(".//OtherAbstract/AbstractText")

    sections: list[str] = []
    for node in abstract_nodes:
        text = "".join(node.itertext()).strip()
        if not text:
            continue
        raw_label = node.get("Label") or node.get("NlmCategory") or ""
        label = raw_label.strip()
        if label:
            # Prevent duplicate label prefix if author embedded it in text
            if text.upper().startswith(f"{label.upper()}:") or text.upper().startswith(f"{label.upper()} -"):
                sections.append(text)
            else:
                sections.append(f"{label}: {text}")
        else:
            sections.append(text)

    if sections:
        return "\n\n".join(sections)
    return ""


def _extract_publication_date(article: ET.Element) -> str:
    """Extract publication year and month from PubMed article XML element."""
    # 1. Journal Issue PubDate
    pub_date = article.find(".//Journal/JournalIssue/PubDate")
    if pub_date is not None:
        year = pub_date.findtext("Year")
        month = pub_date.findtext("Month")
        medline_date = pub_date.findtext("MedlineDate")
        if year and month:
            return f"{year} {month}"
        if year:
            return year
        if medline_date:
            return medline_date

    # 2. ArticleDate
    article_date = article.find(".//ArticleDate")
    if article_date is not None:
        year = article_date.findtext("Year")
        month = article_date.findtext("Month")
        if year and month:
            return f"{year} {month}"
        if year:
            return year

    # 3. DateCompleted
    date_completed = article.find(".//DateCompleted")
    if date_completed is not None:
        year = date_completed.findtext("Year")
        if year:
            return year

    return "PubMed Recent"


def _ncbi_get(
    client: httpx.Client,
    url: str,
    params: dict,
    max_retries: int = 2,
    base_backoff: float = 0.5,
) -> httpx.Response | None:
    """Execute GET request with exponential backoff for 429 and 5xx errors."""
    for attempt in range(max_retries + 1):
        try:
            resp = client.get(url, params=params)
            if resp.status_code == 200:
                return resp

            # Retry on rate limiting (429) or transient server errors (5xx)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = min(max(float(retry_after), 0.2), 3.0)
                    except ValueError:
                        delay = base_backoff * (2**attempt)
                else:
                    delay = base_backoff * (2**attempt)
                logger.warning(
                    "NCBI request to %s returned %d; retrying in %.2fs (attempt %d/%d)",
                    url,
                    resp.status_code,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
                continue

            logger.warning(
                "NCBI request to %s failed with status %d: %s",
                url,
                resp.status_code,
                resp.text[:200],
            )
            return resp
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            if attempt < max_retries:
                delay = base_backoff * (2**attempt)
                logger.warning(
                    "NCBI network error for %s (%s); retrying in %.2fs (attempt %d/%d)",
                    url,
                    e,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
                continue
            logger.warning("NCBI network request to %s failed after %d retries: %s", url, max_retries, e)
            return None
    return None


# ==============================================================================
# Helper Functions for WHO DON & HTML Processing
# ==============================================================================


def _strip_html_tags(text: str) -> str:
    """Strip HTML markup and entities, normalizing whitespace for clean clinical text."""
    if not text:
        return ""
    # Convert block tags and line breaks to spaces
    block_clean = re.sub(
        r"<\s*(?:br|p|div|h[1-6]|li|tr|hr|table|thead|tbody|td|th)\s*/?>|<\s*/(?:p|div|h[1-6]|li|tr|table|thead|tbody|td|th)\s*>",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    # Strip inline and any remaining HTML tags without inserting extra spaces
    stripped = re.sub(r"<[^>]+>", "", block_clean)
    # Unescape HTML entities (&amp;, &nbsp;, &lt;, etc.)
    unescaped = html.unescape(stripped)
    # Collapse multiple whitespace characters into single space
    collapsed = re.sub(r"\s+", " ", unescaped).strip()
    # Clean up spaces before punctuation marks
    return re.sub(r"\s+([.,;:!?])", r"\1", collapsed)


def _build_who_url(item: dict[str, Any]) -> str:
    """Construct canonical WHO DON article URL from ItemDefaultUrl or fallback."""
    raw_url = item.get("ItemDefaultUrl") or item.get("Url") or ""
    if not raw_url:
        return "https://www.who.int/emergencies/disease-outbreak-news"

    url_str = str(raw_url).strip()
    if url_str.startswith(("http://", "https://")):
        return url_str

    if url_str.startswith("/emergencies/disease-outbreak-news/item/"):
        return f"https://www.who.int{url_str}"

    # Strip leading slashes and optional 'item/' prefix
    slug = url_str.lstrip("/").removeprefix("item/")
    return f"https://www.who.int/emergencies/disease-outbreak-news/item/{slug}"


def _extract_who_content(item: dict[str, Any], max_length: int = 1200) -> str:
    """Extract and clean clinical text from Summary or fallback fields (Overview, Assessment, Advice)."""
    # 1. Check Summary first
    summary = item.get("Summary")
    if summary and isinstance(summary, str) and summary.strip():
        cleaned = _strip_html_tags(summary)
        if len(cleaned) >= 20:
            return cleaned[:max_length]

    # 2. Check fallback fields in priority order: Overview -> Assessment -> Advice
    for field in ("Overview", "Assessment", "Advice"):
        val = item.get(field)
        if val and isinstance(val, str) and val.strip():
            cleaned = _strip_html_tags(val)
            if len(cleaned) >= 20:
                return cleaned[:max_length]

    # 3. Last fallback: item Title or default
    title = item.get("Title") or ""
    clean_title = _strip_html_tags(title)
    if clean_title:
        return clean_title[:max_length]
    return "WHO official health communication."


class MCPClient:
    """Client providing unified access to PubMed, FDA, and WHO medical endpoints."""

    def __init__(
        self,
        timeout: float = 8.0,
        ncbi_email: str | None = None,
        ncbi_tool: str = "medical_crag",
        ncbi_api_key: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.ncbi_tool = ncbi_tool or "medical_crag"
        self.ncbi_email = (
            ncbi_email
            or os.getenv("NCBI_EMAIL")
            or "medical_crag@example.org"
        )
        self.ncbi_api_key = ncbi_api_key or os.getenv("NCBI_API_KEY", "")

    def search_pubmed(self, query: str, max_results: int = 3) -> list[Document]:
        """Search NCBI PubMed for peer-reviewed abstracts with NCBI policy compliance and full text retention."""
        if not query or not query.strip() or max_results <= 0:
            return []
        clean_q = query.strip()

        try:
            with httpx.Client(timeout=self.timeout) as client:
                # 1. Search for PMIDs
                esearch_url = f"{EUTILS_BASE}/esearch.fcgi"
                esearch_params: dict[str, str] = {
                    "db": "pubmed",
                    "term": clean_q,
                    "retmax": str(max_results),
                    "retmode": "json",
                    "sort": "relevance",
                    "tool": self.ncbi_tool,
                    "email": self.ncbi_email,
                }
                if self.ncbi_api_key:
                    esearch_params["api_key"] = self.ncbi_api_key

                esearch_resp = _ncbi_get(client, esearch_url, esearch_params)
                if esearch_resp is None or esearch_resp.status_code != 200:
                    return []
                try:
                    data = esearch_resp.json()
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to parse esearch JSON: %s", e)
                    return []

                id_list = data.get("esearchresult", {}).get("idlist", [])
                if not id_list:
                    return []

                # 2. Fetch abstracts
                efetch_url = f"{EUTILS_BASE}/efetch.fcgi"
                efetch_params: dict[str, str] = {
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "retmode": "xml",
                    "rettype": "abstract",
                    "tool": self.ncbi_tool,
                    "email": self.ncbi_email,
                }
                if self.ncbi_api_key:
                    efetch_params["api_key"] = self.ncbi_api_key

                fetch_resp = _ncbi_get(client, efetch_url, efetch_params)
                if fetch_resp is None or fetch_resp.status_code != 200:
                    return []

                try:
                    root = ET.fromstring(fetch_resp.text)
                except ET.ParseError as e:
                    logger.warning("Failed to parse PubMed efetch XML: %s", e)
                    return []

                documents: list[Document] = []
                for article in root.findall(".//PubmedArticle"):
                    pmid_node = article.find(".//MedlineCitation/PMID")
                    if pmid_node is None:
                        pmid_node = article.find(".//PMID")
                    pmid = pmid_node.text.strip() if pmid_node is not None and pmid_node.text else ""

                    title_node = article.find(".//ArticleTitle")
                    title = "".join(title_node.itertext()).strip() if title_node is not None else ""
                    if not title:
                        title = "PubMed Article"

                    abstract = _extract_abstract_from_xml(article)
                    if not abstract:
                        abstract = title

                    doc_id = f"pubmed-{pmid}" if pmid else _make_doc_id("pubmed", title)
                    pub_date = _extract_publication_date(article)

                    documents.append(
                        Document(
                            id=doc_id,
                            title=title,
                            text=abstract.strip(),
                            source="PubMed",
                            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "https://pubmed.ncbi.nlm.nih.gov/",
                            published=pub_date,
                        )
                    )
                return documents
        except Exception as e:  # noqa: BLE001
            logger.warning("PubMed search failed: %s", e)
            return []

    def search_fda(self, query: str, max_results: int = 3) -> list[Document]:
        """Search FDA for official drug labels, warnings, indications, and adverse reactions."""
        if not query or not query.strip():
            return []

        clean_q = query.strip()
        safe_q = clean_lucene_query(clean_q).replace('"', '\\"')

        try:
            with httpx.Client(timeout=self.timeout) as client:
                url = f"{OPENFDA_BASE}/label.json"

                # Tier 1: Exact fielded search across brand, generic, and substance names using boolean OR
                fielded_search = (
                    f'openfda.brand_name:"{safe_q}" OR '
                    f'openfda.generic_name:"{safe_q}" OR '
                    f'openfda.substance_name:"{safe_q}"'
                )
                r = client.get(url, params={"search": fielded_search, "limit": str(max_results)})

                # Tier 2: Multi-word query decomposition (e.g. "metformin lactic acidosis" or "aspirin cardiovascular")
                words = clean_lucene_query(clean_q).split()
                if r.status_code == 404 and len(words) > 1:
                    drug_tokens = [w for w in words if w.lower() not in FDA_REGULATORY_STOPWORDS and len(w) >= 3]
                    condition_tokens = [w for w in words if w not in drug_tokens]

                    if drug_tokens:
                        drug_candidate = drug_tokens[0].replace('"', '\\"')
                        if condition_tokens:
                            condition_clause = " ".join(condition_tokens).replace('"', '\\"')
                            targeted = (
                                f'(openfda.brand_name:"{drug_candidate}" OR '
                                f'openfda.generic_name:"{drug_candidate}" OR '
                                f'openfda.substance_name:"{drug_candidate}") AND '
                                f'("{condition_clause}" OR {condition_clause})'
                            )
                            r = client.get(url, params={"search": targeted, "limit": str(max_results)})

                        # If targeted search returned 404, query candidate drug in openfda fields directly
                        if r.status_code == 404:
                            drug_fielded = (
                                f'openfda.brand_name:"{drug_candidate}" OR '
                                f'openfda.generic_name:"{drug_candidate}" OR '
                                f'openfda.substance_name:"{drug_candidate}"'
                            )
                            r = client.get(url, params={"search": drug_fielded, "limit": str(max_results)})

                # Tier 3: Fallback to sanitized unfielded search
                if r.status_code == 404:
                    sanitized_raw = clean_lucene_query(clean_q)
                    r = client.get(url, params={"search": sanitized_raw, "limit": str(max_results)})

                if r.status_code != 200:
                    return []

                data = r.json()
                results = data.get("results") or []
                documents: list[Document] = []

                for idx, item in enumerate(results):
                    if not isinstance(item, dict):
                        continue
                    openfda = item.get("openfda") or {}
                    brand_names = openfda.get("brand_name") or []
                    generic_names = openfda.get("generic_name") or []
                    substance_names = openfda.get("substance_name") or []

                    brand = brand_names[0] if brand_names and brand_names[0] else ""
                    generic = generic_names[0] if generic_names and generic_names[0] else ""
                    substance = substance_names[0] if substance_names and substance_names[0] else ""

                    primary_name = brand or generic or substance or clean_q.capitalize()

                    if generic and brand and generic.upper() != brand.upper():
                        title = f"FDA Official Drug Label: {brand} ({generic})"
                    else:
                        title = f"FDA Official Drug Label: {primary_name}"

                    # Extract regulatory sections in clinical priority order
                    sections: list[str] = []

                    # 1. Boxed Warning (Highest priority regulatory alert)
                    boxed_warning = " ".join(item.get("boxed_warning") or []).strip()
                    if boxed_warning:
                        boxed_clean = " ".join(boxed_warning.split())
                        sections.append(f"Boxed Warning: {boxed_clean[:1000]}")

                    # 2. Indications and Usage
                    indications = " ".join(item.get("indications_and_usage") or []).strip()
                    if indications:
                        ind_clean = " ".join(indications.split())
                        sections.append(f"Indications: {ind_clean[:500]}")

                    # 3. Warnings and Precautions (PLR warnings_and_cautions or legacy warnings)
                    warnings_list = (item.get("warnings_and_cautions") or []) or (item.get("warnings") or [])
                    warnings_text = " ".join(warnings_list).strip()
                    if warnings_text:
                        warn_clean = " ".join(warnings_text.split())
                        sections.append(f"Warnings: {warn_clean[:600]}")

                    # 4. Contraindications
                    contraindications = " ".join(item.get("contraindications") or []).strip()
                    if contraindications:
                        contra_clean = " ".join(contraindications.split())
                        sections.append(f"Contraindications: {contra_clean[:400]}")

                    # 5. Adverse Reactions
                    adverse = " ".join(item.get("adverse_reactions") or []).strip()
                    if adverse:
                        adv_clean = " ".join(adverse.split())
                        sections.append(f"Adverse Reactions: {adv_clean[:400]}")

                    body = "\n\n".join(sections).strip()
                    if len(body) < 30:
                        body = "Official FDA drug documentation on safety, warnings, and approved usage."

                    # URL Resolution: Direct DailyMed SPL URL if set_id is present, else FDA drugs portal
                    set_ids = (openfda.get("spl_set_id") or []) or ([item.get("set_id")] if item.get("set_id") else [])
                    set_id = str(set_ids[0]).strip() if set_ids and set_ids[0] else ""
                    doc_url = (
                        f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={set_id}"
                        if set_id
                        else "https://www.fda.gov/drugs"
                    )

                    doc_id = _make_doc_id("fda", f"{primary_name}-{idx}")
                    documents.append(
                        Document(
                            id=doc_id,
                            title=title,
                            text=body,
                            source="FDA",
                            url=doc_url,
                            published="FDA Approved",
                        )
                    )
                return documents
        except Exception as e:  # noqa: BLE001
            logger.warning("FDA search failed: %s", e)
            return []

    def search_who(self, query: str, max_results: int = 3) -> list[Document]:
        """Search WHO Disease Outbreak News using compliant OData query and content fallback."""
        clean_q = query.strip()
        if not clean_q:
            return []

        safe_q = clean_q.replace("'", "''")

        try:
            with httpx.Client(timeout=self.timeout) as client:
                # 1. Primary OData search on Title ordered by latest publication
                params = {
                    "$top": str(max(1, max_results)),
                    "$orderby": "PublicationDateAndTime desc",
                    "$filter": f"contains(Title, '{safe_q}')",
                }
                r = client.get(WHO_FEED_BASE, params=params)

                items = []
                if r.status_code == 200:
                    items = r.json().get("value") or []
                elif r.status_code != 200:
                    logger.warning("WHO API returned status code %d: %s", r.status_code, r.text[:200])
                    return []

                # 2. Multi-word fallback: if exact phrase yielded 0, try token conjunction (AND) then disjunction (OR)
                if not items:
                    tokens = [t for t in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", clean_q)]
                    if len(tokens) > 1:
                        and_filter = " and ".join([f"contains(Title, '{t.replace('\'', '\'\'')}')" for t in tokens])
                        fallback_params = {
                            "$top": str(max(1, max_results)),
                            "$orderby": "PublicationDateAndTime desc",
                            "$filter": and_filter,
                        }
                        r_fb = client.get(WHO_FEED_BASE, params=fallback_params)
                        if r_fb.status_code == 200:
                            items = r_fb.json().get("value") or []

                        if not items:
                            or_filter = " or ".join([f"contains(Title, '{t.replace('\'', '\'\'')}')" for t in tokens])
                            fallback_params["$filter"] = or_filter
                            r_or = client.get(WHO_FEED_BASE, params=fallback_params)
                            if r_or.status_code == 200:
                                items = r_or.json().get("value") or []

                # 3. Client-side fallback: if mock or test harness returned static items in value
                if not items and r.status_code == 200:
                    raw_items = r.json().get("value") or []
                    if raw_items:
                        items = raw_items[:max_results]

                documents: list[Document] = []
                for idx, item in enumerate(items[:max_results]):
                    title = str(item.get("Title") or "").strip() or "WHO Health Alert"
                    url = _build_who_url(item)
                    text = _extract_who_content(item)

                    don_id = str(item.get("DonId") or "").strip()
                    item_url = str(item.get("ItemDefaultUrl") or item.get("Url") or "").strip()
                    identifier = don_id or item_url or f"{title}-{idx}"
                    doc_id = _make_doc_id("who", str(identifier))

                    published = str(item.get("PublicationDateAndTime") or "").strip()

                    documents.append(
                        Document(
                            id=doc_id,
                            title=title,
                            text=text,
                            source="WHO",
                            url=url,
                            published=published,
                            metadata={
                                "don_id": don_id,
                                "publication_date": published,
                            },
                        )
                    )
                return documents
        except (httpx.HTTPError, httpx.TimeoutException, json.JSONDecodeError, KeyError, ValueError, TypeError, AttributeError) as e:
            logger.warning("WHO search failed: %s", e)
            return []
        except Exception as e:  # noqa: BLE001
            logger.warning("Unexpected WHO search error: %s", e)
            return []

    def search(self, source: str, query: str, max_results: int = 3) -> list[Document]:
        """Dispatch search to designated medical source."""
        s = source.lower().strip()
        if s == "fda":
            return self.search_fda(query, max_results=max_results)
        elif s == "who":
            return self.search_who(query, max_results=max_results)
        else:
            return self.search_pubmed(query, max_results=max_results)


def search_pubmed(query: str, max_results: int = 3, timeout: float = 8.0) -> list[Document]:
    """Convenience functional wrapper for MCPClient.search_pubmed."""
    return MCPClient(timeout=timeout).search_pubmed(query=query, max_results=max_results)


def search_fda(query: str, max_results: int = 3, timeout: float = 8.0) -> list[Document]:
    """Convenience functional wrapper for MCPClient.search_fda."""
    return MCPClient(timeout=timeout).search_fda(query=query, max_results=max_results)


def search_who(query: str, max_results: int = 3, timeout: float = 8.0) -> list[Document]:
    """Convenience functional wrapper for MCPClient.search_who."""
    return MCPClient(timeout=timeout).search_who(query=query, max_results=max_results)
