"""Model Context Protocol (MCP) compatible medical external search tools."""

from __future__ import annotations

import hashlib
import logging
import re
import xml.etree.ElementTree as ET
from typing import List, Optional
import httpx
from medical_crag.types import Document

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENFDA_BASE = "https://api.fda.gov/drug"
WHO_FEED_BASE = "https://www.who.int/api/news/diseaseoutbreaknews"


def _make_doc_id(prefix: str, identifier: str) -> str:
    """Generate a clean, deterministic, collision-free document ID."""
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "", identifier)
    if 3 <= len(clean) <= 40:
        return f"{prefix}-{clean}"
    h = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{h}"


class MCPClient:
    """Client providing unified access to PubMed, FDA, and WHO medical endpoints."""

    def __init__(self, timeout: float = 8.0) -> None:
        self.timeout = timeout

    def search_pubmed(self, query: str, max_results: int = 3) -> List[Document]:
        """Search NCBI PubMed for peer-reviewed abstracts."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                # 1. Search for PMIDs
                esearch_url = f"{EUTILS_BASE}/esearch.fcgi"
                esearch_params = {
                    "db": "pubmed",
                    "term": query,
                    "retmax": str(max_results),
                    "retmode": "json",
                    "sort": "relevance",
                }
                r = client.get(esearch_url, params=esearch_params)
                if r.status_code != 200:
                    return []
                data = r.json()
                id_list = data.get("esearchresult", {}).get("idlist", [])
                if not id_list:
                    return []

                # 2. Fetch abstracts
                efetch_url = f"{EUTILS_BASE}/efetch.fcgi"
                efetch_params = {
                    "db": "pubmed",
                    "id": ",".join(id_list),
                    "retmode": "xml",
                    "rettype": "abstract",
                }
                fetch_resp = client.get(efetch_url, params=efetch_params)
                if fetch_resp.status_code != 200:
                    return []

                # Parse XML
                root = ET.fromstring(fetch_resp.text)
                documents: List[Document] = []
                for article in root.findall(".//PubmedArticle"):
                    pmid_node = article.find(".//MedlineCitation/PMID")
                    pmid = pmid_node.text.strip() if pmid_node is not None and pmid_node.text else ""
                    title_node = article.find(".//ArticleTitle")
                    title = "".join(title_node.itertext()).strip() if title_node is not None else "PubMed Article"
                    
                    abstract_texts = [
                        "".join(node.itertext()).strip()
                        for node in article.findall(".//AbstractText")
                    ]
                    abstract = " ".join([t for t in abstract_texts if t])
                    if not abstract:
                        abstract = title

                    doc_id = f"pubmed-{pmid}" if pmid else _make_doc_id("pubmed", title)
                    documents.append(
                        Document(
                            id=doc_id,
                            title=title,
                            text=abstract[:1200],
                            source="PubMed",
                            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "https://pubmed.ncbi.nlm.nih.gov/",
                            published="PubMed Recent",
                        )
                    )
                return documents
        except Exception as e:
            logger.warning("PubMed search failed: %s", e)
            return []

    def search_fda(self, query: str, max_results: int = 3) -> List[Document]:
        """Search FDA for official drug labels, warnings, and indications."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                url = f"{OPENFDA_BASE}/label.json"
                # Search by brand or generic name or general text
                params = {
                    "search": f'openfda.brand_name:"{query}"+openfda.generic_name:"{query}"',
                    "limit": str(max_results),
                }
                r = client.get(url, params=params)
                if r.status_code == 404:
                    # Fallback to broader search
                    params = {"search": query, "limit": str(max_results)}
                    r = client.get(url, params=params)

                if r.status_code != 200:
                    return []

                data = r.json()
                results = data.get("results", [])
                documents: List[Document] = []

                for idx, item in enumerate(results):
                    openfda = item.get("openfda", {})
                    brand_names = openfda.get("brand_name", [])
                    brand = brand_names[0] if brand_names else query.capitalize()

                    indications = " ".join(item.get("indications_and_usage", []))[:400]
                    warnings = " ".join(item.get("warnings", []) or item.get("boxed_warning", []))[:500]
                    contraindications = " ".join(item.get("contraindications", []))[:300]

                    body = f"Indications: {indications}\nWarnings: {warnings}\nContraindications: {contraindications}".strip()
                    if len(body) < 30:
                        body = "Official FDA drug documentation on safety and approved usage."

                    doc_id = _make_doc_id("fda", f"{brand}-{idx}")
                    documents.append(
                        Document(
                            id=doc_id,
                            title=f"FDA Official Drug Label: {brand}",
                            text=body,
                            source="FDA",
                            url="https://www.fda.gov/drugs",
                            published="FDA Approved",
                        )
                    )
                return documents
        except Exception as e:
            logger.warning("FDA search failed: %s", e)
            return []

    def search_who(self, query: str, max_results: int = 3) -> List[Document]:
        """Search WHO for global health alerts and guidance."""
        try:
            with httpx.Client(timeout=self.timeout) as client:
                params = {"sf_culture": "en", "$top": "20"}
                r = client.get(WHO_FEED_BASE, params=params)
                if r.status_code != 200:
                    return []

                data = r.json()
                items = data.get("value", [])
                
                # Multi-token matching across title and summary
                query_tokens = [tok for tok in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", query.lower())]
                
                def match_score(item: dict) -> int:
                    full_text = (item.get("Title", "") + " " + item.get("Summary", "")).lower()
                    if not query_tokens:
                        return 1 if query.lower() in full_text else 0
                    return sum(1 for tok in query_tokens if tok in full_text)

                ranked_items = [
                    (match_score(item), item) for item in items
                ]
                ranked_items = [it for it in ranked_items if it[0] > 0]
                ranked_items.sort(key=lambda pair: pair[0], reverse=True)

                documents: List[Document] = []
                for idx, (_, item) in enumerate(ranked_items[:max_results]):
                    url = item.get("Url", "")
                    title = item.get("Title", "WHO Health Alert")
                    doc_id = _make_doc_id("who", url or title or str(idx))
                    documents.append(
                        Document(
                            id=doc_id,
                            title=title,
                            text=item.get("Summary", "WHO official health communication.")[:800],
                            source="WHO",
                            url=f"https://www.who.int{url}" if url else "https://www.who.int",
                            published=item.get("PublicationDateAndTime", ""),
                        )
                    )
                return documents
        except Exception as e:
            logger.warning("WHO search failed: %s", e)
            return []

    def search(self, source: str, query: str, max_results: int = 3) -> List[Document]:
        """Dispatch search to designated medical source."""
        s = source.lower().strip()
        if s == "fda":
            return self.search_fda(query, max_results=max_results)
        elif s == "who":
            return self.search_who(query, max_results=max_results)
        else:
            return self.search_pubmed(query, max_results=max_results)
