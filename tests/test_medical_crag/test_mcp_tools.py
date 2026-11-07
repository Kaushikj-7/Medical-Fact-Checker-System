import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import httpx

from medical_crag.mcp_tools import (
    MCPClient,
    _build_who_url,
    _extract_abstract_from_xml,
    _extract_who_content,
    _ncbi_get,
    _strip_html_tags,
    clean_lucene_query,
    search_fda,
    search_pubmed,
    search_who,
)


def test_pubmed_search_success():
    client = MCPClient()

    fake_esearch_json = {"esearchresult": {"idlist": ["998877"]}}
    fake_efetch_xml = """<?xml version="1.0"?>
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>998877</PMID>
                <Article>
                    <ArticleTitle>Turmeric and Arthritis Clinical Trial</ArticleTitle>
                    <Abstract>
                        <AbstractText>Curcumin demonstrated mild anti-inflammatory efficacy in joint pain.</AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>
    """

    with patch("httpx.Client.get") as mock_get:
        mock_resp1 = patch("httpx.Response").start()
        mock_resp1.status_code = 200
        mock_resp1.json.return_value = fake_esearch_json

        mock_resp2 = patch("httpx.Response").start()
        mock_resp2.status_code = 200
        mock_resp2.text = fake_efetch_xml

        mock_get.side_effect = [mock_resp1, mock_resp2]

        docs = client.search_pubmed("turmeric arthritis", max_results=1)
        assert len(docs) == 1
        assert docs[0].id == "pubmed-998877"
        assert "Turmeric" in docs[0].title
        assert docs[0].source == "PubMed"
        assert docs[0].url == "https://pubmed.ncbi.nlm.nih.gov/998877/"


def test_pubmed_search_empty():
    client = MCPClient()
    fake_esearch_empty = {"esearchresult": {"idlist": []}}

    with patch("httpx.Client.get") as mock_get:
        mock_resp = patch("httpx.Response").start()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_esearch_empty
        mock_get.return_value = mock_resp

        docs = client.search_pubmed("nonexistent_rare_condition_xyz")
        assert docs == []


def test_fda_search_success():
    client = MCPClient()
    fake_fda_json = {
        "results": [
            {
                "openfda": {"brand_name": ["Lipitor"]},
                "warnings": ["Liver enzyme abnormalities have been observed."],
                "indications_and_usage": ["Indicated to reduce cholesterol."],
            }
        ]
    }

    with patch("httpx.Client.get") as mock_get:
        mock_resp = patch("httpx.Response").start()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_fda_json
        mock_get.return_value = mock_resp

        docs = client.search_fda("lipitor")
        assert len(docs) == 1
        assert docs[0].source == "FDA"
        assert "Liver enzyme" in docs[0].text


def test_mcp_dispatch_source():
    client = MCPClient()
    with patch.object(client, "search_pubmed", return_value=[]) as mock_pubmed:
        client.search("pubmed", "vitamin c")
        mock_pubmed.assert_called_once_with("vitamin c", max_results=3)

    with patch.object(client, "search_fda", return_value=[]) as mock_fda:
        client.search("fda", "metformin")
        mock_fda.assert_called_once_with("metformin", max_results=3)

    with patch.object(client, "search_who", return_value=[]) as mock_who:
        client.search("who", "cholera", max_results=2)
        mock_who.assert_called_once_with("cholera", max_results=2)


# ==============================================================================
# Expanded Unit Tests for PubMed Enhancements
# ==============================================================================


def test_pubmed_structured_abstract_with_section_labels():
    xml_content = """<PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345678</PMID>
                <Article>
                    <ArticleTitle>Clinical Study of Compound X</ArticleTitle>
                    <Abstract>
                        <AbstractText Label="BACKGROUND">Compound X is a novel kinase inhibitor.</AbstractText>
                        <AbstractText Label="METHODS">Double-blind randomized controlled trial with 400 patients.</AbstractText>
                        <AbstractText Label="RESULTS">Significant progression-free survival benefit observed (p &lt; 0.001).</AbstractText>
                        <AbstractText Label="CONCLUSIONS">Compound X represents a viable clinical option.</AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>"""

    abstract = _extract_abstract_from_xml(xml_content)
    assert "BACKGROUND: Compound X is a novel kinase inhibitor." in abstract
    assert "METHODS: Double-blind randomized controlled trial with 400 patients." in abstract
    assert "RESULTS: Significant progression-free survival benefit observed" in abstract
    assert "CONCLUSIONS: Compound X represents a viable clinical option." in abstract
    assert "\n\n" in abstract


def test_pubmed_structured_abstract_avoids_duplicate_label_prefix():
    xml_content = """<PubmedArticle>
        <MedlineCitation>
            <PMID>87654321</PMID>
            <Article>
                <ArticleTitle>Trial of Therapy Y</ArticleTitle>
                <Abstract>
                    <AbstractText Label="OBJECTIVE">OBJECTIVE: To assess safety.</AbstractText>
                    <AbstractText Label="RESULTS">RESULTS - Favorable outcomes were seen.</AbstractText>
                </Abstract>
            </Article>
        </MedlineCitation>
    </PubmedArticle>"""

    root = ET.fromstring(xml_content)
    abstract = _extract_abstract_from_xml(root)
    assert "OBJECTIVE: OBJECTIVE:" not in abstract
    assert "OBJECTIVE: To assess safety." in abstract
    assert "RESULTS: RESULTS" not in abstract
    assert "RESULTS - Favorable outcomes were seen." in abstract


def test_pubmed_full_abstract_retention_no_truncation():
    client = MCPClient()
    long_abstract = "Clinical finding paragraph. " * 100  # ~2,800 chars
    fake_esearch_json = {"esearchresult": {"idlist": ["112233"]}}
    fake_efetch_xml = f"""<?xml version="1.0"?>
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>112233</PMID>
                <Article>
                    <ArticleTitle>Long Abstract Study</ArticleTitle>
                    <Abstract>
                        <AbstractText>{long_abstract.strip()}</AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>
    """

    with patch("httpx.Client.get") as mock_get:
        mock_resp1 = MagicMock()
        mock_resp1.status_code = 200
        mock_resp1.json.return_value = fake_esearch_json

        mock_resp2 = MagicMock()
        mock_resp2.status_code = 200
        mock_resp2.text = fake_efetch_xml

        mock_get.side_effect = [mock_resp1, mock_resp2]

        docs = client.search_pubmed("long study", max_results=1)
        assert len(docs) == 1
        # Asserts full retention without 1200 char truncation cutoff
        assert len(docs[0].text) > 2500
        assert docs[0].url == "https://pubmed.ncbi.nlm.nih.gov/112233/"


def test_pubmed_ncbi_params_and_retry_backoff():
    client = MCPClient(ncbi_email="test@clinical.org", ncbi_tool="test_tool", ncbi_api_key="ncbi_key_123")

    fake_esearch_json = {"esearchresult": {"idlist": ["556677"]}}
    fake_efetch_xml = """<PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>556677</PMID>
                <Article>
                    <ArticleTitle>NCBI Compliance Test</ArticleTitle>
                    <Journal><JournalIssue><PubDate><Year>2026</Year><Month>Jan</Month></PubDate></JournalIssue></Journal>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>"""

    # Test retry on HTTP 429
    with patch("httpx.Client.get") as mock_get, patch("time.sleep"):
        resp_429 = MagicMock()
        resp_429.status_code = 429
        resp_429.headers = {"Retry-After": "0.1"}

        resp_esearch = MagicMock()
        resp_esearch.status_code = 200
        resp_esearch.json.return_value = fake_esearch_json

        resp_efetch = MagicMock()
        resp_efetch.status_code = 200
        resp_efetch.text = fake_efetch_xml

        # 429 then 200 for esearch, then 200 for efetch
        mock_get.side_effect = [resp_429, resp_esearch, resp_efetch]

        docs = client.search_pubmed("ncbi compliance", max_results=1)
        assert len(docs) == 1
        assert docs[0].published == "2026 Jan"

        # Check compliance parameters in calls
        first_call_params = mock_get.call_args_list[0][1]["params"]
        assert first_call_params["tool"] == "test_tool"
        assert first_call_params["email"] == "test@clinical.org"
        assert first_call_params["api_key"] == "ncbi_key_123"


def test_ncbi_get_retry_exhaustion():
    mock_client = MagicMock()
    mock_resp_503 = MagicMock()
    mock_resp_503.status_code = 503
    mock_resp_503.text = "Service Unavailable"
    mock_client.get.return_value = mock_resp_503

    with patch("time.sleep"):
        resp = _ncbi_get(mock_client, "https://eutils.ncbi.nlm.nih.gov", {}, max_retries=2)
        assert resp is not None
        assert resp.status_code == 503
        assert mock_client.get.call_count == 3


# ==============================================================================
# Expanded Unit Tests for openFDA Enhancements
# ==============================================================================


def test_fda_boxed_warning_independent_extraction():
    client = MCPClient()
    fake_fda_json = {
        "results": [
            {
                "openfda": {
                    "brand_name": ["GLUCOPHAGE"],
                    "generic_name": ["METFORMIN HYDROCHLORIDE"],
                    "spl_set_id": ["8a946b5a-1b5e-42c2-8012-ef6587c67c55"],
                },
                "boxed_warning": [
                    "WARNING: LACTIC ACIDOSIS. Post-marketing cases of metformin-associated lactic acidosis have resulted in death, hypothermia, hypotension, and resistant bradyarrhythmias."
                ],
                "indications_and_usage": [
                    "GLUCOPHAGE is an oral antihyperglycemic drug indicated as an adjunct to diet and exercise to improve glycemic control in type 2 diabetes mellitus."
                ],
                "warnings_and_cautions": [
                    "Vitamin B12 levels: Metformin may decrease vitamin B12 levels. Measure hematologic parameters annually."
                ],
                "contraindications": [
                    "Severe renal impairment (eGFR below 30 mL/min/1.73 m2). Known hypersensitivity to metformin."
                ],
                "adverse_reactions": [
                    "Most common adverse reactions are diarrhea, nausea/vomiting, flatulence, and abdominal discomfort."
                ],
            }
        ]
    }

    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_fda_json
        mock_get.return_value = mock_resp

        docs = client.search_fda("metformin")
        assert len(docs) == 1
        doc = docs[0]

        # Verify Boxed Warning is independently preserved
        assert "Boxed Warning: WARNING: LACTIC ACIDOSIS" in doc.text
        # Verify Indications and Warnings are preserved alongside Boxed Warning
        assert "Indications: GLUCOPHAGE is an oral antihyperglycemic" in doc.text
        assert "Warnings: Vitamin B12 levels" in doc.text
        assert "Contraindications: Severe renal impairment" in doc.text
        assert "Adverse Reactions: Most common adverse reactions" in doc.text

        # Verify Dual Naming title
        assert doc.title == "FDA Official Drug Label: GLUCOPHAGE (METFORMIN HYDROCHLORIDE)"

        # Verify direct DailyMed SPL URL
        assert doc.url == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=8a946b5a-1b5e-42c2-8012-ef6587c67c55"


def test_fda_search_404_handling():
    client = MCPClient()
    mock_resp_404 = MagicMock()
    mock_resp_404.status_code = 404
    mock_resp_404.json.return_value = {"error": {"code": "NOT_FOUND", "message": "No matches found!"}}

    with patch("httpx.Client.get", return_value=mock_resp_404):
        docs = client.search_fda("nonexistentdrugxyz999888777")
        assert isinstance(docs, list)
        assert docs == []


def test_fda_lucene_sanitizer_and_query_decomposition():
    raw_query = 'Drug [500mg] {oral} (active)*: "safe"^3 ~2 \\test'
    cleaned = clean_lucene_query(raw_query)
    assert "[" not in cleaned
    assert "]" not in cleaned
    assert "{" not in cleaned
    assert "}" not in cleaned
    assert "^" not in cleaned
    assert "~" not in cleaned
    assert "*" not in cleaned
    assert "\\" not in cleaned
    assert ":" not in cleaned

    # Test multi-word decomposition in search_fda
    client = MCPClient()
    resp_404 = MagicMock()
    resp_404.status_code = 404

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {
        "results": [
            {
                "openfda": {"brand_name": ["Zituvimet"], "generic_name": ["METFORMIN HYDROCHLORIDE"]},
                "boxed_warning": ["Boxed warning on lactic acidosis."],
            }
        ]
    }

    with patch("httpx.Client.get") as mock_get:
        # Tier 1 returns 404, Tier 2 targeted search returns 200
        mock_get.side_effect = [resp_404, resp_200]

        docs = client.search_fda("metformin lactic acidosis")
        assert len(docs) == 1
        assert "Zituvimet" in docs[0].title
        assert "Boxed Warning: Boxed warning on lactic acidosis." in docs[0].text


# ==============================================================================
# Expanded Unit Tests for WHO DON Enhancements
# ==============================================================================


def test_who_search_success_with_overview_fallback():
    client = MCPClient()
    fake_who_json = {
        "value": [
            {
                "Title": "Avian Influenza A(H5N1) - Viet Nam",
                "Summary": "",  # Empty summary
                "Overview": "<p>A fatal human case of <strong>avian influenza A(H5N1)</strong> has been confirmed.</p>",
                "ItemDefaultUrl": "/item/2026-DON600",
                "PublicationDateAndTime": "2026-03-15T12:00:00Z",
                "DonId": "2026-DON600",
            }
        ]
    }

    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_who_json
        mock_get.return_value = mock_resp

        docs = client.search_who("avian influenza", max_results=1)
        assert len(docs) == 1
        assert docs[0].source == "WHO"
        assert docs[0].id == "who-2026-DON600"
        assert docs[0].url == "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON600"
        assert "fatal human case of avian influenza A(H5N1)" in docs[0].text
        assert "<strong>" not in docs[0].text


def test_who_search_empty_query_and_errors():
    client = MCPClient()
    # 1. Empty query
    assert client.search_who("   ") == []

    # 2. HTTP 500 error
    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"
        mock_get.return_value = mock_resp
        assert client.search_who("cholera") == []

    # 3. Connection Timeout
    with patch("httpx.Client.get", side_effect=httpx.TimeoutException("Timed out")):
        assert client.search_who("cholera") == []


def test_who_html_stripping_and_url_construction():
    html_input = "<p>Outbreak alert.<br/><b>Severe</b> cases detected &amp; hospitalized.</p><div><span>Notice , please .</span></div>"
    stripped = _strip_html_tags(html_input)
    assert "Outbreak alert. Severe cases detected & hospitalized. Notice, please." == stripped
    assert "<" not in stripped
    assert ">" not in stripped

    # URL variations
    assert _build_who_url({"ItemDefaultUrl": "https://who.int/custom"}) == "https://who.int/custom"
    assert (
        _build_who_url({"ItemDefaultUrl": "/emergencies/disease-outbreak-news/item/2026-DON100"})
        == "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON100"
    )
    assert (
        _build_who_url({"ItemDefaultUrl": "/item/cholera-2024"})
        == "https://www.who.int/emergencies/disease-outbreak-news/item/cholera-2024"
    )
    assert (
        _build_who_url({"ItemDefaultUrl": "/2026-DON623"})
        == "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON623"
    )
    assert _build_who_url({}) == "https://www.who.int/emergencies/disease-outbreak-news"


def test_who_content_fallback_hierarchy():
    # 1. Non-empty Summary
    item1 = {"Summary": "Official summary text explaining outbreak."}
    assert _extract_who_content(item1) == "Official summary text explaining outbreak."

    # 2. Empty Summary -> Overview
    item2 = {"Summary": "", "Overview": "Overview section detailing 500 hospitalized patients."}
    assert _extract_who_content(item2) == "Overview section detailing 500 hospitalized patients."

    # 3. Empty Summary & Overview -> Assessment
    item3 = {"Summary": "", "Overview": "", "Assessment": "Assessment confirms high regional risk of transmission."}
    assert _extract_who_content(item3) == "Assessment confirms high regional risk of transmission."

    # 4. Empty Summary, Overview, Assessment -> Advice
    item4 = {"Summary": "", "Overview": "", "Assessment": "", "Advice": "Advice urges enhanced clinical surveillance."}
    assert _extract_who_content(item4) == "Advice urges enhanced clinical surveillance."

    # 5. Empty body -> Title
    item5 = {"Summary": "", "Overview": "", "Title": "Outbreak of Marburg Virus"}
    assert _extract_who_content(item5) == "Outbreak of Marburg Virus"


def test_convenience_functions():
    with patch("medical_crag.mcp_tools.MCPClient.search_pubmed", return_value=[]) as mock_pubmed, \
         patch("medical_crag.mcp_tools.MCPClient.search_fda", return_value=[]) as mock_fda, \
         patch("medical_crag.mcp_tools.MCPClient.search_who", return_value=[]) as mock_who:

        search_pubmed("term1")
        mock_pubmed.assert_called_once_with(query="term1", max_results=3)

        search_fda("term2")
        mock_fda.assert_called_once_with(query="term2", max_results=3)

        search_who("term3")
        mock_who.assert_called_once_with(query="term3", max_results=3)


# ==============================================================================
# Robustness & Null Safety Expansion Tests (Milestone M3 Iteration 2)
# ==============================================================================


def test_pubmed_abstract_whitespace_label_handling():
    """Verify that whitespace-only XML Label and NlmCategory attributes do not emit dangling colon prefixes."""
    # 1. Direct XML extraction verification
    xml_content = """<PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>11002233</PMID>
                <Article>
                    <ArticleTitle>Label Edge Case Trial</ArticleTitle>
                    <Abstract>
                        <AbstractText Label="   ">First section with whitespace-only label attribute.</AbstractText>
                        <AbstractText NlmCategory=" \t\n ">Second section with whitespace NlmCategory.</AbstractText>
                        <AbstractText Label="METHODS">Third section with valid label.</AbstractText>
                        <AbstractText Label="RESULTS">RESULTS: Fourth section with embedded label prefix.</AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    </PubmedArticleSet>"""

    abstract = _extract_abstract_from_xml(xml_content)
    assert not abstract.startswith(": ")
    assert ": First section" not in abstract
    assert ": Second section" not in abstract
    assert "First section with whitespace-only label attribute." in abstract
    assert "Second section with whitespace NlmCategory." in abstract
    assert "METHODS: Third section with valid label." in abstract
    assert "RESULTS: Fourth section with embedded label prefix." in abstract
    assert "RESULTS: RESULTS:" not in abstract

    # 2. Client search_pubmed integration verification
    client = MCPClient()
    fake_esearch_json = {"esearchresult": {"idlist": ["11002233"]}}

    with patch("httpx.Client.get") as mock_get:
        resp_esearch = MagicMock()
        resp_esearch.status_code = 200
        resp_esearch.json.return_value = fake_esearch_json

        resp_efetch = MagicMock()
        resp_efetch.status_code = 200
        resp_efetch.text = xml_content

        mock_get.side_effect = [resp_esearch, resp_efetch]

        docs = client.search_pubmed("label whitespace test", max_results=1)
        assert len(docs) == 1
        assert not docs[0].text.startswith(": ")
        assert "First section with whitespace-only label attribute." in docs[0].text


def test_fda_null_field_safety_and_missing_sections():
    """Verify openFDA search does not raise TypeError or drop documents when section arrays are explicit null."""
    client = MCPClient()
    fake_fda_json = {
        "results": [
            {
                "openfda": {
                    "brand_name": ["Metformin"],
                    "generic_name": None,
                    "substance_name": None,
                    "spl_set_id": None,
                },
                "boxed_warning": None,
                "indications_and_usage": None,
                "warnings_and_cautions": None,
                "warnings": None,
                "contraindications": None,
                "adverse_reactions": None,
                "set_id": None,
            }
        ]
    }

    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_fda_json
        mock_get.return_value = mock_resp

        docs = client.search_fda("metformin")
        assert len(docs) == 1
        doc = docs[0]
        assert doc.id == "fda-Metformin-0"
        assert doc.title == "FDA Official Drug Label: Metformin"
        assert doc.source == "FDA"
        assert doc.url == "https://www.fda.gov/drugs"
        assert doc.published == "FDA Approved"
        assert doc.text == "Official FDA drug documentation on safety, warnings, and approved usage."


def test_fda_null_openfda_object():
    """Verify openFDA search handles unindexed records where openfda is explicit null without AttributeError."""
    client = MCPClient()
    fake_fda_json = {
        "results": [
            {
                "openfda": None,
                "warnings": ["Use with caution in renal failure."],
                "boxed_warning": None,
            }
        ]
    }

    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_fda_json
        mock_get.return_value = mock_resp

        docs = client.search_fda("aspirin")
        assert len(docs) == 1
        doc = docs[0]
        assert doc.id == "fda-Aspirin-0"
        assert doc.title == "FDA Official Drug Label: Aspirin"
        assert "Warnings: Use with caution in renal failure." in doc.text
        assert doc.url == "https://www.fda.gov/drugs"


def test_fda_partial_null_sections_with_valid_boxed_warning():
    """Verify openFDA search correctly extracts populated sections while ignoring null sections."""
    client = MCPClient()
    fake_fda_json = {
        "results": [
            {
                "openfda": {
                    "brand_name": ["Warfarin Sodium"],
                    "generic_name": ["WARFARIN"],
                    "spl_set_id": ["b3b137d1-9f93-4a6c-9ea6-f709d73d6b63"],
                },
                "boxed_warning": ["WARNING: BLEEDING RISK. Warfarin sodium can cause major or fatal bleeding."],
                "indications_and_usage": None,
                "warnings_and_cautions": None,
                "warnings": ["Periodic determination of INR is recommended."],
                "contraindications": None,
                "adverse_reactions": ["Hemorrhage is the most common adverse reaction."],
            }
        ]
    }

    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_fda_json
        mock_get.return_value = mock_resp

        docs = client.search_fda("warfarin")
        assert len(docs) == 1
        doc = docs[0]
        assert doc.id == "fda-WarfarinSodium-0"
        assert doc.title == "FDA Official Drug Label: Warfarin Sodium (WARFARIN)"
        assert "Boxed Warning: WARNING: BLEEDING RISK" in doc.text
        assert "Warnings: Periodic determination of INR is recommended." in doc.text
        assert "Adverse Reactions: Hemorrhage is the most common" in doc.text
        assert "Indications:" not in doc.text
        assert "Contraindications:" not in doc.text


def test_who_document_id_disambiguation_duplicate_and_missing_titles():
    """Verify WHO search generates unique, deterministic document IDs when items have duplicate or missing titles."""
    client = MCPClient()
    fake_who_json = {
        "value": [
            {
                "Title": "Disease Outbreak Reported",
                "Summary": "First report on regional transmission.",
            },
            {
                "Title": "Disease Outbreak Reported",
                "Summary": "Follow-up report on confirmed cases.",
            },
            {
                # Missing Title entirely (will default to WHO Health Alert)
                "Summary": "Third notification without title.",
            },
        ]
    }

    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_who_json
        mock_get.return_value = mock_resp

        docs = client.search_who("outbreak", max_results=3)
        assert len(docs) == 3

        # Assert no ID collisions across duplicate or defaulted titles
        assert docs[0].id != docs[1].id
        assert docs[1].id != docs[2].id
        assert docs[0].id != docs[2].id
        assert len({d.id for d in docs}) == 3

        # Assert fallback formatting incorporates index disambiguation
        assert docs[0].id == "who-DiseaseOutbreakReported-0"
        assert docs[1].id == "who-DiseaseOutbreakReported-1"
        assert docs[2].id == "who-WHOHealthAlert-2"


def test_who_null_fields_and_stringification_resilience():
    """Verify WHO search does not stringify explicit null JSON values to literal 'None' strings."""
    client = MCPClient()
    fake_who_json = {
        "value": [
            {
                "Title": None,
                "PublicationDateAndTime": None,
                "DonId": None,
                "ItemDefaultUrl": None,
                "Url": None,
                "Summary": "Outbreak of viral hemorrhagic fever.",
            }
        ]
    }

    with patch("httpx.Client.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = fake_who_json
        mock_get.return_value = mock_resp

        docs = client.search_who("fever", max_results=1)
        assert len(docs) == 1
        doc = docs[0]

        assert doc.title == "WHO Health Alert"
        assert doc.title != "None"
        assert doc.published == ""
        assert doc.published != "None"
        assert doc.metadata["don_id"] == ""
        assert doc.metadata["publication_date"] == ""
        assert "None" not in doc.id
        assert doc.id == "who-WHOHealthAlert-0"
        assert doc.url == "https://www.who.int/emergencies/disease-outbreak-news"

