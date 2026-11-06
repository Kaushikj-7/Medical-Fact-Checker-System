import pytest
import httpx
from unittest.mock import MagicMock, patch
from medical_crag.mcp_tools import MCPClient


def test_pubmed_search_success():
    client = MCPClient()

    fake_esearch_json = {
        "esearchresult": {
            "idlist": ["998877"]
        }
    }
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
        # Mock esearch call then efetch call
        mock_resp1 = MagicMock = patch("httpx.Response").start()
        mock_resp1.status_code = 200
        mock_resp1.json.return_value = fake_esearch_json

        mock_resp2 = MagicMock = patch("httpx.Response").start()
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
