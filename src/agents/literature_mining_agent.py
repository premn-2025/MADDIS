#!/usr/bin/env python3
"""
Literature Mining Agent - Autonomous BioBERT-powered PubMed Research Agent
"""

import asyncio
import requests
import json
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from datetime import datetime
import re
import logging
from pathlib import Path

import pandas as pd
import networkx as nx

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers not available. Using rule-based extraction.")


@dataclass
class LiteratureEvidence:
    """Evidence found in literature"""
    pmid: str
    title: str
    abstract: str
    authors: List[str]
    journal: str
    pub_date: str
    entities: List[Dict]
    relations: List[Dict]
    confidence: float
    relevance_score: float


@dataclass
class DrugInteractionEvidence:
    """Specific evidence for drug interactions"""
    drug1: str
    drug2: str
    interaction_type: str
    severity: str
    mechanism: str
    evidence: List[LiteratureEvidence]
    confidence: float


class LiteratureMiningAgent:
    """Agent for mining biomedical literature"""

    def __init__(self, pubmed_api_key: Optional[str] = None):
        self.pubmed_api_key = pubmed_api_key
        self.pubmed_base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        
        self.setup_models()
        self.knowledge_graph = nx.MultiDiGraph()
        
        logger.info("Literature Mining Agent initialized")

    def setup_models(self):
        """Initialize NLP models"""
        self.ner_model = None
        self.device = None
        
        if HAS_TRANSFORMERS:
            try:
                self.ner_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
                self.ner_model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.ner_model.to(self.device)
                logger.info("BioBERT models loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load BioBERT: {e}")
                self.ner_model = None

    async def search_literature(self, query: str, max_results: int = 50) -> Dict:
        """Search PubMed for relevant literature"""
        logger.info(f"Searching PubMed for: {query}")

        try:
            search_url = f"{self.pubmed_base_url}esearch.fcgi"
            search_params = {
                'db': 'pubmed',
                'term': query,
                'retmax': max_results,
                'retmode': 'json',
                'sort': 'relevance'
            }

            if self.pubmed_api_key:
                search_params['api_key'] = self.pubmed_api_key

            response = requests.get(search_url, params=search_params, timeout=30)
            search_results = response.json()

            pmids = search_results.get('esearchresult', {}).get('idlist', [])

            if not pmids:
                return {'papers': [], 'count': 0, 'query': query}

            papers = await self.fetch_paper_details(pmids)

            for paper in papers:
                if paper.get('abstract'):
                    entities = self.extract_entities_from_text(paper['abstract'])
                    paper['extracted_entities'] = entities

            result = {
                'papers': papers,
                'count': len(papers),
                'query': query,
                'search_timestamp': datetime.now().isoformat()
            }

            self.update_knowledge_graph_from_papers(papers)

            return result

        except Exception as e:
            logger.error(f"Search error: {e}")
            return {'error': str(e), 'papers': []}

    async def fetch_paper_details(self, pmids: List[str]) -> List[Dict]:
        """Fetch detailed information for papers"""
        if not pmids:
            return []

        fetch_url = f"{self.pubmed_base_url}efetch.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml'
        }

        if self.pubmed_api_key:
            fetch_params['api_key'] = self.pubmed_api_key

        response = requests.get(fetch_url, params=fetch_params, timeout=30)

        papers = []
        try:
            root = ET.fromstring(response.content)

            for article in root.findall('.//PubmedArticle'):
                paper_info = self.parse_pubmed_article(article)
                papers.append(paper_info)

        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")

        return papers

    def parse_pubmed_article(self, article_xml) -> Dict:
        """Parse PubMed article XML"""
        paper = {}

        try:
            pmid_elem = article_xml.find('.//PMID')
            paper['pmid'] = pmid_elem.text if pmid_elem is not None else ''

            title_elem = article_xml.find('.//ArticleTitle')
            paper['title'] = title_elem.text if title_elem is not None else ''

            abstract_elem = article_xml.find('.//AbstractText')
            paper['abstract'] = abstract_elem.text if abstract_elem is not None else ''

            authors = []
            for author in article_xml.findall('.//Author'):
                lastname = author.find('LastName')
                forename = author.find('ForeName')
                if lastname is not None and forename is not None:
                    authors.append(f"{forename.text} {lastname.text}")
            paper['authors'] = authors

            journal_elem = article_xml.find('.//Journal/Title')
            paper['journal'] = journal_elem.text if journal_elem is not None else ''

            pub_date = article_xml.find('.//PubDate')
            if pub_date is not None:
                year = pub_date.find('Year')
                month = pub_date.find('Month')
                paper['pub_date'] = f"{year.text if year is not None else ''}-{month.text if month is not None else ''}"
            else:
                paper['pub_date'] = ''

        except Exception as e:
            logger.error(f"Article parsing error: {e}")

        return paper

    def extract_entities_from_text(self, text: str) -> Dict:
        """Extract biomedical entities from text"""
        if not text:
            return {'drugs': [], 'proteins': [], 'diseases': [], 'genes': []}

        entities = {
            'drugs': self.extract_drug_entities(text),
            'proteins': self.extract_protein_entities(text),
            'diseases': self.extract_disease_entities(text),
            'genes': self.extract_gene_entities(text)
        }

        return entities

    def extract_drug_entities(self, text: str) -> List[Dict]:
        """Extract drug entities"""
        patterns = [
            r'\b(aspirin|ibuprofen|acetaminophen|warfarin|metformin|lisinopril|atorvastatin|simvastatin|omeprazole)\b'
        ]
        return self._extract_with_patterns(text, patterns)

    def extract_protein_entities(self, text: str) -> List[Dict]:
        """Extract protein entities"""
        patterns = [
            r'\b(COX-?\d|CYP\d\w+|p53|EGFR|VEGF|TNF|IL-?\d+)\b'
        ]
        return self._extract_with_patterns(text, patterns)

    def extract_disease_entities(self, text: str) -> List[Dict]:
        """Extract disease entities"""
        patterns = [
            r'\b(cancer|diabetes|hypertension|depression|arthritis|cardiovascular)\b'
        ]
        return self._extract_with_patterns(text, patterns)

    def extract_gene_entities(self, text: str) -> List[Dict]:
        """Extract gene entities"""
        patterns = [
            r'\b(BRCA\d|TP53|EGFR)\b'
        ]
        return self._extract_with_patterns(text, patterns)

    def _extract_with_patterns(self, text: str, patterns: List[str]) -> List[Dict]:
        """Extract entities using regex patterns"""
        entities = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'confidence': 0.6
                })

        unique_entities = []
        seen = set()
        for entity in entities:
            if entity['text'].lower() not in seen:
                unique_entities.append(entity)
                seen.add(entity['text'].lower())

        return unique_entities

    async def find_drug_interaction_evidence(self, drug1: str, drug2: str) -> Dict:
        """Find literature evidence for drug-drug interaction"""
        logger.info(f"Searching for interaction evidence: {drug1} + {drug2}")

        queries = [
            f'"{drug1}" AND "{drug2}" AND "drug interaction"',
            f'"{drug1}" AND "{drug2}" AND ("contraindicated" OR "adverse")',
        ]

        all_evidence = []

        for query in queries:
            search_result = await self.search_literature(query, max_results=25)

            for paper in search_result.get('papers', []):
                evidence = self._extract_interaction_evidence(paper, drug1, drug2)
                if evidence:
                    all_evidence.append(evidence)

        if all_evidence:
            return self._synthesize_interaction_evidence(all_evidence, drug1, drug2)
        else:
            return {
                'drug1': drug1,
                'drug2': drug2,
                'interaction_found': False,
                'confidence': 0.1,
                'evidence_papers': 0,
                'summary': f"No strong evidence found for {drug1}-{drug2} interaction"
            }

    def _extract_interaction_evidence(self, paper: Dict, drug1: str, drug2: str) -> Optional[Dict]:
        """Extract interaction evidence from a paper"""
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        text = f"{title} {abstract}"

        drug1_mentioned = drug1.lower() in text
        drug2_mentioned = drug2.lower() in text

        if not (drug1_mentioned and drug2_mentioned):
            return None

        interaction_keywords = [
            'interaction', 'contraindicated', 'adverse', 'avoid', 'caution',
            'bleeding', 'toxicity', 'pharmacokinetic'
        ]

        evidence_strength = 0
        found_keywords = []

        for keyword in interaction_keywords:
            if keyword in text:
                evidence_strength += 1
                found_keywords.append(keyword)

        if evidence_strength > 0:
            return {
                'pmid': paper.get('pmid'),
                'title': paper.get('title'),
                'evidence_strength': evidence_strength,
                'keywords': found_keywords,
                'confidence': min(evidence_strength / 3.0, 1.0)
            }

        return None

    def _synthesize_interaction_evidence(self, evidence_list: List[Dict], drug1: str, drug2: str) -> Dict:
        """Synthesize evidence from multiple papers"""
        total_papers = len(evidence_list)
        avg_confidence = np.mean([e['confidence'] for e in evidence_list])
        all_keywords = []

        for evidence in evidence_list:
            all_keywords.extend(evidence['keywords'])

        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

        if 'contraindicated' in keyword_counts or 'avoid' in keyword_counts:
            severity = 'HIGH'
            interaction_type = 'contraindicated'
        elif 'bleeding' in keyword_counts or 'toxicity' in keyword_counts:
            severity = 'HIGH'
            interaction_type = 'adverse_effect'
        elif 'caution' in keyword_counts:
            severity = 'MODERATE'
            interaction_type = 'requires_monitoring'
        else:
            severity = 'LOW'
            interaction_type = 'pharmacokinetic'

        return {
            'drug1': drug1,
            'drug2': drug2,
            'interaction_found': True,
            'severity': severity,
            'interaction_type': interaction_type,
            'confidence': avg_confidence,
            'evidence_papers': total_papers,
            'summary': f"Found {total_papers} papers suggesting {severity.lower()} risk {interaction_type} between {drug1} and {drug2}"
        }

    def update_knowledge_graph_from_papers(self, papers: List[Dict]):
        """Update knowledge graph with entities from papers"""
        for paper in papers:
            entities = paper.get('extracted_entities', {})

            for entity_type, entity_list in entities.items():
                for entity in entity_list:
                    entity_name = entity['text']
                    self.knowledge_graph.add_node(
                        entity_name,
                        type=entity_type,
                        confidence=entity['confidence']
                    )

            drugs = [e['text'] for e in entities.get('drugs', [])]
            diseases = [e['text'] for e in entities.get('diseases', [])]

            for drug in drugs:
                for disease in diseases:
                    self.knowledge_graph.add_edge(
                        drug, disease,
                        relation='treats_or_causes',
                        evidence_pmid=paper.get('pmid', ''),
                        confidence=0.5
                    )

    def query_knowledge_graph(self, entity: str, relation_type: str = None) -> Dict:
        """Query the knowledge graph"""
        if entity not in self.knowledge_graph:
            return {
                'entity': entity,
                'found': False,
                'message': 'Entity not found in knowledge graph'
            }

        connected = list(self.knowledge_graph.neighbors(entity))

        if relation_type:
            filtered_connected = []
            for neighbor in connected:
                edges = self.knowledge_graph.get_edge_data(entity, neighbor)
                if edges:
                    for edge_data in edges.values():
                        if edge_data.get('relation') == relation_type:
                            filtered_connected.append(neighbor)
                            break
            connected = filtered_connected

        return {
            'entity': entity,
            'found': True,
            'connected_entities': connected[:20],
            'entity_type': self.knowledge_graph.nodes[entity].get('type', 'unknown'),
            'graph_stats': {
                'total_nodes': self.knowledge_graph.number_of_nodes(),
                'total_edges': self.knowledge_graph.number_of_edges()
            }
        }


async def test_literature_agent():
    """Test the literature mining agent"""
    agent = LiteratureMiningAgent()
    
    search_result = await agent.search_literature("aspirin warfarin interaction", max_results=5)
    print(f"Found {len(search_result['papers'])} papers")
    
    interaction_evidence = await agent.find_drug_interaction_evidence("aspirin", "warfarin")
    print(f"Interaction evidence: {interaction_evidence.get('interaction_found')}")
    print(f"Confidence: {interaction_evidence.get('confidence', 0):.2f}")


if __name__ == "__main__":
    import os
    os.makedirs('logs/agents', exist_ok=True)
    asyncio.run(test_literature_agent())
