"""
Simple Stanza PII Analyzer using Presidio pattern
"""

from typing import List
from presidio_analyzer.nlp_engine import NlpEngineProvider
from pii_analyzer_base import PIIAnalyzer, PIIEntity, AnalysisResult


class StanzaPIIAnalyzer(PIIAnalyzer):
    """Simple Stanza-based PII analyzer"""
    
    def __init__(self, lang_code: str = "en"):
        super().__init__("StanzaPIIAnalyzer")
        
        # Simple configuration like you requested
        configuration = {
            "nlp_engine_name": "stanza",
            "models": [{"lang_code": lang_code, "model_name": lang_code}]
        }
        
        # Create NLP engine using Presidio's provider
        provider = NlpEngineProvider(nlp_configuration=configuration)
        self.nlp_engine = provider.create_engine()
    
    def analyze(self, text: str) -> AnalysisResult:
        """Analyze text using Stanza NLP engine"""
        entities = []
        
        # Process text with Stanza
        nlp_artifacts = self.nlp_engine.process_text(text, "en")
        
        # Extract entities from NLP artifacts
        for entity in nlp_artifacts.entities:
            pii_entity = PIIEntity(
                entity_type=entity.label_,
                text=entity.text,
                start=entity.start_char,
                end=entity.end_char,
                confidence=0.8  # Default confidence
            )
            entities.append(pii_entity)
        
        return AnalysisResult(entities=entities, text=text)
