"""
Simple Stanza PII Analyzer using Presidio pattern
Aligned with comprehensive base class
"""

import time
from typing import List, Dict, Any
from presidio_analyzer.nlp_engine import NlpEngineProvider
from pii_analyzer_base import PIIAnalyzer, PIIEntity, PIIType, PIIAnalysisResult, ConfidenceLevel


class StanzaPIIAnalyzer(PIIAnalyzer):
    """Simple Stanza-based PII analyzer using your base class"""
    
    def __init__(self, lang_code: str = "en", config: Dict[str, Any] = None):
        # Simple configuration like you wanted
        self.lang_code = lang_code
        super().__init__("StanzaPIIAnalyzer", config)
    
    def _initialize(self) -> None:
        """Initialize the Stanza NLP engine using Presidio"""
        try:
            # Simple configuration exactly like you requested
            configuration = {
                "nlp_engine_name": "stanza",
                "models": [{"lang_code": self.lang_code, "model_name": self.lang_code}]
            }
            
            # Create NLP engine using Presidio's provider
            provider = NlpEngineProvider(nlp_configuration=configuration)
            self.nlp_engine = provider.create_engine()
            
            self.logger.info(f"Stanza analyzer initialized for language: {self.lang_code}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Stanza analyzer: {e}")
            raise
    
    def detect_entities(self, text: str) -> List[PIIEntity]:
        """Core method to detect PII entities using Stanza"""
        entities = []
        
        try:
            # Process text with Stanza via Presidio
            nlp_artifacts = self.nlp_engine.process_text(text, self.lang_code)
            
            # Extract entities from NLP artifacts
            for entity in nlp_artifacts.entities:
                # Map Stanza labels to our PIIType enum
                pii_type = self._map_entity_type(entity.label_)
                confidence = 0.8  # Default confidence for Stanza NER
                
                pii_entity = PIIEntity(
                    text=entity.text,
                    pii_type=pii_type,
                    start_pos=entity.start_char,
                    end_pos=entity.end_char,
                    confidence=confidence,
                    confidence_level=self.get_confidence_level(confidence),
                    detector_name=self.name,
                    context=self.get_context(text, entity.start_char, entity.end_char)
                )
                entities.append(pii_entity)
                
        except Exception as e:
            self.logger.error(f"Error detecting entities: {e}")
        
        return entities
    
    def analyze(self, text: str, **kwargs) -> PIIAnalysisResult:
        """Analyze text for PII - implements abstract method"""
        start_time = time.time()
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Detect entities
        entities = self.detect_entities(processed_text)
        
        # Post-process entities
        entities = self.postprocess_entities(entities)
        
        # Create masked text
        masked_text = self.mask_text(processed_text, entities)
        
        processing_time = time.time() - start_time
        
        # Create detector stats
        detector_stats = {
            self.name: {
                "entities_found": len(entities),
                "entity_types": list(set([e.pii_type.value for e in entities])),
                "avg_confidence": sum([e.confidence for e in entities]) / len(entities) if entities else 0
            }
        }
        
        # Create metadata
        metadata = {
            "language": self.lang_code,
            "text_length": len(text),
            "processed_text_length": len(processed_text)
        }
        
        return PIIAnalysisResult(
            original_text=text,
            entities=entities,
            masked_text=masked_text,
            processing_time=processing_time,
            detector_stats=detector_stats,
            metadata=metadata
        )
    
    def _map_entity_type(self, stanza_label: str) -> PIIType:
        """Map Stanza NER labels to PIIType enum"""
        mapping = {
            "PERSON": PIIType.PERSON_NAME,
            "PER": PIIType.PERSON_NAME,
            "ORG": PIIType.ORGANIZATION,
            "ORGANIZATION": PIIType.ORGANIZATION,
            "LOC": PIIType.LOCATION,
            "LOCATION": PIIType.LOCATION,
            "GPE": PIIType.LOCATION,  # Geopolitical entity
        }
        
        return mapping.get(stanza_label, PIIType.OTHER)
