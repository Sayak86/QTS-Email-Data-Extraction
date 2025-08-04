"""
Stanza-based PII analyzer implementation
Designed for UBS restrictive environment
"""

import re
import logging
from typing import Dict, List, Any, Optional
try:
    import stanza
except ImportError:
    stanza = None

from .pii_analyzer_base import PIIAnalyzer, PIIEntity, PIIType, AnalysisResult


class StanzaPIIAnalyzer(PIIAnalyzer):
    """
    PII analyzer using Stanza NLP library
    Focuses on NER and custom pattern matching
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Stanza PII Analyzer
        
        Args:
            config: Configuration dictionary with following keys:
                - language: Language code (default: 'en')
                - processors: List of processors to use (default: ['tokenize', 'ner'])
                - model_dir: Custom model directory path
                - use_gpu: Whether to use GPU (default: False for UBS environment)
                - custom_patterns: Dictionary of custom regex patterns
        """
        default_config = {
            'language': 'en',
            'processors': ['tokenize', 'ner'],
            'use_gpu': False,  # Conservative for UBS environment
            'model_dir': None,
            'custom_patterns': {},
            'confidence_threshold': 0.7
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(name="StanzaPIIAnalyzer", config=default_config)
        
        self.nlp = None
        self.logger = logging.getLogger(__name__)
        
        # Custom regex patterns for PII detection
        self.default_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            'url': r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?',
            'iban': r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b'
        }
        
        # Merge custom patterns with defaults
        self.patterns = {**self.default_patterns, **self.config.get('custom_patterns', {})}
        
        # Compile regex patterns
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE) 
            for name, pattern in self.patterns.items()
        }
        
        # Stanza NER to PIIType mapping
        self.ner_to_pii_mapping = {
            'PERSON': PIIType.PERSON,
            'PER': PIIType.PERSON,
            'ORG': PIIType.ORGANIZATION,
            'ORGANIZATION': PIIType.ORGANIZATION,
            'LOC': PIIType.LOCATION,
            'LOCATION': PIIType.LOCATION,
            'GPE': PIIType.LOCATION,  # Geopolitical entity
            'DATE': PIIType.DATE_TIME,
            'TIME': PIIType.DATE_TIME,
        }
    
    def initialize(self) -> bool:
        """
        Initialize Stanza NLP pipeline
        
        Returns:
            bool: True if initialization successful
        """
        if stanza is None:
            self.logger.error("Stanza library not available. Please install with: pip install stanza")
            return False
        
        try:
            # Download model if needed (may require internet access)
            try:
                stanza.download(
                    lang=self.config['language'], 
                    model_dir=self.config.get('model_dir'),
                    verbose=False
                )
            except Exception as e:
                self.logger.warning(f"Could not download Stanza model: {e}")
                # Continue anyway - model might already be available
            
            # Initialize pipeline
            pipeline_config = {
                'lang': self.config['language'],
                'processors': ','.join(self.config['processors']),
                'use_gpu': self.config['use_gpu'],
                'verbose': False
            }
            
            if self.config.get('model_dir'):
                pipeline_config['model_dir'] = self.config['model_dir']
            
            self.nlp = stanza.Pipeline(**pipeline_config)
            self.is_initialized = True
            self.logger.info("Stanza PII Analyzer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Stanza: {e}")
            return False
    
    def analyze(self, text: str) -> AnalysisResult:
        """
        Analyze text for PII using Stanza NER and custom patterns
        
        Args:
            text: Input text to analyze
            
        Returns:
            AnalysisResult: Analysis results
        """
        if not self.is_initialized:
            raise RuntimeError("Analyzer not initialized. Call initialize() first.")
        
        if not self.validate_input(text):
            return AnalysisResult(
                entities=[],
                processed_text="",
                analysis_metadata={"error": "Invalid input text"}
            )
        
        processed_text = self.preprocess_text(text)
        entities = []
        
        try:
            # Stanza NER analysis
            ner_entities = self._extract_ner_entities(processed_text)
            entities.extend(ner_entities)
            
            # Custom pattern matching
            pattern_entities = self._extract_pattern_entities(processed_text)
            entities.extend(pattern_entities)
            
            # Post-process to remove duplicates and filter by confidence
            entities = self.postprocess_entities(entities)
            entities = [e for e in entities if e.confidence >= self.config['confidence_threshold']]
            
            analysis_metadata = {
                "analyzer": self.name,
                "total_entities_found": len(entities),
                "ner_entities": len(ner_entities),
                "pattern_entities": len(pattern_entities),
                "text_length": len(processed_text)
            }
            
            return AnalysisResult(
                entities=entities,
                processed_text=processed_text,
                analysis_metadata=analysis_metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error during analysis: {e}")
            return AnalysisResult(
                entities=[],
                processed_text=processed_text,
                analysis_metadata={"error": str(e)}
            )
    
    def _extract_ner_entities(self, text: str) -> List[PIIEntity]:
        """
        Extract entities using Stanza NER
        
        Args:
            text: Text to analyze
            
        Returns:
            List[PIIEntity]: Detected NER entities
        """
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.entities:
                # Map Stanza NER labels to PIIType
                pii_type = self.ner_to_pii_mapping.get(ent.type, PIIType.CUSTOM)
                
                # Calculate confidence (Stanza doesn't provide confidence scores directly)
                # Using a heuristic based on entity length and type
                confidence = self._calculate_ner_confidence(ent)
                
                entity = PIIEntity(
                    entity_type=pii_type,
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=confidence,
                    source_analyzer=f"{self.name}_NER",
                    metadata={
                        "stanza_type": ent.type,
                        "stanza_start_char": ent.start_char,
                        "stanza_end_char": ent.end_char
                    }
                )
                entities.append(entity)
                
        except Exception as e:
            self.logger.error(f"Error in NER extraction: {e}")
        
        return entities
    
    def _extract_pattern_entities(self, text: str) -> List[PIIEntity]:
        """
        Extract entities using custom regex patterns
        
        Args:
            text: Text to analyze
            
        Returns:
            List[PIIEntity]: Detected pattern entities
        """
        entities = []
        
        pattern_to_pii_type = {
            'email': PIIType.EMAIL_ADDRESS,
            'phone': PIIType.PHONE_NUMBER,
            'ssn': PIIType.SSN,
            'credit_card': PIIType.CREDIT_CARD,
            'ip_address': PIIType.IP_ADDRESS,
            'url': PIIType.URL,
            'iban': PIIType.IBAN
        }
        
        for pattern_name, compiled_pattern in self.compiled_patterns.items():
            try:
                matches = compiled_pattern.finditer(text)
                
                for match in matches:
                    pii_type = pattern_to_pii_type.get(pattern_name, PIIType.CUSTOM)
                    
                    # Calculate confidence based on pattern match quality
                    confidence = self._calculate_pattern_confidence(match.group(), pattern_name)
                    
                    entity = PIIEntity(
                        entity_type=pii_type,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                        source_analyzer=f"{self.name}_Pattern",
                        metadata={
                            "pattern_name": pattern_name,
                            "pattern": self.patterns[pattern_name]
                        }
                    )
                    entities.append(entity)
                    
            except Exception as e:
                self.logger.error(f"Error in pattern matching for {pattern_name}: {e}")
        
        return entities
    
    def _calculate_ner_confidence(self, ent) -> float:
        """
        Calculate confidence score for NER entities
        Stanza doesn't provide confidence directly, so using heuristics
        
        Args:
            ent: Stanza entity object
            
        Returns:
            float: Confidence score between 0 and 1
        """
        base_confidence = 0.7
        
        # Adjust based on entity type
        high_confidence_types = ['PERSON', 'ORG', 'ORGANIZATION']
        if ent.type in high_confidence_types:
            base_confidence = 0.8
        
        # Adjust based on entity length (longer entities often more reliable)
        entity_length = len(ent.text)
        if entity_length > 10:
            base_confidence += 0.1
        elif entity_length < 3:
            base_confidence -= 0.1
        
        # Adjust based on capitalization (proper nouns)
        if ent.text.istitle():
            base_confidence += 0.05
        
        return min(1.0, max(0.1, base_confidence))
    
    def _calculate_pattern_confidence(self, match_text: str, pattern_name: str) -> float:
        """
        Calculate confidence score for pattern matches
        
        Args:
            match_text: Matched text
            pattern_name: Name of the pattern
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Base confidence varies by pattern type
        pattern_confidence = {
            'email': 0.9,
            'phone': 0.8,
            'ssn': 0.95,
            'credit_card': 0.85,
            'ip_address': 0.9,
            'url': 0.9,
            'iban': 0.9
        }
        
        base_confidence = pattern_confidence.get(pattern_name, 0.7)
        
        # Additional validation for specific patterns
        if pattern_name == 'email':
            # Basic email validation
            if '@' in match_text and '.' in match_text.split('@')[1]:
                return base_confidence
            else:
                return base_confidence * 0.7
        
        elif pattern_name == 'phone':
            # Remove non-digits and check length
            digits_only = re.sub(r'\D', '', match_text)
            if 10 <= len(digits_only) <= 15:
                return base_confidence
            else:
                return base_confidence * 0.6
        
        return base_confidence
    
    def get_supported_entities(self) -> List[PIIType]:
        """
        Get list of PII types supported by this analyzer
        
        Returns:
            List[PIIType]: Supported PII types
        """
        return [
            PIIType.PERSON,
            PIIType.ORGANIZATION,
            PIIType.LOCATION,
            PIIType.DATE_TIME,
            PIIType.EMAIL_ADDRESS,
            PIIType.PHONE_NUMBER,
            PIIType.SSN,
            PIIType.CREDIT_CARD,
            PIIType.IP_ADDRESS,
            PIIType.URL,
            PIIType.IBAN
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded Stanza model
        
        Returns:
            Dict containing model information
        """
        if not self.is_initialized:
            return {"error": "Analyzer not initialized"}
        
        return {
            "language": self.config['language'],
            "processors": self.config['processors'],
            "model_dir": self.config.get('model_dir', 'default'),
            "use_gpu": self.config['use_gpu'],
            "custom_patterns": list(self.patterns.keys())
        }
