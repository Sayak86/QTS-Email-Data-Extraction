import time
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict
import logging

# Import the abstract base classes
from pii_analyzer_base import PIIAnalyzer, PIIEntity, PIIAnalysisResult, PIIType, ConfidenceLevel

# External libraries (install as needed in UBS environment)
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logging.warning("Presidio not available")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available")

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available")

try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False
    logging.warning("GLiNER not available")


class PresidioPIIAnalyzer(PIIAnalyzer):
    """Presidio-based PII analyzer"""
    
    def _initialize(self) -> None:
        if not PRESIDIO_AVAILABLE:
            raise ImportError("Presidio not available")
        
        # Configure NLP engine for offline use
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
        }
        
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        self.supported_entities = [
            "CREDIT_CARD", "EMAIL_ADDRESS", "IBAN_CODE", "IP_ADDRESS",
            "PERSON", "PHONE_NUMBER", "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE"
        ]
    
    def detect_entities(self, text: str) -> List[PIIEntity]:
        results = self.analyzer.analyze(text=text, entities=self.supported_entities, language='en')
        
        entities = []
        for result in results:
            pii_type = self._map_presidio_type(result.entity_type)
            confidence_level = self.get_confidence_level(result.score)
            
            entity = PIIEntity(
                text=text[result.start:result.end],
                pii_type=pii_type,
                start_pos=result.start,
                end_pos=result.end,
                confidence=result.score,
                confidence_level=confidence_level,
                detector_name=self.name,
                context=self.get_context(text, result.start, result.end)
            )
            entities.append(entity)
        
        return entities
    
    def _map_presidio_type(self, presidio_type: str) -> PIIType:
        mapping = {
            "PERSON": PIIType.PERSON_NAME,
            "EMAIL_ADDRESS": PIIType.EMAIL_ADDRESS,
            "PHONE_NUMBER": PIIType.PHONE_NUMBER,
            "US_SSN": PIIType.SSN,
            "CREDIT_CARD": PIIType.CREDIT_CARD,
            "IBAN_CODE": PIIType.IBAN,
            "IP_ADDRESS": PIIType.IP_ADDRESS,
            "US_PASSPORT": PIIType.PASSPORT_NUMBER,
            "US_DRIVER_LICENSE": PIIType.DRIVER_LICENSE
        }
        return mapping.get(presidio_type, PIIType.OTHER)
    
    def analyze(self, text: str, **kwargs) -> PIIAnalysisResult:
        start_time = time.time()
        
        preprocessed_text = self.preprocess_text(text)
        entities = self.detect_entities(preprocessed_text)
        entities = self.postprocess_entities(entities)
        masked_text = self.mask_text(text, entities)
        
        processing_time = time.time() - start_time
        
        return PIIAnalysisResult(
            original_text=text,
            entities=entities,
            masked_text=masked_text,
            processing_time=processing_time,
            detector_stats={self.name: {"entities_found": len(entities)}},
            metadata={"analyzer": self.name, "model": "presidio"}
        )


class TransformerNERAnalyzer(PIIAnalyzer):
    """BERT/DistilBERT-based NER analyzer"""
    
    def _initialize(self) -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
        
        model_name = self.config.get("model_name", "dbmdz/bert-large-cased-finetuned-conll03-english")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline("ner", 
                                   model=self.model, 
                                   tokenizer=self.tokenizer,
                                   aggregation_strategy="simple")
    
    def detect_entities(self, text: str) -> List[PIIEntity]:
        results = self.ner_pipeline(text)
        
        entities = []
        for result in results:
            pii_type = self._map_transformer_type(result['entity_group'])
            confidence_level = self.get_confidence_level(result['score'])
            
            entity = PIIEntity(
                text=result['word'],
                pii_type=pii_type,
                start_pos=result['start'],
                end_pos=result['end'],
                confidence=result['score'],
                confidence_level=confidence_level,
                detector_name=self.name,
                context=self.get_context(text, result['start'], result['end'])
            )
            entities.append(entity)
        
        return entities
    
    def _map_transformer_type(self, entity_type: str) -> PIIType:
        mapping = {
            "PER": PIIType.PERSON_NAME,
            "PERSON": PIIType.PERSON_NAME,
            "ORG": PIIType.ORGANIZATION,
            "LOC": PIIType.LOCATION,
            "MISC": PIIType.OTHER
        }
        return mapping.get(entity_type, PIIType.OTHER)
    
    def analyze(self, text: str, **kwargs) -> PIIAnalysisResult:
        start_time = time.time()
        
        preprocessed_text = self.preprocess_text(text)
        entities = self.detect_entities(preprocessed_text)
        entities = self.postprocess_entities(entities)
        masked_text = self.mask_text(text, entities)
        
        processing_time = time.time() - start_time
        
        return PIIAnalysisResult(
            original_text=text,
            entities=entities,
            masked_text=masked_text,
            processing_time=processing_time,
            detector_stats={self.name: {"entities_found": len(entities)}},
            metadata={"analyzer": self.name, "model": "transformer"}
        )


class SpacyNERAnalyzer(PIIAnalyzer):
    """spaCy-based NER analyzer with custom patterns"""
    
    def _initialize(self) -> None:
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy not available")
        
        model_name = self.config.get("model_name", "en_core_web_sm")
        self.nlp = spacy.load(model_name)
        
        # Add custom patterns for financial entities
        self._add_custom_patterns()
    
    def _add_custom_patterns(self):
        """Add custom patterns for financial PII detection"""
        from spacy.matcher import Matcher
        
        self.matcher = Matcher(self.nlp.vocab)
        
        # IBAN pattern
        iban_pattern = [{"TEXT": {"REGEX": r"[A-Z]{2}[0-9]{2}[A-Z0-9]{4}[0-9]{7}([A-Z0-9]?){0,16}"}}]
        self.matcher.add("IBAN", [iban_pattern])
        
        # SWIFT code pattern
        swift_pattern = [{"TEXT": {"REGEX": r"[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?"}}]
        self.matcher.add("SWIFT", [swift_pattern])
        
        # Account number pattern (simple)
        account_pattern = [{"TEXT": {"REGEX": r"\b\d{8,17}\b"}}]
        self.matcher.add("ACCOUNT", [account_pattern])
    
    def detect_entities(self, text: str) -> List[PIIEntity]:
        doc = self.nlp(text)
        entities = []
        
        # Standard NER entities
        for ent in doc.ents:
            pii_type = self._map_spacy_type(ent.label_)
            if pii_type != PIIType.OTHER:
                confidence = 0.8  # spaCy doesn't provide confidence scores
                confidence_level = self.get_confidence_level(confidence)
                
                entity = PIIEntity(
                    text=ent.text,
                    pii_type=pii_type,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    confidence=confidence,
                    confidence_level=confidence_level,
                    detector_name=self.name,
                    context=self.get_context(text, ent.start_char, ent.end_char)
                )
                entities.append(entity)
        
        # Custom pattern matches
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            pii_type = self._map_custom_type(label)
            confidence = 0.7  # Lower confidence for pattern matches
            confidence_level = self.get_confidence_level(confidence)
            
            entity = PIIEntity(
                text=span.text,
                pii_type=pii_type,
                start_pos=span.start_char,
                end_pos=span.end_char,
                confidence=confidence,
                confidence_level=confidence_level,
                detector_name=self.name,
                context=self.get_context(text, span.start_char, span.end_char)
            )
            entities.append(entity)
        
        return entities
    
    def _map_spacy_type(self, spacy_type: str) -> PIIType:
        mapping = {
            "PERSON": PIIType.PERSON_NAME,
            "ORG": PIIType.ORGANIZATION,
            "GPE": PIIType.LOCATION,
            "LOC": PIIType.LOCATION,
            "DATE": PIIType.OTHER,  # Could be DATE_OF_BIRTH in some contexts
        }
        return mapping.get(spacy_type, PIIType.OTHER)
    
    def _map_custom_type(self, custom_type: str) -> PIIType:
        mapping = {
            "IBAN": PIIType.IBAN,
            "SWIFT": PIIType.SWIFT_CODE,
            "ACCOUNT": PIIType.ACCOUNT_NUMBER
        }
        return mapping.get(custom_type, PIIType.OTHER)
    
    def analyze(self, text: str, **kwargs) -> PIIAnalysisResult:
        start_time = time.time()
        
        preprocessed_text = self.preprocess_text(text)
        entities = self.detect_entities(preprocessed_text)
        entities = self.postprocess_entities(entities)
        masked_text = self.mask_text(text, entities)
        
        processing_time = time.time() - start_time
        
        return PIIAnalysisResult(
            original_text=text,
            entities=entities,
            masked_text=masked_text,
            processing_time=processing_time,
            detector_stats={self.name: {"entities_found": len(entities)}},
            metadata={"analyzer": self.name, "model": "spacy"}
        )


class GLiNERAnalyzer(PIIAnalyzer):
    """GLiNER-based zero-shot NER analyzer"""
    
    def _initialize(self) -> None:
        if not GLINER_AVAILABLE:
            raise ImportError("GLiNER not available")
        
        model_name = self.config.get("model_name", "urchade/gliner_base")
        self.model = GLiNER.from_pretrained(model_name)
        
        # Define entity labels for zero-shot detection
        self.labels = [
            "person", "email", "phone number", "credit card", "bank account",
            "social security number", "passport", "driver license", "address",
            "organization", "location", "date of birth", "medical license"
        ]
    
    def detect_entities(self, text: str) -> List[PIIEntity]:
        entities_found = self.model.predict_entities(text, self.labels, threshold=0.5)
        
        entities = []
        for entity in entities_found:
            pii_type = self._map_gliner_type(entity["label"])
            confidence = entity["score"]
            confidence_level = self.get_confidence_level(confidence)
            
            pii_entity = PIIEntity(
                text=entity["text"],
                pii_type=pii_type,
                start_pos=entity["start"],
                end_pos=entity["end"],
                confidence=confidence,
                confidence_level=confidence_level,
                detector_name=self.name,
                context=self.get_context(text, entity["start"], entity["end"])
            )
            entities.append(pii_entity)
        
        return entities
    
    def _map_gliner_type(self, gliner_type: str) -> PIIType:
        mapping = {
            "person": PIIType.PERSON_NAME,
            "email": PIIType.EMAIL_ADDRESS,
            "phone number": PIIType.PHONE_NUMBER,
            "credit card": PIIType.CREDIT_CARD,
            "bank account": PIIType.BANK_ACCOUNT,
            "social security number": PIIType.SSN,
            "passport": PIIType.PASSPORT_NUMBER,
            "driver license": PIIType.DRIVER_LICENSE,
            "address": PIIType.ADDRESS,
            "organization": PIIType.ORGANIZATION,
            "location": PIIType.LOCATION,
            "date of birth": PIIType.DATE_OF_BIRTH,
            "medical license": PIIType.MEDICAL_LICENSE
        }
        return mapping.get(gliner_type.lower(), PIIType.OTHER)
    
    def analyze(self, text: str, **kwargs) -> PIIAnalysisResult:
        start_time = time.time()
        
        preprocessed_text = self.preprocess_text(text)
        entities = self.detect_entities(preprocessed_text)
        entities = self.postprocess_entities(entities)
        masked_text = self.mask_text(text, entities)
        
        processing_time = time.time() - start_time
        
        return PIIAnalysisResult(
            original_text=text,
            entities=entities,
            masked_text=masked_text,
            processing_time=processing_time,
            detector_stats={self.name: {"entities_found": len(entities)}},
            metadata={"analyzer": self.name, "model": "gliner"}
        )


class EnsemblePIIAnalyzer(PIIAnalyzer):
    """Ensemble analyzer that combines multiple PII analyzers"""
    
    def _initialize(self) -> None:
        self.analyzers = []
        self.voting_strategy = self.config.get("voting_strategy", "majority")  # "majority", "unanimous", "any"
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        
        # Initialize available analyzers
        analyzer_configs = self.config.get("analyzers", {})
        
        if PRESIDIO_AVAILABLE and analyzer_configs.get("presidio", {}).get("enabled", True):
            self.analyzers.append(PresidioPIIAnalyzer("presidio", analyzer_configs.get("presidio", {})))
        
        if TRANSFORMERS_AVAILABLE and analyzer_configs.get("transformer", {}).get("enabled", True):
            self.analyzers.append(TransformerNERAnalyzer("transformer", analyzer_configs.get("transformer", {})))
        
        if SPACY_AVAILABLE and analyzer_configs.get("spacy", {}).get("enabled", True):
            self.analyzers.append(SpacyNERAnalyzer("spacy", analyzer_configs.get("spacy", {})))
        
        if GLINER_AVAILABLE and analyzer_configs.get("gliner", {}).get("enabled", True):
            self.analyzers.append(GLiNERAnalyzer("gliner", analyzer_configs.get("gliner", {})))
        
        if not self.analyzers:
            raise RuntimeError("No analyzers available")
    
    def detect_entities(self, text: str) -> List[PIIEntity]:
        all_entities = []
        
        # Collect entities from all analyzers
        for analyzer in self.analyzers:
            try:
                entities = analyzer.detect_entities(text)
                all_entities.extend(entities)
            except Exception as e:
                self.logger.warning(f"Analyzer {analyzer.name} failed: {e}")
        
        # Apply ensemble voting
        return self._apply_voting(all_entities)
    
    def _apply_voting(self, all_entities: List[PIIEntity]) -> List[PIIEntity]:
        """Apply voting strategy to combine results from multiple analyzers"""
        
        # Group overlapping entities
        entity_groups = self._group_overlapping_entities(all_entities)
        
        final_entities = []
        
        for group in entity_groups:
            if self.voting_strategy == "any":
                # Take the entity with highest confidence
                best_entity = max(group, key=lambda x: x.confidence)
                final_entities.append(best_entity)
            
            elif self.voting_strategy == "majority":
                # Require at least half of analyzers to agree
                required_votes = len(self.analyzers) // 2 + 1
                if len(group) >= required_votes:
                    best_entity = max(group, key=lambda x: x.confidence)
                    final_entities.append(best_entity)
            
            elif self.voting_strategy == "unanimous":
                # Require all analyzers to agree
                if len(group) == len(self.analyzers):
                    best_entity = max(group, key=lambda x: x.confidence)
                    final_entities.append(best_entity)
        
        return final_entities
    
    def _group_overlapping_entities(self, entities: List[PIIEntity]) -> List[List[PIIEntity]]:
        """Group entities that overlap in text positions"""
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda x: x.start_pos)
        
        groups = []
        current_group = [entities[0]]
        
        for entity in entities[1:]:
            # Check if current entity overlaps with any in current group
            overlapping = False
            for group_entity in current_group:
                if (entity.start_pos < group_entity.end_pos and 
                    entity.end_pos > group_entity.start_pos):
                    overlapping = True
                    break
            
            if overlapping:
                current_group.append(entity)
            else:
                groups.append(current_group)
                current_group = [entity]
        
        groups.append(current_group)
        return groups
    
    def analyze(self, text: str, **kwargs) -> PIIAnalysisResult:
        start_time = time.time()
        
        preprocessed_text = self.preprocess_text(text)
        entities = self.detect_entities(preprocessed_text)
        entities = self.postprocess_entities(entities)
        masked_text = self.mask_text(text, entities)
        
        processing_time = time.time() - start_time
        
        # Collect stats from all analyzers
        detector_stats = {}
        for analyzer in self.analyzers:
            detector_stats[analyzer.name] = {
                "available": True,
                "entities_contributed": len([e for e in entities if e.detector_name == analyzer.name])
            }
        
        return PIIAnalysisResult(
            original_text=text,
            entities=entities,
            masked_text=masked_text,
            processing_time=processing_time,
            detector_stats=detector_stats,
            metadata={
                "analyzer": self.name,
                "voting_strategy": self.voting_strategy,
                "analyzers_used": [a.name for a in self.analyzers]
            }
        )