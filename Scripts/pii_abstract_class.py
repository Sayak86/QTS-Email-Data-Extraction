from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

class PIIType(Enum):
    """Enumeration of PII types commonly found in emails"""
    PERSON_NAME = "PERSON"
    EMAIL_ADDRESS = "EMAIL"
    PHONE_NUMBER = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    ADDRESS = "ADDRESS"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    PASSPORT_NUMBER = "PASSPORT"
    DRIVER_LICENSE = "DRIVER_LICENSE"
    IP_ADDRESS = "IP_ADDRESS"
    URL = "URL"
    IBAN = "IBAN"
    SWIFT_CODE = "SWIFT"
    ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    MEDICAL_LICENSE = "MEDICAL_LICENSE"
    OTHER = "OTHER"

class ConfidenceLevel(Enum):
    """Confidence levels for PII detection"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"

@dataclass
class PIIEntity:
    """Data class representing a detected PII entity"""
    text: str
    pii_type: PIIType
    start_pos: int
    end_pos: int
    confidence: float
    confidence_level: ConfidenceLevel
    detector_name: str
    context: Optional[str] = None
    masked_value: Optional[str] = None
    
    def __post_init__(self):
        """Auto-generate masked value if not provided"""
        if self.masked_value is None:
            self.masked_value = self._generate_mask()
    
    def _generate_mask(self) -> str:
        """Generate appropriate mask based on PII type"""
        if self.pii_type == PIIType.EMAIL_ADDRESS:
            return "***@***.***"
        elif self.pii_type == PIIType.PHONE_NUMBER:
            return "***-***-****"
        elif self.pii_type == PIIType.SSN:
            return "***-**-****"
        elif self.pii_type == PIIType.CREDIT_CARD:
            return "****-****-****-****"
        elif len(self.text) <= 3:
            return "*" * len(self.text)
        else:
            return self.text[:2] + "*" * (len(self.text) - 4) + self.text[-2:]

@dataclass
class PIIAnalysisResult:
    """Complete result of PII analysis"""
    original_text: str
    entities: List[PIIEntity]
    masked_text: str
    processing_time: float
    detector_stats: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    
    @property
    def has_pii(self) -> bool:
        """Check if any PII was detected"""
        return len(self.entities) > 0
    
    @property
    def high_confidence_entities(self) -> List[PIIEntity]:
        """Get entities with high or very high confidence"""
        return [e for e in self.entities if e.confidence_level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]]
    
    def get_entities_by_type(self, pii_type: PIIType) -> List[PIIEntity]:
        """Get all entities of a specific type"""
        return [e for e in self.entities if e.pii_type == pii_type]

class PIIAnalyzer(ABC):
    """Abstract base class for PII analyzers"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the analyzer (load models, set up configurations, etc.)"""
        pass
    
    @abstractmethod
    def analyze(self, text: str, **kwargs) -> PIIAnalysisResult:
        """
        Analyze text for PII entities
        
        Args:
            text: The text to analyze
            **kwargs: Additional parameters specific to the analyzer
            
        Returns:
            PIIAnalysisResult containing detected entities and analysis metadata
        """
        pass
    
    @abstractmethod
    def detect_entities(self, text: str) -> List[PIIEntity]:
        """
        Core method to detect PII entities in text
        
        Args:
            text: The text to analyze
            
        Returns:
            List of detected PIIEntity objects
        """
        pass
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text before analysis (can be overridden by subclasses)
        
        Args:
            text: Raw text
            
        Returns:
            Preprocessed text
        """
        # Basic preprocessing - remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def postprocess_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """
        Post-process detected entities (remove duplicates, merge overlapping, etc.)
        
        Args:
            entities: List of detected entities
            
        Returns:
            Cleaned list of entities
        """
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda x: x.start_pos)
        
        # Remove duplicates and handle overlaps
        cleaned_entities = []
        for entity in entities:
            # Check for overlaps with existing entities
            overlapping = False
            for existing in cleaned_entities:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # Keep the entity with higher confidence
                    if entity.confidence > existing.confidence:
                        cleaned_entities.remove(existing)
                        break
                    else:
                        overlapping = True
                        break
            
            if not overlapping:
                cleaned_entities.append(entity)
        
        return cleaned_entities
    
    def mask_text(self, text: str, entities: List[PIIEntity]) -> str:
        """
        Create masked version of text
        
        Args:
            text: Original text
            entities: List of detected PII entities
            
        Returns:
            Text with PII entities masked
        """
        if not entities:
            return text
        
        masked_text = text
        # Sort entities by start position in reverse order to maintain positions
        sorted_entities = sorted(entities, key=lambda x: x.start_pos, reverse=True)
        
        for entity in sorted_entities:
            masked_text = (masked_text[:entity.start_pos] + 
                          entity.masked_value + 
                          masked_text[entity.end_pos:])
        
        return masked_text
    
    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """
        Convert numeric confidence to confidence level
        
        Args:
            confidence: Numeric confidence score (0.0 to 1.0)
            
        Returns:
            ConfidenceLevel enum
        """
        if confidence >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def get_context(self, text: str, start_pos: int, end_pos: int, 
                   context_window: int = 50) -> str:
        """
        Extract context around a detected entity
        
        Args:
            text: Full text
            start_pos: Start position of entity
            end_pos: End position of entity
            context_window: Number of characters to include on each side
            
        Returns:
            Context string
        """
        context_start = max(0, start_pos - context_window)
        context_end = min(len(text), end_pos + context_window)
        
        context = text[context_start:context_end]
        # Add ellipsis if truncated
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."
            
        return context