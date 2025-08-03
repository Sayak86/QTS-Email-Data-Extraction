#!/usr/bin/env python3
"""
PII Analysis Usage Example
Demonstrates how to use the PII analysis pipeline for email content
"""

import json
import logging
from typing import Dict, Any
from pii_pipeline import (
    EnsemblePIIAnalyzer, PresidioPIIAnalyzer, TransformerNERAnalyzer,
    SpacyNERAnalyzer, GLiNERAnalyzer, PIIType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_analyzer_config() -> Dict[str, Any]:
    """Create configuration for the PII analyzer"""
    return {
        "voting_strategy": "majority",  # "majority", "unanimous", "any"
        "confidence_threshold": 0.5,
        "analyzers": {
            "presidio": {
                "enabled": True,
                "model_config": {}
            },
            "transformer": {
                "enabled": True,
                "model_name": "dbmdz/bert-large-cased-finetuned-conll03-english"
            },
            "spacy": {
                "enabled": True,
                "model_name": "en_core_web_sm"
            },
            "gliner": {
                "enabled": True,
                "model_name": "urchade/gliner_base"
            }
        }
    }

def analyze_outlook_email(email_text: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyze Outlook email text for PII
    
    Args:
        email_text: OCR extracted text from Outlook email
        config: Configuration dictionary for analyzers
    
    Returns:
        Dictionary containing analysis results
    """
    if config is None:
        config = create_analyzer_config()
    
    # Initialize the ensemble analyzer
    analyzer = EnsemblePIIAnalyzer("ensemble_pii", config)
    
    try:
        # Perform analysis
        result = analyzer.analyze(email_text)
        
        # Create summary report
        report = {
            "analysis_summary": {
                "has_pii": result.has_pii,
                "total_entities": len(result.entities),
                "high_confidence_entities": len(result.high_confidence_entities),
                "processing_time": result.processing_time,
                "analyzers_used": result.metadata.get("analyzers_used", [])
            },
            "entities_by_type": {},
            "entities_details": [],
            "masked_text": result.masked_text,
            "detector_performance": result.detector_stats
        }
        
        # Group entities by type
        for pii_type in PIIType:
            entities = result.get_entities_by_type(pii_type)
            if entities:
                report["entities_by_type"][pii_type.value] = {
                    "count": len(entities),
                    "entities": [
                        {
                            "text": entity.text,
                            "confidence": entity.confidence,
                            "confidence_level": entity.confidence_level.value,
                            "detector": entity.detector_name,
                            "masked_value": entity.masked_value,
                            "position": f"{entity.start_pos}-{entity.end_pos}"
                        }
                        for entity in entities
                    ]
                }
        
        # Detailed entity information
        for entity in result.entities:
            report["entities_details"].append({
                "text": entity.text,
                "type": entity.pii_type.value,
                "confidence": entity.confidence,
                "confidence_level": entity.confidence_level.value,
                "detector": entity.detector_name,
                "position": {
                    "start": entity.start_pos,
                    "end": entity.end_pos
                },
                "context": entity.context,
                "masked_value": entity.masked_value
            })
        
        return report
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return {
            "error": str(e),
            "analysis_summary": {"has_pii": False, "total_entities": 0}
        }

def generate_compliance_report(analysis_result: Dict[str, Any]) -> str:
    """Generate a compliance-focused report"""
    
    if "error" in analysis_result:
        return f"Analysis Error: {analysis_result['error']}"
    
    summary = analysis_result["analysis_summary"]
    entities_by_type = analysis_result["entities_by_type"]
    
    report_lines = [
        "=== PII COMPLIANCE ANALYSIS REPORT ===",
        f"Analysis Date: {__import__('datetime').datetime.now().isoformat()}",
        f"Processing Time: {summary['processing_time']:.3f} seconds",
        f"Analyzers Used: {', '.join(summary['analyzers_used'])}",
        "",
        "=== SUMMARY ===",
        f"PII Detected: {'YES' if summary['has_pii'] else 'NO'}",
        f"Total PII Entities: {summary['total_entities']}",
        f"High Confidence Entities: {summary['high_confidence_entities']}",
        ""
    ]
    
    if summary["has_pii"]:
        report_lines.append("=== PII BREAKDOWN BY TYPE ===")
        for pii_type, details in entities_by_type.items():
            report_lines.append(f"{pii_type}: {details['count']} entities")
            for entity in details["entities"]:
                confidence_indicator = "🔴" if entity["confidence"] > 0.8 else "🟡" if entity["confidence"] > 0.6 else "🟢"
                report_lines.append(f"  {confidence_indicator} {entity['masked_value']} (confidence: {entity['confidence']:.2f})")
        
        report_lines.extend([
            "",
            "=== RISK ASSESSMENT ===",
            f"HIGH RISK entities (>0.8 confidence): {len([e for e in analysis_result['entities_details'] if e['confidence'] > 0.8])}",
            f"MEDIUM RISK entities (0.6-0.8 confidence): {len([e for e in analysis_result['entities_details'] if 0.6 <= e['confidence'] <= 0.8])}",
            f"LOW RISK entities (<0.6 confidence): {len([e for e in analysis_result['entities_details'] if e['confidence'] < 0.6])}"
        ])
    else:
        report_lines.append("✅ No PII detected in the email content.")
    
    return "\n".join(report_lines)

def main():
    """Main function demonstrating usage"""
    
    # Sample email text (typically from OCR)
    sample_email_text = """
    From: john.doe@ubsbank.com
    To: jane.smith@client.com
    Subject: Account Information Update
    
    Dear Ms. Smith,
    
    Thank you for contacting UBS regarding your account. Here are the details you requested:
    
    Account Holder: Jane Smith
    Account Number: 1234567890123456
    Phone: +1-555-123-4567
    Email: jane.smith@client.com
    Address: 123 Main Street, New York, NY 10001
    
    Your IBAN is: GB29 NWBK 6016 1331 9268 19
    SWIFT Code: UBSWCHZH80A
    
    Please verify this information and let us know if there are any discrepancies.
    
    For security purposes, please do not share this information via email.
    
    Best regards,
    John Doe
    Senior Account Manager
    UBS Investment Bank
    Phone: +41-44-234-5678
    """
    
    print("Analyzing email for PII...")
    print("=" * 50)
    
    # Analyze the email
    config = create_analyzer_config()
    result = analyze_outlook_email(sample_email_text, config)
    
    # Generate and display compliance report
    compliance_report = generate_compliance_report(result)
    print(compliance_report)
    
    print("\n" + "=" * 50)
    print("MASKED EMAIL PREVIEW:")
    print("=" * 50)
    print(result.get("masked_text", "No masked text available"))
    
    # Optional: Save detailed results to JSON
    with open("pii_analysis_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: pii_analysis_results.json")

def analyze_single_analyzer_demo():
    """Demonstrate using individual analyzers"""
    
    sample_text = "Contact John Smith at john.smith@email.com or call 555-123-4567"
    
    print("\n=== INDIVIDUAL ANALYZER COMPARISON ===")
    
    analyzers = []
    
    # Try each analyzer individually
    try:
        presidio = PresidioPIIAnalyzer("presidio")
        analyzers.append(("Presidio", presidio))
    except ImportError:
        print("Presidio not available")
    
    try:
        transformer = TransformerNERAnalyzer("transformer")
        analyzers.append(("Transformer", transformer))
    except ImportError:
        print("Transformer not available")
    
    try:
        spacy_analyzer = SpacyNERAnalyzer("spacy")
        analyzers.append(("spaCy", spacy_analyzer))
    except ImportError:
        print("spaCy not available")
    
    try:
        gliner = GLiNERAnalyzer("gliner")
        analyzers.append(("GLiNER", gliner))
    except ImportError:
        print("GLiNER not available")
    
    for name, analyzer in analyzers:
        try:
            result = analyzer.analyze(sample_text)
            print(f"\n{name} Results:")
            print(f"  Entities found: {len(result.entities)}")
            for entity in result.entities:
                print(f"    {entity.pii_type.value}: {entity.text} (confidence: {entity.confidence:.2f})")
        except Exception as e:
            print(f"{name} failed: {e}")

def email_preprocessing_demo():
    """Demonstrate email-specific preprocessing"""
    
    raw_outlook_email = """
    From: sender@company.com
    To: recipient@client.com; cc-recipient@client.com
    Subject: Re: Confidential Information
    Sent: Monday, January 15, 2024 10:30 AM
    
    [TABLE]
    | Name | Account | Phone |
    |------|---------|-------|
    | John | 123456789 | 555-0123 |
    | Jane | 987654321 | 555-0456 |
    [/TABLE]
    
    • Bullet point with SSN: 123-45-6789
    • Another point with email: test@example.com
    
    [EMBEDDED_IMAGE: screenshot_001.png]
    
    Best regards,
    Account Manager
    """
    
    print("\n=== EMAIL PREPROCESSING DEMO ===")
    print("Raw email text:")
    print(raw_outlook_email[:200] + "..." if len(raw_outlook_email) > 200 else raw_outlook_email)
    
    # Custom email preprocessor
    def preprocess_outlook_email(text: str) -> str:
        """Custom preprocessing for Outlook emails"""
        import re
        
        # Remove email headers
        text = re.sub(r'^(From|To|Subject|Sent|Cc|Bcc):.*, '', text, flags=re.MULTILINE)
        
        # Extract table content
        table_pattern = r'\[TABLE\](.*?)\[/TABLE\]'
        tables = re.findall(table_pattern, text, re.DOTALL)
        for table in tables:
            # Simple table parsing - extract cell content
            cells = re.findall(r'\|\s*([^|]+)\s*\|', table)
            table_text = ' '.join(cells)
            text = text.replace(f'[TABLE]{table}[/TABLE]', table_text)
        
        # Remove image references
        text = re.sub(r'\[EMBEDDED_IMAGE:.*?\]', '[IMAGE_REMOVED]', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    processed_text = preprocess_outlook_email(raw_outlook_email)
    print(f"\nProcessed text:")
    print(processed_text)
    
    # Analyze processed text
    config = create_analyzer_config()
    result = analyze_outlook_email(processed_text, config)
    
    print(f"\nPII found in processed email: {result['analysis_summary']['total_entities']} entities")

if __name__ == "__main__":
    main()
    analyze_single_analyzer_demo()
    email_preprocessing_demo()