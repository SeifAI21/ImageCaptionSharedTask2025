"""
Arabic text processing utilities for evaluation
"""

import re
import pandas as pd
from typing import List, Optional
import evaluation_config as config


class ArabicTextProcessor:
    """Advanced Arabic text processing for evaluation metrics."""
    
    def __init__(self):
        """Initialize with Arabic-specific patterns."""
        self.patterns = config.ARABIC_PATTERNS
    
    def normalize_arabic_text(self, text: str) -> str:
        """
        Advanced Arabic text normalization.
        
        Args:
            text: Input Arabic text
            
        Returns:
            Normalized Arabic text
        """
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).strip()
        
        # Remove Arabic diacritics (Tashkeel)
        text = self.patterns["diacritics"].sub('', text)
        
        # Normalize Arabic letters
        text = self.patterns["alef_patterns"].sub('ا', text)  # Normalize all Alef variants
        text = self.patterns["yeh_patterns"].sub('ي', text)    # Normalize Yeh variants  
        text = self.patterns["teh_patterns"].sub('ة', text)    # Normalize Teh Marbuta
        
        # Remove punctuation (both Arabic and general)
        text = self.patterns["arabic_punctuation"].sub(' ', text)
        text = self.patterns["general_punctuation"].sub(' ', text)
        
        # Remove numbers and English characters
        text = re.sub(r'[0-9a-zA-Z]', ' ', text)
        
        # Remove extra whitespace
        text = self.patterns["multiple_spaces"].sub(' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def clean_for_rouge(self, text: str) -> str:
        """
        Clean text specifically for ROUGE evaluation.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text for ROUGE
        """
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).strip()
        
        # Remove diacritics
        text = self.patterns["diacritics"].sub('', text)
        
        # Normalize Arabic letters
        text = self.patterns["alef_pateh_patternstterns"].sub('ا', text)
        text = self.patterns["yeh_patterns"].sub('ي', text)
        text = self.patterns["teh_patterns"].sub('ة', text)
        
        # Keep some punctuation for ROUGE but clean excessive ones
        text = re.sub(r'[0-9a-zA-Z]', ' ', text)
        text = self.patterns["multiple_spaces"].sub(' ', text)
        
        return text.strip()
    
    def advanced_tokenize_arabic(self, text: str) -> List[str]:
        """
        Advanced Arabic tokenization with normalization.
        
        Args:
            text: Input Arabic text
            
        Returns:
            List of normalized tokens
        """
        normalized_text = self.normalize_arabic_text(text)
        
        if not normalized_text:
            return []
        
        # Split by whitespace
        tokens = normalized_text.split()
        
        # Filter out very short tokens (less than min_token_length)
        min_length = config.TFIDF_CONFIG["min_token_length"]
        tokens = [token for token in tokens if len(token) >= min_length]
        
        return tokens
    
    def get_text_statistics(self, text: str) -> dict:
        """
        Get comprehensive text statistics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        if pd.isna(text) or text == '':
            return {
                'char_count': 0,
                'word_count': 0,
                'token_count': 0,
                'unique_tokens': 0,
                'avg_word_length': 0.0
            }
        
        normalized = self.normalize_arabic_text(text)
        tokens = self.advanced_tokenize_arabic(text)
        
        stats = {
            'char_count': len(str(text)),
            'char_count_normalized': len(normalized),
            'word_count': len(str(text).split()),
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'avg_word_length': sum(len(token) for token in tokens) / len(tokens) if tokens else 0.0
        }
        
        return stats
    
    def validate_arabic_text(self, text: str) -> dict:
        """
        Validate Arabic text quality.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with validation results
        """
        if pd.isna(text) or text == '':
            return {
                'is_valid': False,
                'has_arabic': False,
                'has_content': False,
                'error': 'Empty or null text'
            }
        
        text_str = str(text).strip()
        
        # Check if text has Arabic characters
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        has_arabic = bool(arabic_pattern.search(text_str))
        
        # Check if text has meaningful content after normalization
        normalized = self.normalize_arabic_text(text_str)
        has_content = len(normalized) > 0
        
        # Check for minimum length
        tokens = self.advanced_tokenize_arabic(text_str)
        has_min_tokens = len(tokens) >= 1
        
        validation = {
            'is_valid': has_arabic and has_content and has_min_tokens,
            'has_arabic': has_arabic,
            'has_content': has_content,
            'has_min_tokens': has_min_tokens,
            'token_count': len(tokens),
            'error': None
        }
        
        if not validation['is_valid']:
            errors = []
            if not has_arabic:
                errors.append('No Arabic characters found')
            if not has_content:
                errors.append('No meaningful content after normalization')
            if not has_min_tokens:
                errors.append('Insufficient tokens')
            validation['error'] = '; '.join(errors)
        
        return validation


def create_text_processor() -> ArabicTextProcessor:
    """Factory function to create text processor."""
    return ArabicTextProcessor()


def batch_normalize_texts(texts: List[str]) -> List[str]:
    """
    Normalize a batch of texts.
    
    Args:
        texts: List of input texts
        
    Returns:
        List of normalized texts
    """
    processor = create_text_processor()
    return [processor.normalize_arabic_text(text) for text in texts]


def batch_validate_texts(texts: List[str]) -> List[dict]:
    """
    Validate a batch of texts.
    
    Args:
        texts: List of input texts
        
    Returns:
        List of validation results
    """
    processor = create_text_processor()
    return [processor.validate_arabic_text(text) for text in texts]