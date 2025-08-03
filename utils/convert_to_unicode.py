#!/usr/bin/env python3
"""
Script to convert CSV file to Unicode-compatible format and remove incompatible characters.
"""

import csv
import re
import sys

def clean_text(text):
    """
    Clean text by removing or replacing incompatible Unicode characters.
    """
    if not text:
        return text
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Replace common problematic characters
    replacements = {
        '\u2018': "'",  # Left single quotation mark
        '\u2019': "'",  # Right single quotation mark
        '\u201C': '"',  # Left double quotation mark
        '\u201D': '"',  # Right double quotation mark
        '\u2013': '-',  # En dash
        '\u2014': '--', # Em dash
        '\u2022': '•',  # Bullet
        '\u2026': '...', # Horizontal ellipsis
        '\u00A0': ' ',  # Non-breaking space
        '\u00B0': '°',  # Degree sign
        '\u00B1': '±',  # Plus-minus sign
        '\u00B2': '²',  # Superscript two
        '\u00B3': '³',  # Superscript three
        '\u00BC': '¼',  # Fraction one quarter
        '\u00BD': '½',  # Fraction one half
        '\u00BE': '¾',  # Fraction three quarters
        '\u00F7': '÷',  # Division sign
        '\u00D7': '×',  # Multiplication sign
        '\u2260': '≠',  # Not equal to
        '\u2264': '≤',  # Less than or equal to
        '\u2265': '≥',  # Greater than or equal to
        '\u221E': '∞',  # Infinity
        '\u03C0': 'π',  # Greek small letter pi
        '\u03A3': 'Σ',  # Greek capital letter sigma
        '\u03B1': 'α',  # Greek small letter alpha
        '\u03B2': 'β',  # Greek small letter beta
        '\u03B3': 'γ',  # Greek small letter gamma
        '\u03B4': 'δ',  # Greek small letter delta
        '\u03B5': 'ε',  # Greek small letter epsilon
        '\u03B6': 'ζ',  # Greek small letter zeta
        '\u03B7': 'η',  # Greek small letter eta
        '\u03B8': 'θ',  # Greek small letter theta
        '\u03B9': 'ι',  # Greek small letter iota
        '\u03BA': 'κ',  # Greek small letter kappa
        '\u03BB': 'λ',  # Greek small letter lambda
        '\u03BC': 'μ',  # Greek small letter mu
        '\u03BD': 'ν',  # Greek small letter nu
        '\u03BE': 'ξ',  # Greek small letter xi
        '\u03BF': 'ο',  # Greek small letter omicron
        '\u03C0': 'π',  # Greek small letter pi
        '\u03C1': 'ρ',  # Greek small letter rho
        '\u03C2': 'ς',  # Greek small letter final sigma
        '\u03C3': 'σ',  # Greek small letter sigma
        '\u03C4': 'τ',  # Greek small letter tau
        '\u03C5': 'υ',  # Greek small letter upsilon
        '\u03C6': 'φ',  # Greek small letter phi
        '\u03C7': 'χ',  # Greek small letter chi
        '\u03C8': 'ψ',  # Greek small letter psi
        '\u03C9': 'ω',  # Greek small letter omega
    }
    
    # Apply replacements
    for old_char, new_char in replacements.items():
        text = text.replace(old_char, new_char)
    
    # Remove any remaining control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Normalize Unicode characters
    import unicodedata
    text = unicodedata.normalize('NFKC', text)
    
    return text.strip()

def convert_csv_to_unicode(input_file, output_file):
    """
    Convert CSV file to Unicode-compatible format.
    """
    try:
        # Read the original file with different encodings to find the best one
        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        data = None
        
        for encoding in encodings_to_try:
            try:
                with open(input_file, 'r', encoding=encoding, errors='replace') as f:
                    data = f.read()
                print(f"Successfully read file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if data is None:
            print("Error: Could not read file with any of the attempted encodings")
            return False
        
        # Parse CSV data
        lines = data.split('\n')
        cleaned_rows = []
        
        for line in lines:
            if not line.strip():
                continue
                
            # Clean the line
            cleaned_line = clean_text(line)
            if cleaned_line:
                cleaned_rows.append(cleaned_line)
        
        # Write the cleaned data to output file
        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            for row in cleaned_rows:
                f.write(row + '\n')
        
        print(f"Successfully converted {input_file} to {output_file}")
        print(f"Processed {len(cleaned_rows)} rows")
        return True
        
    except Exception as e:
        print(f"Error converting file: {e}")
        return False

if __name__ == "__main__":
    input_file = "data/peter_lynch_qna.csv"
    output_file = "data/peter_lynch_qna_unicode.csv"
    
    success = convert_csv_to_unicode(input_file, output_file)
    
    if success:
        print(f"\nConversion completed successfully!")
        print(f"Original file: {input_file}")
        print(f"Unicode-compatible file: {output_file}")
    else:
        print("Conversion failed!")
        sys.exit(1) 