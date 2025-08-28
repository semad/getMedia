#!/usr/bin/env python3
"""
Document Filename Analysis Script

This script analyzes PDF and EPUB documents from Telegram data to identify:
- Unique vs non-unique filenames
- Potential duplicates
- Naming patterns
- File size variations for same-named files
"""

import json
import pandas as pd
from collections import Counter, defaultdict
import click
import logging
from pathlib import Path
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    def __init__(self, json_file_path):
        """Initialize the analyzer with the JSON file path."""
        self.json_file_path = Path(json_file_path)
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and prepare the data from JSON file."""
        try:
            logger.info(f"Loading data from {self.json_file_path}")
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle the structured JSON format with export_info and data
            if isinstance(data, dict) and 'export_info' in data:
                # Extract the actual data array
                if 'data' in data:
                    data_array = data['data']
                else:
                    # If no data key, try to find the messages array
                    for key, value in data.items():
                        if key != 'export_info' and isinstance(value, list):
                            data_array = value
                            break
                    else:
                        raise ValueError("Could not find data array in JSON file")
                
                logger.info(f"Found structured JSON with {len(data_array)} messages")
                self.df = pd.DataFrame(data_array)
            else:
                # Handle simple array format
                self.df = pd.DataFrame(data)
            
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_document_names(self):
        """Analyze PDF and EPUB document filenames."""
        logger.info("Analyzing document filenames...")
        
        # Filter for PDF and EPUB documents
        doc_data = self.df[
            (self.df['media_type'] == 'document') & 
            (self.df['file_name'].notna()) &
            (self.df['file_name'] != '')
        ].copy()
        
        if len(doc_data) == 0:
            logger.warning("No documents with filenames found")
            return None
        
        # Filter for PDF and EPUB files
        pdf_epub_data = doc_data[
            doc_data['file_name'].str.lower().str.endswith(('.pdf', '.epub'), na=False)
        ].copy()
        
        if len(pdf_epub_data) == 0:
            logger.warning("No PDF or EPUB files found")
            return None
        
        logger.info(f"Found {len(pdf_epub_data)} PDF/EPUB documents")
        
        # Analyze filenames
        filename_counts = Counter(pdf_epub_data['file_name'])
        
        # Separate unique and non-unique filenames
        unique_filenames = {name: count for name, count in filename_counts.items() if count == 1}
        non_unique_filenames = {name: count for name, count in filename_counts.items() if count > 1}
        
        # Analyze file types
        pdf_files = pdf_epub_data[pdf_epub_data['file_name'].str.lower().str.endswith('.pdf', na=False)]
        epub_files = pdf_epub_data[pdf_epub_data['file_name'].str.lower().str.endswith('.epub', na=False)]
        
        # Create detailed analysis
        analysis = {
            'total_documents': len(pdf_epub_data),
            'pdf_count': len(pdf_files),
            'epub_count': len(epub_files),
            'unique_filenames_count': len(unique_filenames),
            'non_unique_filenames_count': len(non_unique_filenames),
            'total_unique_names': len(set(pdf_epub_data['file_name'])),
            'duplicate_ratio': len(non_unique_filenames) / len(set(pdf_epub_data['file_name'])) * 100,
            'filename_counts': filename_counts,
            'unique_filenames': unique_filenames,
            'non_unique_filenames': non_unique_filenames,
            'pdf_files': pdf_files,
            'epub_files': epub_files
        }
        
        return analysis
    
    def analyze_duplicates(self, analysis):
        """Analyze duplicate filenames in detail."""
        if not analysis or not analysis['non_unique_filenames']:
            return None
        
        logger.info("Analyzing duplicate filenames...")
        
        duplicates_analysis = {}
        
        for filename, count in analysis['non_unique_filenames'].items():
            # Get all instances of this filename
            instances = self.df[
                (self.df['file_name'] == filename) & 
                (self.df['media_type'] == 'document')
            ].copy()
            
            # Analyze differences between instances
            duplicate_info = {
                'count': count,
                'instances': instances,
                'size_variations': instances['file_size'].unique().tolist() if 'file_size' in instances.columns else [],
                'channel_sources': instances['channel_username'].unique().tolist() if 'channel_username' in instances.columns else [],
                'creator_variations': instances['creator_username'].unique().tolist() if 'creator_username' in instances.columns else [],
                'date_range': {
                    'earliest': instances['date'].min() if 'date' in instances.columns else None,
                    'latest': instances['date'].max() if 'date' in instances.columns else None
                } if 'date' in instances.columns else {},
                'view_variations': instances['views'].unique().tolist() if 'views' in instances.columns else [],
                'forward_variations': instances['forwards'].unique().tolist() if 'forwards' in instances.columns else []
            }
            
            duplicates_analysis[filename] = duplicate_info
        
        return duplicates_analysis
    
    def generate_report(self, analysis, duplicates_analysis):
        """Generate a comprehensive report."""
        if not analysis:
            logger.error("No analysis data available")
            return
        
        logger.info("Generating document analysis report...")
        
        print("\n" + "="*80)
        print("📚 DOCUMENT FILENAME ANALYSIS REPORT")
        print("="*80)
        
        # Summary Statistics
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"  Total PDF/EPUB Documents: {analysis['total_documents']:,}")
        print(f"  PDF Files: {analysis['pdf_count']:,}")
        print(f"  EPUB Files: {analysis['epub_count']:,}")
        print(f"  Unique Filenames: {analysis['unique_filenames_count']:,}")
        print(f"  Non-Unique Filenames: {analysis['non_unique_filenames_count']:,}")
        print(f"  Total Unique Names: {analysis['total_unique_names']:,}")
        print(f"  Duplicate Ratio: {analysis['duplicate_ratio']:.1f}%")
        
        # File Type Breakdown
        print(f"\n📁 FILE TYPE BREAKDOWN:")
        print(f"  PDF Files: {analysis['pdf_count']:,} ({analysis['pdf_count']/analysis['total_documents']*100:.1f}%)")
        print(f"  EPUB Files: {analysis['epub_count']:,} ({analysis['epub_count']/analysis['total_documents']*100:.1f}%)")
        
        # Duplicate Analysis
        if duplicates_analysis:
            print(f"\n🔄 DUPLICATE FILENAME ANALYSIS:")
            print(f"  Files with Duplicate Names: {len(duplicates_analysis)}")
            
            # Show top duplicates
            top_duplicates = sorted(
                [(name, info['count']) for name, info in duplicates_analysis.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            print(f"\n  Top 10 Most Duplicated Filenames:")
            for i, (filename, count) in enumerate(top_duplicates, 1):
                print(f"    {i:2d}. {filename[:60]:<60} (×{count})")
            
            # Analyze duplicate patterns
            print(f"\n  Duplicate Patterns:")
            size_variations = sum(1 for info in duplicates_analysis.values() if len(info['size_variations']) > 1)
            channel_variations = sum(1 for info in duplicates_analysis.values() if len(info['channel_sources']) > 1)
            creator_variations = sum(1 for info in duplicates_analysis.values() if len(info['creator_variations']) > 1)
            
            print(f"    Files with different sizes: {size_variations}")
            print(f"    Files from different channels: {channel_variations}")
            print(f"    Files from different creators: {creator_variations}")
        
        # Unique Filenames Sample
        if analysis['unique_filenames']:
            print(f"\n✅ UNIQUE FILENAMES SAMPLE (showing first 10):")
            unique_sample = list(analysis['unique_filenames'].keys())[:10]
            for i, filename in enumerate(unique_sample, 1):
                print(f"    {i:2d}. {filename}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if analysis['duplicate_ratio'] > 10:
            print(f"  ⚠️  High duplicate ratio ({analysis['duplicate_ratio']:.1f}%) - consider deduplication")
        else:
            print(f"  ✅ Low duplicate ratio ({analysis['duplicate_ratio']:.1f}%) - good filename uniqueness")
        
        if duplicates_analysis:
            print(f"  🔍 Review {len(duplicates_analysis)} files with duplicate names")
            print(f"  📏 Check for files with same names but different sizes (potential quality variations)")
            print(f"  🌐 Verify files from different channels aren't duplicates")
        
        print("\n" + "="*80)
    
    def export_duplicates_csv(self, duplicates_analysis, output_file='duplicate_documents.csv'):
        """Export duplicate analysis to CSV."""
        if not duplicates_analysis:
            logger.warning("No duplicate data to export")
            return
        
        logger.info(f"Exporting duplicate analysis to {output_file}")
        
        # Prepare data for CSV export
        csv_data = []
        
        for filename, info in duplicates_analysis.items():
            for idx, instance in info['instances'].iterrows():
                row = {
                    'filename': filename,
                    'duplicate_count': info['count'],
                    'file_size': instance.get('file_size', 'N/A'),
                    'channel_username': instance.get('channel_username', 'N/A'),
                    'creator_username': instance.get('creator_username', 'N/A'),
                    'date': instance.get('date', 'N/A'),
                    'views': instance.get('views', 'N/A'),
                    'forwards': instance.get('forwards', 'N/A'),
                    'message_id': instance.get('message_id', 'N/A')
                }
                csv_data.append(row)
        
        # Create DataFrame and export
        df_export = pd.DataFrame(csv_data)
        df_export.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Exported {len(csv_data)} duplicate instances to {output_file}")
        
        return output_file

@click.command()
@click.option('--input-file', '-i', default='telegram_messages_export.json', 
              help='Input JSON file path')
@click.option('--export-csv', '-e', is_flag=True, help='Export duplicate analysis to CSV')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(input_file, export_csv, verbose):
    """Analyze PDF and EPUB document filenames for uniqueness and duplicates."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Check if input file exists
        if not Path(input_file).exists():
            logger.error(f"Input file not found: {input_file}")
            return
        
        # Create analyzer and run analysis
        analyzer = DocumentAnalyzer(input_file)
        analysis = analyzer.analyze_document_names()
        
        if not analysis:
            logger.error("No document analysis could be performed")
            return
        
        # Analyze duplicates
        duplicates_analysis = analyzer.analyze_duplicates(analysis)
        
        # Generate report
        analyzer.generate_report(analysis, duplicates_analysis)
        
        # Export CSV if requested
        if export_csv and duplicates_analysis:
            csv_file = analyzer.export_duplicates_csv(duplicates_analysis)
            logger.info(f"✅ Duplicate analysis exported to: {csv_file}")
        
        logger.info("✅ Document filename analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error analyzing documents: {e}")
        raise

if __name__ == '__main__':
    main()
