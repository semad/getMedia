#!/usr/bin/env python3
"""
Test script to verify file data sources output to correct directory.
"""

import asyncio
import logging
from modules.analysis import AnalysisConfig, AnalysisOrchestrator

async def test_file_output_paths():
    """Test that file data sources output to file_messages directory."""
    print("🧪 Testing file output paths...")
    
    # Create configuration with file-only sources
    config = AnalysisConfig(
        enable_file_source=True,   # Enable file sources
        enable_api_source=False,   # Disable API sources
        enable_diff_analysis=False,  # Disable diff analysis
        verbose=True,
        chunk_size=5000
    )
    
    print(f"Configuration: {config}")
    
    # Create orchestrator
    orchestrator = AnalysisOrchestrator(config)
    
    # Test with specific channel
    try:
        result = await orchestrator.run_comprehensive_analysis(
            channels=["@books"],
            analysis_types=["filename", "filesize", "message"]
        )
        
        print("✅ File output paths test completed!")
        print(f"📊 Analysis ID: {result.get('analysis_id', 'N/A')}")
        print(f"📈 Data sources: {len(result.get('data_sources', []))}")
        print(f"🔍 Analysis results: {list(result.get('analysis_results', {}).keys())}")
        print(f"📁 Output paths: {list(result.get('output_paths', {}).keys())}")
        
        # Check if output paths contain file_messages
        output_paths = result.get('output_paths', {})
        has_file_messages = any('file_messages' in str(path) for path in output_paths.values())
        print(f"🗂️  Contains file_messages paths: {has_file_messages}")
        
        if has_file_messages:
            print("✅ SUCCESS: File data sources correctly output to file_messages directory!")
        else:
            print("❌ FAILURE: File data sources not outputting to file_messages directory!")
        
        return has_file_messages
        
    except Exception as e:
        print(f"❌ File output paths test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_file_output_paths())
