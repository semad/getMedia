#!/usr/bin/env python3
"""
Test script to verify API data sources output to correct directory.
"""

import asyncio
import logging
from modules.analysis import AnalysisConfig, AnalysisOrchestrator

async def test_api_output_paths():
    """Test that API data sources output to db_messages directory."""
    print("🧪 Testing API output paths...")
    
    # Create configuration with API-only sources
    config = AnalysisConfig(
        enable_file_source=False,  # Disable file sources
        enable_api_source=True,    # Enable API sources
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
        
        print("✅ API output paths test completed!")
        print(f"📊 Analysis ID: {result.get('analysis_id', 'N/A')}")
        print(f"📈 Data sources: {len(result.get('data_sources', []))}")
        print(f"🔍 Analysis results: {list(result.get('analysis_results', {}).keys())}")
        print(f"📁 Output paths: {list(result.get('output_paths', {}).keys())}")
        
        # Check if output paths contain db_messages
        output_paths = result.get('output_paths', {})
        has_db_messages = any('db_messages' in str(path) for path in output_paths.values())
        print(f"🗂️  Contains db_messages paths: {has_db_messages}")
        
        if has_db_messages:
            print("✅ SUCCESS: API data sources correctly output to db_messages directory!")
        else:
            print("❌ FAILURE: API data sources not outputting to db_messages directory!")
        
        return has_db_messages
        
    except Exception as e:
        print(f"❌ API output paths test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_api_output_paths())
