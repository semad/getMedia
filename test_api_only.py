#!/usr/bin/env python3
"""
Test script for API-only analysis functionality.
"""

import asyncio
import logging
from modules.analysis import AnalysisConfig, AnalysisOrchestrator

async def test_api_only_analysis():
    """Test analysis with API-only data sources."""
    print("🧪 Testing API-only analysis...")
    
    # Create configuration with file sources disabled
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
        
        print("✅ API-only analysis completed successfully!")
        print(f"📊 Analysis ID: {result.get('analysis_id', 'N/A')}")
        print(f"📈 Data sources: {len(result.get('data_sources', []))}")
        print(f"🔍 Analysis results: {list(result.get('analysis_results', {}).keys())}")
        print(f"📁 Output paths: {list(result.get('output_paths', {}).keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ API-only analysis failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_api_only_analysis())
