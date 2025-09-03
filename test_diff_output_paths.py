#!/usr/bin/env python3
"""
Test script to verify diff analysis output goes to diff_messages directory.
"""

import asyncio
import logging
from modules.analysis import AnalysisConfig, AnalysisOrchestrator

async def test_diff_output_paths():
    """Test that diff analysis outputs to diff_messages directory."""
    print("🧪 Testing diff analysis output paths...")
    
    # Create configuration with both file and API sources enabled (diff analysis)
    config = AnalysisConfig(
        enable_file_source=True,   # Enable file sources
        enable_api_source=True,    # Enable API sources
        enable_diff_analysis=True, # Enable diff analysis
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
        
        print("✅ Diff analysis output paths test completed!")
        print(f"📊 Analysis ID: {result.get('analysis_id', 'N/A')}")
        print(f"📈 Data sources: {len(result.get('data_sources', []))}")
        print(f"🔍 Analysis results: {list(result.get('analysis_results', {}).keys())}")
        print(f"📁 Output paths: {list(result.get('output_paths', {}).keys())}")
        
        # Check if output paths contain diff_messages
        output_paths = result.get('output_paths', {})
        has_diff_messages = any('diff_messages' in str(path) for path in output_paths.values())
        print(f"🗂️  Contains diff_messages paths: {has_diff_messages}")
        
        if has_diff_messages:
            print("✅ SUCCESS: Diff analysis correctly output to diff_messages directory!")
        else:
            print("❌ FAILURE: Diff analysis not outputting to diff_messages directory!")
        
        return has_diff_messages
        
    except Exception as e:
        print(f"❌ Diff analysis output paths test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_diff_output_paths())
