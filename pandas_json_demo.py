#!/usr/bin/env python3
"""
Pandas JSON Reading Demo

This script demonstrates different ways to read JSON files using pandas
instead of the standard json module.
"""

import pandas as pd
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_pandas_json_reading():
    """Demonstrate pandas JSON reading capabilities."""
    
    print("🐼 Pandas JSON Reading Demo")
    print("=" * 50)
    
    # Example 1: Basic pd.read_json()
    print("\n1️⃣ Basic pd.read_json()")
    print("-" * 30)
    
    # Create a sample JSON file for demonstration
    sample_data = {
        "messages": [
            {"id": 1, "text": "Hello", "timestamp": "2025-08-31T10:00:00"},
            {"id": 2, "text": "World", "timestamp": "2025-08-31T10:01:00"},
            {"id": 3, "text": "Test", "timestamp": "2025-08-31T10:02:00"}
        ],
        "channel_info": {
            "name": "test_channel",
            "total_messages": 3
        }
    }
    
    # Save sample data
    sample_file = Path("sample_data.json")
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"✅ Created sample file: {sample_file}")
    
    # Read with pandas
    try:
        df = pd.read_json(sample_file)
        print(f"✅ Pandas DataFrame shape: {df.shape}")
        print(f"✅ DataFrame columns: {list(df.columns)}")
        print(f"✅ First few rows:\n{df.head()}")
    except Exception as e:
        print(f"❌ Pandas read failed: {e}")
    
    # Example 2: Reading nested JSON
    print("\n2️⃣ Reading Nested JSON")
    print("-" * 30)
    
    try:
        # Read and normalize nested structure
        df_normalized = pd.json_normalize(sample_data, 
                                        record_path=['messages'],
                                        meta=['channel_info'])
        print(f"✅ Normalized DataFrame shape: {df_normalized.shape}")
        print(f"✅ Normalized columns: {list(df_normalized.columns)}")
        print(f"✅ Normalized data:\n{df_normalized}")
    except Exception as e:
        print(f"❌ JSON normalization failed: {e}")
    
    # Example 3: Reading with specific options
    print("\n3️⃣ Reading with Options")
    print("-" * 30)
    
    try:
        # Read with specific options
        df_options = pd.read_json(sample_file, 
                                 orient='records',
                                 lines=False)
        print(f"✅ Options DataFrame shape: {df_options.shape}")
        print(f"✅ Options data:\n{df_options}")
    except Exception as e:
        print(f"❌ Options read failed: {e}")
    
    # Example 4: Handling different JSON structures
    print("\n4️⃣ Handling Different Structures")
    print("-" * 30)
    
    # Create different JSON structures
    structures = {
        "list_of_dicts": [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"}
        ],
        "single_dict": {"key": "value", "number": 42},
        "nested_list": {"items": [1, 2, 3, 4, 5]}
    }
    
    for name, data in structures.items():
        print(f"\n📁 Structure: {name}")
        try:
            df_struct = pd.read_json(json.dumps(data))
            print(f"   ✅ Shape: {df_struct.shape}")
            print(f"   ✅ Type: {type(df_struct)}")
            print(f"   ✅ Data: {df_struct}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
    
    # Example 5: Performance comparison
    print("\n5️⃣ Performance Comparison")
    print("-" * 30)
    
    import time
    
    # Test pandas vs json performance
    iterations = 1000
    
    # Pandas timing
    start_time = time.time()
    for _ in range(iterations):
        pd.read_json(sample_file)
    pandas_time = time.time() - start_time
    
    # JSON timing
    start_time = time.time()
    for _ in range(iterations):
        with open(sample_file, 'r') as f:
            json.load(f)
    json_time = time.time() - start_time
    
    print(f"⏱️  Pandas time: {pandas_time:.4f}s ({iterations} iterations)")
    print(f"⏱️  JSON time: {json_time:.4f}s ({iterations} iterations)")
    print(f"📊 Pandas is {json_time/pandas_time:.2f}x {'slower' if pandas_time > json_time else 'faster'}")
    
    # Cleanup
    sample_file.unlink()
    print(f"\n🧹 Cleaned up sample file")
    
    print("\n🎉 Demo completed!")
    print("\n💡 Key Benefits of Pandas JSON:")
    print("   • Direct DataFrame conversion")
    print("   • Built-in data analysis tools")
    print("   • Better handling of large files")
    print("   • JSON normalization capabilities")
    print("   • Easy data filtering and manipulation")

if __name__ == "__main__":
    demo_pandas_json_reading()
