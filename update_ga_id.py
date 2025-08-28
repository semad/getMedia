#!/usr/bin/env python3
"""
Script to update Google Analytics measurement ID across all HTML files.
Usage: python update_ga_id.py YOUR_GA_MEASUREMENT_ID
"""

import sys
import os
import re
from pathlib import Path

def update_ga_id(ga_id):
    """Update GA measurement ID in all HTML files."""
    if not ga_id or not ga_id.strip():
        print("❌ Error: Please provide a valid Google Analytics measurement ID")
        print("Usage: python update_ga_id.py G-XXXXXXXXXX")
        sys.exit(1)
    
    # Remove any 'G-' prefix if user included it
    ga_id = ga_id.strip().replace('G-', '')
    
    # Add 'G-' prefix if not present
    if not ga_id.startswith('G-'):
        ga_id = f'G-{ga_id}'
    
    print(f"🔄 Updating Google Analytics ID to: {ga_id}")
    
    # Find all HTML files
    html_files = list(Path('.').glob('*.html'))
    
    if not html_files:
        print("❌ No HTML files found in current directory")
        sys.exit(1)
    
    updated_files = []
    
    for html_file in html_files:
        print(f"📄 Processing: {html_file}")
        
        try:
            # Read file content
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace GA_MEASUREMENT_ID placeholder
            if 'GA_MEASUREMENT_ID' in content:
                new_content = content.replace('GA_MEASUREMENT_ID', ga_id)
                
                # Write updated content
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                updated_files.append(html_file)
                print(f"✅ Updated: {html_file}")
            else:
                print(f"⚠️  No GA placeholder found in: {html_file}")
                
        except Exception as e:
            print(f"❌ Error processing {html_file}: {e}")
    
    print(f"\n🎉 Successfully updated {len(updated_files)} files:")
    for file in updated_files:
        print(f"   - {file}")
    
    print(f"\n📊 Your Google Analytics ID '{ga_id}' is now active!")
    print("💡 Remember to:")
    print("   1. Set up your Google Analytics property")
    print("   2. Wait 24-48 hours for data to appear")
    print("   3. Test with Google Analytics Real-Time reports")

def main():
    if len(sys.argv) != 2:
        print("❌ Error: Please provide your Google Analytics measurement ID")
        print("Usage: python update_ga_id.py G-XXXXXXXXXX")
        print("\nExample:")
        print("  python update_ga_id.py G-ABC123DEF4")
        print("  python update_ga_id.py ABC123DEF4")
        sys.exit(1)
    
    ga_id = sys.argv[1]
    update_ga_id(ga_id)

if __name__ == "__main__":
    main()
