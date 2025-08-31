#!/usr/bin/env python3
"""
Script to convert legacy Telegram JSON files to proper DataFrame format.

This script reads the old format where messages are stored as string representations
of TelegramMessage objects and converts them to a structured DataFrame-compatible format.

Usage:
    python convert_legacy_json.py [input_file] [output_file]
    python convert_legacy_json.py --batch --input-dir ./reports/collections
    python convert_legacy_json.py --help
"""

import json
import os
import sys
import argparse
from datetime import datetime
import re
from pathlib import Path
import glob


def parse_telegram_message_string(msg_str):
    """Parse a string representation of a TelegramMessage object."""
    try:
        # Remove the TelegramMessage( wrapper
        content = msg_str.strip()
        if content.startswith('TelegramMessage(') and content.endswith(')'):
            content = content[16:-1]  # Remove 'TelegramMessage(' and ')'
        
        # Parse the key-value pairs
        parsed = {}
        current_key = None
        current_value = ""
        paren_count = 0
        in_quotes = False
        quote_char = None
        
        i = 0
        while i < len(content):
            char = content[i]
            
            if char in ['"', "'"] and (i == 0 or content[i-1] != '\\'):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                current_value += char
            elif char == '(' and not in_quotes:
                paren_count += 1
                current_value += char
            elif char == ')' and not in_quotes:
                paren_count -= 1
                current_value += char
            elif char == ',' and paren_count == 0 and not in_quotes:
                # End of current key-value pair
                if current_key is not None:
                    current_value = current_value.strip()
                    # Handle datetime objects
                    if 'datetime.datetime(' in current_value:
                        parsed[current_key] = safe_eval_datetime(current_value)
                    else:
                        # Handle different value types
                        if current_value == 'None':
                            parsed[current_key] = None
                        elif current_value == 'True':
                            parsed[current_key] = True
                        elif current_value == 'False':
                            parsed[current_key] = False
                        elif current_value.startswith('"') and current_value.endswith('"'):
                            parsed[current_key] = current_value[1:-1]
                        elif current_value.startswith("'") and current_value.endswith("'"):
                            parsed[current_key] = current_value[1:-1]
                        else:
                            # Try to convert to number
                            try:
                                if '.' in current_value:
                                    parsed[current_key] = float(current_value)
                                else:
                                    parsed[current_key] = int(current_value)
                            except ValueError:
                                parsed[current_key] = current_value
                
                # Reset for next pair
                current_key = None
                current_value = ""
            elif char == '=' and current_key is None and paren_count == 0 and not in_quotes:
                # Found the equals sign, current_value contains the key
                current_key = current_value.strip()
                current_value = ""
            else:
                current_value += char
            
            i += 1
        
        # Handle the last key-value pair
        if current_key is not None:
            current_value = current_value.strip()
            # Handle datetime objects
            if 'datetime.datetime(' in current_value:
                parsed[current_key] = safe_eval_datetime(current_value)
            else:
                # Handle different value types
                if current_value == 'None':
                    parsed[current_key] = None
                elif current_value == 'True':
                    parsed[current_key] = True
                elif current_value == 'False':
                    parsed[current_key] = False
                elif current_value.startswith('"') and current_value.endswith('"'):
                    parsed[current_key] = current_value[1:-1]
                elif current_value.startswith("'") and current_value.endswith("'"):
                    parsed[current_key] = current_value[1:-1]
                else:
                    # Try to convert to number
                    try:
                        if '.' in current_value:
                            parsed[current_key] = float(current_value)
                        else:
                            parsed[current_key] = int(current_value)
                    except ValueError:
                        parsed[current_key] = current_value
        
        return parsed
    except Exception as e:
        print(f"Error parsing message: {e}")
        return None


def safe_eval_datetime(dt_str):
    """Safely evaluate datetime string using eval."""
    try:
        # Replace datetime.timezone.utc with a string representation
        dt_str = dt_str.replace('datetime.timezone.utc', '"UTC"')
        
        # Use eval to parse the datetime
        result = eval(dt_str)
        
        # Convert to ISO format if it's a datetime object
        if hasattr(result, 'isoformat'):
            return result.isoformat()
        else:
            return str(result)
    except Exception as e:
        # If eval fails, try to parse manually
        try:
            # Extract datetime components using regex
            # Try pattern with seconds first
            pattern_with_seconds = r'datetime\.datetime\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)'
            match = re.search(pattern_with_seconds, dt_str)
            if match:
                year, month, day, hour, minute, second = map(int, match.groups())
                dt = datetime(year, month, day, hour, minute, second)
                return dt.isoformat()
            
            # Try pattern without seconds
            pattern_without_seconds = r'datetime\.datetime\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)'
            match = re.search(pattern_without_seconds, dt_str)
            if match:
                year, month, day, hour, minute = map(int, match.groups())
                dt = datetime(year, month, day, hour, minute)
                return dt.isoformat()
            
        except Exception as regex_error:
            pass
        return None


def convert_legacy_json_to_dataframe_format(input_file, output_file=None, verbose=True):
    """Convert legacy JSON file to DataFrame-compatible format."""
    
    # Read the input file
    if verbose:
        print(f"📖 Reading legacy JSON file: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    # Check if this is a legacy format
    if 'messages' not in data:
        print("❌ File doesn't contain 'messages' field")
        return False
    
    messages = data['messages']
    if not messages:
        print("❌ No messages found in file")
        return False
    
    # Check if messages are string representations
    if not isinstance(messages[0], str):
        if verbose:
            print("✅ File is already in proper format")
        return True
    
    if verbose:
        print(f"🔧 Converting {len(messages)} string representations to structured format...")
    
    # Parse each message
    structured_messages = []
    success_count = 0
    error_count = 0
    
    for i, msg_str in enumerate(messages):
        try:
            parsed = parse_telegram_message_string(msg_str)
            if parsed:
                structured_messages.append(parsed)
                success_count += 1
            else:
                if verbose:
                    print(f"⚠️  Failed to parse message {i+1}")
                error_count += 1
        except Exception as e:
            if verbose:
                print(f"❌ Error parsing message {i+1}: {e}")
            error_count += 1
    
    if verbose:
        print(f"✅ Successfully parsed {success_count} messages")
        if error_count > 0:
            print(f"⚠️  Failed to parse {error_count} messages")
    
    if not structured_messages:
        print("❌ No messages were successfully parsed")
        return False
    
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.parent / f"{input_path.stem}_converted{input_path.suffix}"
    
    # Prepare export data
    export_data = {
        'metadata': {
            'collected_at': data.get('metadata', {}).get('collected_at', datetime.now().isoformat()),
            'channels': data.get('metadata', {}).get('channels', []),
            'total_messages': len(structured_messages),
            'data_format': 'structured_dataframe',
            'fields': list(structured_messages[0].keys()) if structured_messages else [],
            'conversion_info': {
                'converted_at': datetime.now().isoformat(),
                'original_format': 'legacy_string_representation',
                'successful_conversions': success_count,
                'failed_conversions': error_count
            }
        },
        'messages': structured_messages
    }
    
    # Write the converted file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        if verbose:
            print(f"💾 Converted data saved to: {output_file}")
            print(f"📊 Data now has {len(structured_messages[0]) if structured_messages else 0} fields")
            print(f"🔢 Sample fields: {', '.join(list(structured_messages[0].keys())[:10])}{'...' if len(structured_messages[0]) > 10 else ''}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving converted file: {e}")
        return False


def batch_convert(input_dir, pattern="*.json", verbose=True):
    """Convert multiple legacy JSON files in a directory."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    # Find all JSON files
    json_files = list(input_path.glob(pattern))
    if not json_files:
        print(f"❌ No JSON files found in {input_dir} matching pattern: {pattern}")
        return False
    
    print(f"🔍 Found {len(json_files)} JSON files to process")
    
    success_count = 0
    error_count = 0
    
    for json_file in json_files:
        print(f"\n{'='*60}")
        print(f"📁 Processing: {json_file.name}")
        
        try:
            success = convert_legacy_json_to_dataframe_format(json_file, verbose=verbose)
            if success:
                success_count += 1
                print(f"✅ Successfully converted: {json_file.name}")
            else:
                error_count += 1
                print(f"❌ Failed to convert: {json_file.name}")
        except Exception as e:
            error_count += 1
            print(f"❌ Error processing {json_file.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"🎉 Batch conversion completed!")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {error_count}")
    
    return success_count > 0


def main():
    """Main function to run the conversion."""
    
    parser = argparse.ArgumentParser(
        description="Convert legacy Telegram JSON files to DataFrame-compatible format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_legacy_json.py input.json
  python convert_legacy_json.py input.json output.json
  python convert_legacy_json.py --batch --input-dir ./reports/collections
  python convert_legacy_json.py --batch --input-dir ./reports/collections --pattern "tg_*.json"
        """
    )
    
    parser.add_argument('input_file', nargs='?', 
                       help='Input JSON file to convert')
    parser.add_argument('output_file', nargs='?',
                       help='Output JSON file (optional, auto-generated if not provided)')
    parser.add_argument('--batch', action='store_true',
                       help='Process multiple files in batch mode')
    parser.add_argument('--input-dir', default='./reports/collections',
                       help='Input directory for batch processing (default: ./reports/collections)')
    parser.add_argument('--pattern', default='*.json',
                       help='File pattern for batch processing (default: *.json)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Batch mode
    if args.batch:
        success = batch_convert(args.input_dir, args.pattern, verbose=not args.quiet)
        if not success:
            sys.exit(1)
        return
    
    # Single file mode
    if not args.input_file:
        # Default input file
        input_file = "./reports/collections/tg_SherwinVakiliLibrary_1_150244.json"
        
        # Check if default file exists
        if not os.path.exists(input_file):
            print(f"❌ No input file specified and default file not found: {input_file}")
            print("Use --help for usage information.")
            return
    else:
        input_file = args.input_file
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Convert the file
    success = convert_legacy_json_to_dataframe_format(input_file, args.output_file, verbose=not args.quiet)
    
    if success:
        print("\n🎉 Conversion completed successfully!")
        print("The converted file is now ready for analysis with the enhanced analyze command.")
    else:
        print("\n❌ Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
