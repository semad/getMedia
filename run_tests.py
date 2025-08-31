#!/usr/bin/env python3
"""
Test runner script for Telegram Media Library.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_pattern=None, verbose=False, coverage=False):
    """Run the test suite."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    tests_dir = script_dir / "tests"
    
    if not tests_dir.exists():
        print("❌ Tests directory not found!")
        print(f"   Expected: {tests_dir}")
        return False
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if coverage:
        cmd.extend(["--cov=modules", "--cov-report=html", "--cov-report=term"])
    
    if test_pattern:
        cmd.append(f"tests/{test_pattern}")
    else:
        cmd.append("tests/")
    
    # Add pytest options
    cmd.extend([
        "--tb=short",
        "--strict-markers",
        "--disable-warnings"
    ])
    
    print(f"🚀 Running tests with command: {' '.join(cmd)}")
    print(f"📁 Tests directory: {tests_dir}")
    print("=" * 60)
    
    try:
        # Run pytest
        result = subprocess.run(cmd, cwd=script_dir, check=False)
        
        print("=" * 60)
        if result.returncode == 0:
            print("✅ All tests passed!")
        else:
            print(f"❌ Some tests failed (exit code: {result.returncode})")
        
        return result.returncode == 0
        
    except FileNotFoundError:
        print("❌ pytest not found!")
        print("   Please install pytest: pip install pytest")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Telegram Media Library tests")
    parser.add_argument(
        "--pattern", "-p",
        help="Test pattern to run (e.g., test_models.py, test_telegram_collector.py)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run with coverage reporting"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available test files"
    )
    
    args = parser.parse_args()
    
    if args.list:
        # List available test files
        tests_dir = Path(__file__).parent / "tests"
        if tests_dir.exists():
            test_files = list(tests_dir.glob("test_*.py"))
            print("📋 Available test files:")
            for test_file in sorted(test_files):
                print(f"   {test_file.name}")
        else:
            print("❌ Tests directory not found!")
        return
    
    # Run tests
    success = run_tests(args.pattern, args.verbose, args.coverage)
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
