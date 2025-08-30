#!/bin/bash
# Activation script for Python 3.11.13 environment
echo "Activating Python 3.11.13 environment..."
source .venv/bin/activate
echo "Environment activated! Python version:"
python --version
echo ""
echo "Available commands:"
echo "  python main.py --help          # Show main help"
echo "  python main.py collect --help  # Show collect command help"
echo "  python main.py analyze --help  # Show analyze command help"
echo "  python main.py import --help   # Show import command help"
echo ""
echo "To deactivate, run: deactivate"
