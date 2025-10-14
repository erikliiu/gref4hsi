"""
Quick test script to run the multiple TIF overlay feature.

This will create a test HTML map with:
1. Main MBES overlay (visible by default)
2. Multiple TIF files from the specified folder (hidden by default, toggleable)
3. HSI footprints (if available)

Usage:
    python test_run_multiple_tifs.py
"""

from create_html import test_multiple_tifs

if __name__ == "__main__":
    test_multiple_tifs()
