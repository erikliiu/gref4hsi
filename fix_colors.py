#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fix the color entries in georef.py"""

# Read the file
with open(
    r"e:\mjosa_complete\gref4hsi\mjosa_code\utils\uhi\georef.py", "r", encoding="utf-8"
) as f:
    content = f.read()

# Find and replace the malformed entry
old_text = """            # 🔥 Merged class for validation
            "new_sediment": "#8B4513",  # brown (same as training_sediment)
            # ?? Display names for plot_classification_map - same colors as training_*`n            "bombs": "#1E90FF",  # dodger blue (same as training_bombs)`n            "dark": "#000000",  # black (same as training_dark)`n            "sediment": "#8B4513",  # brown (same as training_sediment)`n        }"""

new_text = """            # 🔥 Merged class for validation
            "new_sediment": "#8B4513",  # brown (same as training_sediment)
            # 🔥 Display names for plot_classification_map - same colors as training_*
            "bombs": "#1E90FF",  # dodger blue (same as training_bombs)
            "dark": "#000000",  # black (same as training_dark)
            "sediment": "#8B4513",  # brown (same as training_sediment)
        }"""

if old_text in content:
    content = content.replace(old_text, new_text)
    print("✓ Found and replaced the malformed text")
else:
    print("✗ Could not find the exact text to replace")
    print("\nSearching for partial matches...")
    if "`n" in content:
        print("✓ Found backtick-n sequences in file")
        # Try a more flexible replacement
        import re

        pattern = r"# \?\? Display names.*?`n.*?`n.*?`n.*?`n\s+}"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            print(f"✓ Found pattern at position {match.start()}")
            content = (
                content[: match.start()]
                + new_text.split("\n", 2)[2]
                + content[match.end() :]
            )
            print("✓ Applied flexible replacement")

# Write the file back
with open(
    r"e:\mjosa_complete\gref4hsi\mjosa_code\utils\uhi\georef.py", "w", encoding="utf-8"
) as f:
    f.write(content)

print("\n✓ File written successfully!")

# Verify the change
with open(
    r"e:\mjosa_complete\gref4hsi\mjosa_code\utils\uhi\georef.py", "r", encoding="utf-8"
) as f:
    lines = f.readlines()
    for i, line in enumerate(lines[774:785], start=775):
        print(f"{i}: {line.rstrip()}")
