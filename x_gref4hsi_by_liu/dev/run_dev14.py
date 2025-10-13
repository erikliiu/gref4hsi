"""Quick test to verify MBES + UHI footprint overlay works"""

import os
import sys

# Set UTF-8 encoding for Windows terminal
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Now run the actual script
exec(open("dev14_cont.py").read())
