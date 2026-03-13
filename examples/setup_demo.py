"""
Setup demo data files for the self-healing agent demonstration.
Run this ONCE before running main.py.
"""

import os

DEMO_DIR = "/tmp/self_healing_demo"
os.makedirs(DEMO_DIR, exist_ok=True)

# ── Create a CSV with intentionally different name from what agent expects ──
# The agent will ask for 'sales.csv' but only 'revenue_2024.csv' exists.
# This triggers a read_file failure → reflection → list_directory → correct read.

csv_content = """\
month,product,revenue,units_sold
January,Widget A,12500.00,250
February,Widget A,13200.00,264
March,Widget A,11800.00,236
April,Widget B,8900.00,178
May,Widget B,9400.00,188
June,Widget B,10100.00,202
July,Widget A,14200.00,284
August,Widget A,15000.00,300
September,Widget B,11200.00,224
October,Widget A,13800.00,276
November,Widget B,12400.00,248
December,Widget A,16500.00,330
"""

with open(os.path.join(DEMO_DIR, "revenue_2024.csv"), "w") as f:
    f.write(csv_content)

print(f"Demo data created in {DEMO_DIR}/")
print(f"  • revenue_2024.csv  (NOT 'sales.csv' — this triggers the healing!)")
