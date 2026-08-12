#!/usr/bin/env python3
import glob
import csv
import os

# Get all CSV files in current directory
csv_files = glob.glob("*.csv")

# Write all rows to merged.csv
with open('merged.csv', 'w', newline='', encoding='utf-8') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['ru', 'en'])
    
    for filename in csv_files:
        with open(filename, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            for row in reader:
                writer.writerow(row)

print(f"Merged {len(csv_files)} files into merged.csv")