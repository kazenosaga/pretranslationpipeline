#!/usr/bin/env python3
import openpyxl
import csv
import glob

excel_files = glob.glob('*.xlsx')

for file in excel_files:
    wb = openpyxl.load_workbook(file)
    sheet = wb.active # Gets the first sheet
    
    csv_file = file.replace('.xlsx', '.csv')
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row in sheet.iter_rows(values_only=True):
            writer.writerow(row)
            
    print(f"Converted {file} to {csv_file}")