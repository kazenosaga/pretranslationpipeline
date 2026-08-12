import csv, sys, argparse, os, subprocess
import xml.etree.ElementTree as ET

# Идем по папкам в пути, ищем все csv
def csvs(paths):
    seen = set()
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith(".csv"):
            ap = os.path.abspath(p)
            if ap not in seen: seen.add(ap); yield ap
        elif os.path.isdir(p):
            for name in os.listdir(p):
                fp = os.path.join(p, name)
                if os.path.isfile(fp) and fp.lower().endswith(".csv"):
                    ap = os.path.abspath(fp)
                    if ap not in seen: seen.add(ap); yield ap
        else:
            print(f"[skip] not found: {p}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="Run converter over many CSV files.")
    ap.add_argument("paths", nargs="*", default=["./ttr/"], help="CSV files and/or folders containing CSVs")
    args, passthrough = ap.parse_known_args()

    found = list(csvs(args.paths))
    if not found:
        print("!!! No CSVs found", file=sys.stderr); return
    
    print(found)

    print(f">>> Running on {len(found)} file(s)")

    out_path = "output.tmx"
    if os.path.exists(out_path):
        tree = ET.parse(out_path)
        tmx = tree.getroot()
        body = tmx.find('body') or ET.SubElement(tmx, 'body')
    else:
        tmx = ET.Element('tmx', version='1.4')
        ET.SubElement(tmx, 'header', srclang='ru')
        body = ET.SubElement(tmx, 'body')
        tree = ET.ElementTree(tmx)

    for f in found:
        with open(f, 'r', encoding='utf-8') as csvfile:
            csv_reader = csv.reader(csvfile)
            SEG_COUNTER = 0
            # По каждой строке CSV:
            for row in csv_reader:
                if len(row) >= 2:
                    SEG_COUNTER+=1
                    tu = ET.SubElement(body, 'tu')
                    # RU
                    tuv_ru = ET.SubElement(tu, 'tuv', {'xml:lang': 'ru'})
                    ET.SubElement(tuv_ru, 'seg').text = row[0]
                    # EN
                    tuv_en = ET.SubElement(tu, 'tuv', {'xml:lang': 'en'})
                    ET.SubElement(tuv_en, 'seg').text = row[1]


            # Записать в файл
            tree = ET.ElementTree(tmx)
            tree.write('output.tmx', encoding='utf-8', xml_declaration=True)

            print(">>> " + str(SEG_COUNTER) + " rows written to " + out_path)

if __name__ == "__main__":
    main()