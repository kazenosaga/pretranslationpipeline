#!/usr/bin/env python3
import argparse, os, sys, subprocess

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
    ap = argparse.ArgumentParser(description="Run try.py over many CSV files.")
    ap.add_argument("paths", nargs="*", default=["./ttr/"], help="CSV files and/or folders containing CSVs")
    ap.add_argument("--try", dest="tryscript", default="try.py", help="Path to try.py (default: try.py)")
    args, passthrough = ap.parse_known_args()

    found = list(csvs(args.paths))
    if not found:
        print("[info] no CSVs found", file=sys.stderr); return
    
    print(found)

    print(f"[info] running {args.tryscript} on {len(found)} file(s)")
    for f in found:
        cmd = [sys.executable, args.tryscript, "-f", f] + passthrough
        print(">", " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[fail] {f} (rc={r.returncode})", file=sys.stderr)

if __name__ == "__main__":
    main()
