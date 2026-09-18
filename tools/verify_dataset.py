#!/usr/bin/env python3
import argparse, csv, hashlib
from pathlib import Path

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='Dataset root containing train/ val/ test/')
    p.add_argument('--manifest', default='datasets/manifests/dataset_manifest.csv')
    p.add_argument('--check_sha256', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='0 checks all rows')
    a=p.parse_args()
    root=Path(a.root)
    missing=[]; bad_hash=[]; n=0
    split_counts={}
    with open(a.manifest, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            n += 1
            if a.limit and n>a.limit: break
            split=row['split'].strip()
            split_counts[split]=split_counts.get(split,0)+1
            path=root/split/row['relative_path']
            if not path.is_file():
                missing.append(str(path)); continue
            if a.check_sha256 and row.get('sha256'):
                got=sha256(path)
                if got.lower()!=row['sha256'].strip().lower(): bad_hash.append(str(path))
    print('checked_rows =', n)
    print('split_counts =', split_counts)
    print('missing =', len(missing))
    print('bad_sha256 =', len(bad_hash))
    for x in missing[:10]: print('MISSING',x)
    for x in bad_hash[:10]: print('HASH_MISMATCH',x)
    raise SystemExit(1 if missing or bad_hash else 0)
if __name__=='__main__': main()
