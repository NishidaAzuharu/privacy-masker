"""日本郵便 CSV と抽出済み CSV の再現可能な変換。"""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.build_municipalities import build, SOURCE_URL

META = dict(acquired_date='2026-09-22', source_sha256='a' * 64)


def test_official_defaults(tmp_path):
    src, out = tmp_path / 'postal.csv', tmp_path / 'result.tsv'
    with src.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        for pref, city in [('北海道', '札幌市中央区'), ('東京都', '新宿区'), ('北海道', '札幌市中央区')]:
            w.writerow(['0'] * 6 + [pref, city, '町名,続き'] + ['0'] * 6)
    manifest = build(src, out, **META)
    assert out.read_text() == '北海道\t札幌市中央区\n東京都\t新宿区\n'
    assert manifest['entries'] == 2
    assert manifest['sha256'] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert manifest['source_url'] == SOURCE_URL and manifest['source_sha256'] == 'a' * 64
    assert '著作権を主張していない' in manifest['copyright']
    assert str(tmp_path) not in json.dumps(manifest)
    assert json.loads(out.with_suffix('.manifest.json').read_text()) == manifest


def test_three_columns_cli_and_determinism(tmp_path):
    src, out = tmp_path / 'postal.csv', tmp_path / 'result.tsv'
    src.write_text('prefecture,city,town\n東京都,新宿区,西新宿\n東京都,新宿区,新宿\n', encoding='utf-8-sig')
    args = [sys.executable, 'tools/build_municipalities.py', '--src', str(src), '--out', str(out),
            '--pref-column', '1', '--city-column', '2', '--header',
            '--acquired-date', META['acquired_date'], '--source-sha256', META['source_sha256']]
    subprocess.run(args, check=True, capture_output=True)
    before = out.read_bytes(), out.with_suffix('.manifest.json').read_bytes()
    subprocess.run(args, check=True, capture_output=True)
    assert before == (out.read_bytes(), out.with_suffix('.manifest.json').read_bytes())
    assert out.read_text() == '東京都\t新宿区\n'


@pytest.mark.parametrize('payload,overrides', [
    ('', {}), ('東京都\n', {}), (',新宿区\n', {}), ('東京都,新宿\n', {}),
    ('東京都,新宿区\n', {'pref_column': 0}), ('東京都,新宿区\n', {'city_column': 1}),
    ('東京都,新宿区\n', {'acquired_date': 'invalid'}),
    ('東京都,新宿区\n', {'source_sha256': 'invalid'}),
    ('東京都,"新宿区\n"\n', {'city_column': 3}),
])
def test_invalid_input(tmp_path, payload, overrides):
    src = tmp_path / 'bad.csv'
    src.write_text(payload)
    kwargs = dict(pref_column=1, city_column=2, **META)
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        build(src, tmp_path / 'bad.tsv', **kwargs)
    assert not (tmp_path / 'bad.tsv').exists()
