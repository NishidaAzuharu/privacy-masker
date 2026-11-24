"""市区町村・県・役所・駅の連動と旧互換の境界。"""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from privacy_masker.annotators import PublicKnowledge
from privacy_masker.data import municipalities as gm
from privacy_masker.data.ja_pools import PREFECTURES, REGION_BLOCKS
from privacy_masker.engine import PrivacyEngine
from privacy_masker.exceptions import ConfigurationError
from privacy_masker.operators import Surrogate
from privacy_masker.operators import surrogate_geo as geo
from privacy_masker.operators.surrogate_filter import SurrogateFilter
from privacy_masker.policy import Policy
from privacy_masker.stores import SQLiteStore
from privacy_masker.types import Entity
from .conftest import SurrogateCtx, FakeCrypto, FakeDetector


def apply(op, value, ctx, etype='LOCATION'):
    return op.apply(value, Entity(etype, 0, len(value), 1.0), ctx)


def test_packaged_data():
    data = gm.load_municipalities()
    assert data is gm.load_municipalities()
    assert len(data.rows) == 1892
    assert Counter(r.kind for r in data.rows) == {'市': 772, '町': 743, '村': 183, '政令市の区': 171, '特別区': 23}
    assert sum('郡' in r.name and r.kind == '町' for r in data.rows) == 741
    assert sum('郡' in r.name and r.kind == '村' for r in data.rows) == 176
    assert sum('郡' not in r.name and r.kind in ('町', '村') for r in data.rows) == 9
    assert sum(r.kind == '政令市' for r in data.records) == 20
    root = Path('privacy_masker/data')
    manifest = json.loads((root / 'ja_municipalities.manifest.json').read_text())
    assert manifest['entries'] == len(data.rows)
    assert manifest['sha256'] == hashlib.sha256((root / gm.NAME).read_bytes()).hexdigest()
    assert data.prefectures_by_name['府中市'] == ('広島県', '東京都')
    assert data.resolve('当別町') == data.resolve('石狩郡当別町')
    assert data.resolve('札幌市').kind == '政令市'
    assert data.resolve('中央区', '札幌市') == data.resolve('札幌市中央区')


@pytest.mark.parametrize('damage', ['missing', 'checksum', 'count', 'malformed', 'empty'])
def test_bad_package_fails_closed(tmp_path, monkeypatch, damage):
    payload = b'' if damage == 'empty' else '東京都\t新宿区\n'.encode()
    if damage == 'malformed':
        payload = '東京都\t新宿\n'.encode()
    manifest = {'entries': 2 if damage == 'count' else (0 if damage == 'empty' else 1),
                'sha256': '0' * 64 if damage == 'checksum' else hashlib.sha256(payload).hexdigest()}
    if damage != 'missing':
        (tmp_path / gm.NAME).write_bytes(payload)
    (tmp_path / 'ja_municipalities.manifest.json').write_text(json.dumps(manifest))
    gm.load_municipalities.cache_clear()
    monkeypatch.setattr(gm.resources, 'files', lambda _: tmp_path)
    try:
        with pytest.raises(ConfigurationError):
            gm.load_municipalities()
    finally:
        gm.load_municipalities.cache_clear()


@pytest.mark.parametrize('index', range(24))
def test_all_prefectures_and_municipality_kinds(index):
    op = Surrogate(kind='auto', key=f'geo-key-{index}'.encode())
    ctx = SurrogateCtx()
    data = gm.load_municipalities()
    for pref in PREFECTURES:
        target_pref = apply(op, pref, ctx)
        block = next(v for v in REGION_BLOCKS.values() if pref in v)
        assert target_pref in (REGION_BLOCKS['東北'] if pref == '北海道' else block)
        assert target_pref != pref
        assert target_pref == apply(op, pref, ctx, 'LOCATION_PUBLIC')
        assert target_pref == apply(op, pref, ctx, 'ADDRESS')
    for name in ('名取市', '仙台市', '仙台市青葉区', '新宿区', '石狩郡当別町', '南都留郡忍野村', '八丈島八丈町'):
        source = data.resolve(name)
        target_pref = apply(op, source.prefecture, ctx)
        out = apply(op, name, ctx)
        target = data.resolve(out, prefecture=target_pref)
        assert target is not None, (name, out, target_pref)
        assert target in data.candidates(source, target_pref, name)
        if '郡' in name:
            assert '郡' in out
        assert out == apply(op, name, ctx)
    pref = apply(op, '宮城県', ctx)
    city = apply(op, '仙台市', ctx)
    assert apply(op, '宮城県仙台市', ctx) == pref + city
    assert data.resolve(city, prefecture=pref) is not None
    full = apply(op, '石狩郡当別町', ctx)
    assert apply(op, '当別町', ctx) == full.split('郡')[-1]
    contextual = SurrogateCtx(text='仙台市青葉区の青葉区')
    target2 = geo.municipality_for(op, data.resolve('仙台市青葉区'), contextual)
    assert apply(op, '青葉区', contextual) == target2.short


VALUES = [('LOCATION_PUBLIC', '東京都'), ('LOCATION', '新宿区'),
          ('ADDRESS', '東京都新宿区西新宿2-8-1'), ('FACILITY', '新宿区役所'),
          ('FACILITY', '新宿駅'), ('LOCATION', '新宿駅前')]


def assert_linked(outputs):
    pref, city, address, office, station, front = outputs
    target = gm.load_municipalities().resolve(city, prefecture=pref)
    assert target is not None
    assert address.startswith(pref + city)
    assert office == target.office
    assert station == target.station_stem + '駅'
    assert front == station + '前'


@pytest.mark.parametrize('strategy', ['hmac', 'random', 'vault'])
@pytest.mark.parametrize('reverse', [False, True])
def test_shared_context_across_operators(strategy, reverse):
    ctx = SurrogateCtx(text='、'.join(v for _, v in VALUES))
    outputs = {}
    for t, v in reversed(VALUES) if reverse else VALUES:
        # 別 operator instance でも同じ context / store を共有する。
        outputs[v] = apply(Surrogate(kind='auto', strategy=strategy, key=b'geo-link'), v, ctx, t)
    assert_linked([outputs[v] for _, v in VALUES])
    for t, v in VALUES:
        assert apply(Surrogate(kind='auto', strategy=strategy, key=b'geo-link'), v, ctx, t) == outputs[v]
    if strategy != 'hmac':
        assert any(t == geo._INTERNAL_TYPE for t, _ in ctx.tokens)
        assert all(not token.startswith('\x00') for token in ctx.saved_originals)


@pytest.mark.parametrize('strategy', ['hmac', 'random', 'vault'])
@pytest.mark.parametrize('sqlite', [False, True])
def test_engine_integration_and_restore(tmp_path, strategy, sqlite):
    text = '、'.join(v for _, v in VALUES)
    entities, pos = [], 0
    for t, v in VALUES:
        entities.append(Entity(t, pos, pos + len(v), 1.0))
        pos += len(v) + 1
    store = SQLiteStore(str(tmp_path / 'geo.db')) if sqlite else None
    eng = PrivacyEngine(policy=Policy(default=Surrogate(kind='auto', strategy=strategy)),
                        detectors=[FakeDetector(entities)], refiner=None, crypto=FakeCrypto(),
                        secret_key=b'geo-engine-key', store=store)
    try:
        result = eng.transform(text, scope_id='geo')
        assert len(result.entities) == 6
        assert_linked(result.text.split('、'))
        assert eng.transform(text, scope_id='geo').text == result.text
        if strategy == 'vault':
            restored = eng.restore(result.text, scope_id='geo')
            assert restored.text == text and restored.restored == 6 and restored.unresolved == ()
            for t, token in eng.store.iter_tokens('geo'):
                if t == geo._INTERNAL_TYPE:
                    assert eng.store.get_value('geo', token) is None
                    assert token not in eng._surrogate_tokens('geo')
    finally:
        if store:
            store.close()


@pytest.mark.parametrize('name,text,pref,parent', [
    ('中央区', '札幌市', '北海道', '札幌市'), ('中央区', '福岡県', '福岡県', '福岡市'),
    ('府中市', '東京都', '東京都', ''), ('府中市', '広島県', '広島県', ''),
    ('池田町', '長野県', '長野県', ''),
])
def test_context_disambiguation(name, text, pref, parent):
    data = gm.load_municipalities()
    source = data.resolve(name, text)
    assert (source.prefecture, source.parent) == (pref, parent)
    op, ctx = Surrogate(kind='auto', key=b'geo-context'), SurrogateCtx(text=text)
    target = geo.municipality_for(op, source, ctx)
    assert apply(op, name, ctx) == target.render(source, name)


@pytest.mark.parametrize('name,text', [('中央区', ''), ('中央区', '東京都と札幌市'),
                                       ('府中市', ''), ('府中市', '東京都と広島県'), ('池田町', '')])
def test_ambiguous_is_old_path(name, text, monkeypatch):
    op, ctx = Surrogate(kind='auto', key=b'geo-context'), SurrogateCtx(text=text)
    assert gm.load_municipalities().resolve(name, text) is None
    assert geo.plan(op, 'location', name, ctx) is None
    out = apply(op, name, ctx)
    monkeypatch.setattr(geo, 'plan', lambda *args: None)
    assert apply(op, name, ctx) == out


@pytest.mark.parametrize('kind,value', [('facility', '新大久保駅'), ('facility', '羽澄市役所'),
    ('facility', '仙台大学'), ('facility', '仙台病院'), ('facility', '仙台郵便局'),
    ('facility', '仙台小学校'), ('facility', '中央駅'), ('facility', '旭駅'),
    ('location', '青凪市'), ('location', '羽澄駅前')])
def test_other_names_keep_old_generator(kind, value, monkeypatch):
    op, ctx = Surrogate(kind=kind, key=b'geo-context'), SurrogateCtx(text=value)
    assert geo.plan(op, kind, value, ctx) is None
    out = apply(op, value, ctx, kind.upper())
    monkeypatch.setattr(geo, 'plan', lambda *args: None)
    assert apply(op, value, ctx, kind.upper()) == out
    if value == '新大久保駅':
        assert out.endswith('駅') and '新大久保' not in out


@pytest.mark.parametrize('value,kind', [('仙台市役所', 'facility'), ('仙台駅', 'facility'),
    ('仙台駅北口', 'facility'), ('仙台駅前', 'facility'), ('仙台駅前', 'location'),
    ('当別町役場', 'facility'), ('忍野村役場', 'facility')])
def test_linked_public_names_only_exempt_public_check(value, kind):
    ctx = SurrogateCtx(text=value)
    op = Surrogate(kind=kind, key=b'geo-public')
    first = apply(op, value, ctx, kind.upper())
    assert not first.startswith('<')
    knowledge = PublicKnowledge.from_iterable([first])
    op._filter = SurrogateFilter(packaged=knowledge, knowledge=knowledge)
    assert apply(op, value, ctx, kind.upper()) == first
    op._filter = SurrogateFilter(denylist=[first])
    assert apply(op, value, ctx, kind.upper()) != first


def test_prefecture_existing_store_mapping_is_used():
    ctx = SurrogateCtx()
    ctx.save_token('LOCATION', '東京都', '千葉県', original_value='東京都')
    op = Surrogate(kind='auto', strategy='vault')
    assert apply(op, '東京都', ctx, 'LOCATION_PUBLIC') == '千葉県'
    target = gm.load_municipalities().resolve(apply(op, '新宿区', ctx), prefecture='千葉県')
    assert target is not None


@pytest.mark.parametrize('strategy', ['hmac', 'random', 'vault'])
def test_filter_exhaustion_tokens(strategy):
    op, ctx = Surrogate(kind='auto', strategy=strategy, key=b'geo-exhaust'), SurrogateCtx()
    op._filter.reject_reason = lambda *args, **kwargs: 'denylist'
    for t, value in VALUES:
        out = apply(op, value, ctx, t)
        assert out.startswith('<') and value not in out
        if strategy == 'vault':
            assert ctx.saved_originals[out] == value


def test_document_collision_avoids_city_and_building_core():
    op = Surrogate(kind='auto', key=b'type-word-key-19')
    ctx = SurrogateCtx(text='東京都新宿区西新宿2丁目8番1号 青葉ビル3階')
    city = apply(op, '新宿区', ctx)
    address = apply(op, ctx.text, ctx, 'ADDRESS')
    assert '青葉' not in city and '青葉' not in address
    assert city in address and not address.startswith('<')


def test_legacy_does_not_load_geography(monkeypatch):
    def fail():
        raise AssertionError('legacy must not load municipalities')
    monkeypatch.setattr(geo, 'load_municipalities', fail)
    op = Surrogate(kind='address', semantic=False, key=b'geo-legacy')
    out = apply(op, '東京都新宿区西新宿2-8-1', SurrogateCtx(), 'ADDRESS')
    assert out and '西新宿' not in out


@pytest.mark.parametrize('city', ['東村山市', '郡山市', '大和郡山市', '蒲郡市', '武蔵村山市'])
def test_internal_type_characters_in_city_names(city):
    data = gm.load_municipalities()
    source = data.resolve(city)
    assert source.short == city
    ctx = SurrogateCtx(text=city + '栄町1-2-3')
    op = Surrogate(kind='auto', key=b'geo-inner')
    plain = apply(op, city, ctx)
    address = apply(op, ctx.text, ctx, 'ADDRESS')
    assert plain in address and '栄町' not in address
    kept = apply(Surrogate(kind='address', key=b'geo-inner', keep='city'), ctx.text, ctx, 'ADDRESS')
    assert kept.startswith(city) and '栄町' not in kept
    assert any(ch.isdigit() for ch in kept)


@pytest.mark.parametrize('suffix', ['前', '北口', '3階', '５F', '地下2階', '3号館'])
def test_station_fixed_affix_is_linked(suffix):
    ctx = SurrogateCtx(text='宮城県仙台市の仙台駅' + suffix)
    op = Surrogate(kind='auto', key=b'geo-affix')
    target = geo.municipality_for(op, gm.load_municipalities().resolve('仙台市'), ctx)
    out = apply(op, '仙台駅' + suffix, ctx, 'FACILITY')
    assert out.startswith(target.station_stem + '駅')
    if suffix in ('前', '北口'):
        assert out.endswith(suffix)
    else:
        from privacy_masker.operators.semantic import facility_shape
        _word, _core, attrs = facility_shape(out)
        _word, _core, before = facility_shape('仙台駅' + suffix)
        assert attrs['affix'] == before['affix']
        assert attrs['excluded_number'] != before['excluded_number']
        assert attrs['affix_fullwidth'] == before['affix_fullwidth']


def test_vault_collision_of_two_station_spellings_uses_restorable_token():
    values = [('FACILITY', '仙台駅'), ('FACILITY', '仙台市駅')]
    text = '、'.join(v for _, v in values)
    entities = [Entity('FACILITY', 0, 3, 1.0), Entity('FACILITY', 4, 8, 1.0)]
    eng = PrivacyEngine(policy=Policy(default=Surrogate(kind='auto', strategy='vault')),
                        detectors=[FakeDetector(entities)], refiner=None, crypto=FakeCrypto(),
                        secret_key=b'geo-collision')
    result = eng.transform(text, scope_id='collision')
    assert result.text.count('<FACILITY_') == 1
    restored = eng.restore(result.text, scope_id='collision')
    assert restored.text == text and restored.restored == 2


def test_other_facility_names_still_check_public_knowledge():
    ctx = SurrogateCtx(text='新大久保駅')
    op = Surrogate(kind='facility', key=b'geo-public')
    first = apply(op, '新大久保駅', ctx, 'FACILITY')
    op._filter = SurrogateFilter(packaged=PublicKnowledge.from_iterable([first]))
    assert apply(op, '新大久保駅', ctx, 'FACILITY') != first


def test_countyless_form_stays_countyless():
    op = Surrogate(kind='auto', key=b'geo-countyless')
    ctx = SurrogateCtx(text='東京都八丈島八丈町')
    out = apply(op, '八丈島八丈町', ctx)
    assert '郡' not in out and out.endswith('町')


def test_type_fallback_preferences():
    data = gm.load_municipalities()
    # 兵庫に村は無く、栃木に政令市・政令市の区は無い。
    for source_name, pref, expected in [('南都留郡忍野村', '兵庫県', '町'),
                                      ('仙台市', '栃木県', '市'), ('新宿区', '栃木県', '市'),
                                      ('仙台市青葉区', '栃木県', '市')]:
        source = data.resolve(source_name)
        candidates = data.candidates(source, pref, source.name)
        assert candidates and all(r.kind == expected for r in candidates)


@pytest.mark.parametrize('value', ['東京都新宿区西新宿2-8-1', '東村山市栄町1-2-3'])
def test_keep_city_vault_saves_actual_output(value):
    eng = PrivacyEngine(policy=Policy(default=Surrogate(kind='address', strategy='vault', keep='city')),
                        detectors=[FakeDetector([Entity('ADDRESS', 0, len(value), 1.0)])],
                        refiner=None, crypto=FakeCrypto(), secret_key=b'geo-keep')
    result = eng.transform(value, scope_id='keep')
    assert result.text != value
    assert result.text.startswith('東京都新宿区' if value.startswith('東京都') else '東村山市')
    assert eng.transform(value, scope_id='keep').text == result.text
    restored = eng.restore(result.text, scope_id='keep')
    assert restored.text == value and restored.restored == 1 and restored.unresolved == ()


@pytest.mark.parametrize('index', range(24))
def test_prefecture_derangement_is_injective(index):
    op = Surrogate(kind='auto', key=f'geo-permutation-{index}'.encode())
    for block in REGION_BLOCKS.values():
        if len(block) == 1:
            continue
        outputs = {p: apply(op, p, SurrogateCtx()) for p in block}
        assert set(outputs.values()) == set(block)
        assert all(p != out for p, out in outputs.items())
        assert {p: apply(op, p, SurrogateCtx()) for p in reversed(block)} == outputs


@pytest.mark.parametrize('index', range(24))
def test_city_assignment_is_injective_and_order_independent(index):
    data = gm.load_municipalities()
    op = Surrogate(kind='auto', key=f'geo-city-order-{index}'.encode())
    # 同地方のどの県よりも通常市が少ない県。全市について候補数が足りる。
    sources = [r for r in data.by_prefecture['鳥取県'] if r.kind == '市']
    ctx = SurrogateCtx()
    pref = geo.prefecture_for(op, '鳥取県', ctx)
    outputs = [geo.municipality_for(op, r, ctx) for r in sources]
    assert all(r is not None and r.prefecture == pref and r.kind == '市' for r in outputs)
    assert len(set(outputs)) == len(sources)
    assert list(reversed([geo.municipality_for(op, r, SurrogateCtx())
                          for r in reversed(sources)])) == outputs


@pytest.mark.parametrize('index', range(24))
def test_designated_city_shortage_uses_distinct_cities(index):
    op = Surrogate(kind='auto', key=f'geo-designated-{index}'.encode())
    data, ctx = gm.load_municipalities(), SurrogateCtx()
    sources = [data.resolve(n) for n in ('横浜市', '川崎市', '相模原市')]
    outputs = [geo.municipality_for(op, r, ctx) for r in sources]
    assert all(r is not None and r.kind in ('市', '政令市') for r in outputs)
    assert len(set(outputs)) == 3
    pref = geo.prefecture_for(op, '神奈川県', ctx)
    if pref in ('埼玉県', '千葉県'):
        assert sum(r.kind == '政令市' for r in outputs) == 1


@pytest.mark.parametrize('value', ['博多駅', '博多駅西口', '羽澄駅北口', '横浜駅', '横浜駅西口',
                                    '仙台駅前', '仙台駅3階', '仙台駅５F'])
@pytest.mark.parametrize('strategy', ['hmac', 'random', 'vault'])
def test_location_station_shares_facility_output(value, strategy):
    op = Surrogate(kind='auto', strategy=strategy, key=b'geo-station-types')
    ctx = SurrogateCtx(text='横浜市 仙台市 ' + value)
    out = apply(op, value, ctx, 'LOCATION')
    assert '駅' in out and out != value
    assert apply(op, value, ctx, 'FACILITY') == out
    if strategy == 'vault':
        assert ctx.saved_originals[out] == value


@pytest.mark.parametrize('text', ['横浜市の横浜駅', '横浜市の横浜駅と青森県', ''])
def test_yokohama_station_resolves_city(text):
    data = gm.load_municipalities()
    assert data.resolve('横浜', text, stem=True) == data.resolve('横浜市')
    op, ctx = Surrogate(kind='auto', key=b'geo-yokohama'), SurrogateCtx(text=text)
    target = geo.municipality_for(op, data.resolve('横浜市'), ctx)
    assert apply(op, '横浜駅', ctx) == target.station_stem + '駅'


def test_stem_city_preference_follows_explicit_context():
    data = gm.load_municipalities()
    town = data.resolve('横浜町')
    assert data.resolve('横浜', '横浜町', stem=True) == town
    assert data.resolve('横浜', '青森県', stem=True) == town
    assert data.resolve('横浜') is None
    synthetic = gm.Municipalities([('東京都', '羽澄市'), ('千葉県', '羽澄市'),
                                   ('青森県', '羽澄町')])
    assert synthetic.resolve('羽澄', stem=True) is None


@pytest.mark.parametrize('strategy', ['random', 'vault'])
def test_saved_geo_mappings_avoid_collisions(strategy):
    op, ctx = Surrogate(kind='auto', strategy=strategy), SurrogateCtx()
    # 既存の通常LOCATION対応も予約済みとして扱う。
    ctx.save_token('LOCATION', '東京都', '埼玉県')
    for source in ('千葉県', '神奈川県'):
        out = geo.prefecture_for(op, source, ctx)
        assert out != '埼玉県'
    data = gm.load_municipalities()
    ctx = SurrogateCtx()
    ctx.save_token('LOCATION', '横浜市', 'さいたま市')
    geo._mapping(op, ctx, 'pref:神奈川県', lambda: '埼玉県')
    targets = [geo.municipality_for(op, data.resolve(n), ctx)
               for n in ('横浜市', '川崎市', '相模原市')]
    assert all(targets) and len(set(targets)) == 3
    assert targets[0].name == 'さいたま市'


@pytest.mark.parametrize('index', range(24))
@pytest.mark.parametrize('values,region', [
    (('東京都', '埼玉県', '千葉県', '神奈川県'), '関東'),
    (REGION_BLOCKS['関東'], '関東'),
    (('京都府', '大阪府', '兵庫県', '奈良県', '和歌山県'), '近畿'),
    (('北海道', '青森県', '宮城県'), '東北'),
])
def test_prefecture_document_assignment_avoids_collisions(index, values, region):
    op = Surrogate(kind='auto', key=f'geo-document-{index}'.encode())
    ctx = SurrogateCtx(text='と'.join(values) + '。')
    outputs = {p: apply(op, p, ctx) for p in values}
    targets = set(outputs.values())
    assert targets <= set(PREFECTURES)
    assert len(targets) == len(values)
    assert targets.isdisjoint(values)
    available = set(REGION_BLOCKS[region]) - set(values)
    assert len(targets & available) == min(len(available), len(values))
    for order in (values[::-1], values[1:] + values[:1]):
        # 新しいoperator / contextでも処理順と本文の出現順に依存しない。
        other = Surrogate(kind='auto', key=f'geo-document-{index}'.encode())
        for text in (ctx.text, 'と'.join(order)):
            fresh = SurrogateCtx(text=text)
            assert {p: apply(other, p, fresh) for p in order} == outputs


@pytest.mark.parametrize('index', range(24))
def test_single_prefecture_document_preserves_cycle(index):
    op = Surrogate(kind='auto', key=f'geo-document-{index}'.encode())
    for pref in PREFECTURES:
        assert apply(op, pref, SurrogateCtx(text=pref)) == apply(op, pref, SurrogateCtx())
    # 衝突のない複数県の文も従来の巡回先を維持する。
    values = ('東京都', '大阪府', '宮城県')
    ctx = SurrogateCtx(text='と'.join(values))
    assert [apply(op, p, ctx) for p in values] == [apply(op, p, SurrogateCtx()) for p in values]


def test_single_prefecture_keeps_existing_filter_fallback():
    op = Surrogate(kind='auto', key=b'geo-single-denied')
    target = apply(op, '東京都', SurrogateCtx())
    op._filter = SurrogateFilter(denylist=[target])
    without_text = apply(op, '東京都', SurrogateCtx())
    assert without_text.startswith('<LOCATION_')
    assert apply(op, '東京都', SurrogateCtx(text='東京都')) == without_text


@pytest.mark.parametrize('strategy', ['random', 'vault'])
@pytest.mark.parametrize('values', [REGION_BLOCKS['関東'], ('北海道', '青森県', '宮城県')])
def test_stored_strategy_document_prefectures_are_distinct(strategy, values):
    op = Surrogate(kind='auto', strategy=strategy)
    ctx = SurrogateCtx(text='と'.join(values))
    outputs = {p: apply(op, p, ctx) for p in values}
    assert set(outputs.values()) <= set(PREFECTURES) - set(values)
    assert len(set(outputs.values())) == len(values)
    assert {p: apply(op, p, ctx) for p in reversed(values)} == outputs


@pytest.mark.parametrize('strategy', ['random', 'vault'])
def test_saved_prefecture_outside_region_is_preserved_and_reserved(strategy):
    op, ctx = Surrogate(kind='auto', strategy=strategy), SurrogateCtx(text='青森県と東京都と北海道')
    ctx.save_token('LOCATION', '東京都', '青森県')
    assert geo.prefecture_for(op, '東京都', ctx) == '青森県'
    assert geo.prefecture_for(op, '北海道', ctx) not in ('青森県', '東京都', '北海道')
    # 旧ストアに市対応しかなくても、地方をまたぐ保存値を優先する。
    ctx = SurrogateCtx(text='東京都')
    ctx.save_token('LOCATION', '新宿区', '仙台市')
    assert geo.prefecture_for(op, '東京都', ctx) == '宮城県'


@pytest.mark.parametrize('strategy', ['random', 'vault'])
@pytest.mark.parametrize('saved_kind', ['prefecture', 'internal', 'city'])
def test_saved_target_in_another_region_is_not_reused(strategy, saved_kind, monkeypatch):
    op, ctx = Surrogate(kind='auto', strategy=strategy), SurrogateCtx(text='北海道')
    if saved_kind == 'prefecture':
        ctx.save_token('LOCATION', '東京都', '宮城県')
    elif saved_kind == 'internal':
        geo._mapping(op, ctx, 'pref:東京都', lambda: '宮城県')
    else:
        ctx.save_token('LOCATION', '新宿区', '仙台市')
    monkeypatch.setattr(geo._RNG, 'sample', lambda choices, size:
                        sorted(choices, key=lambda c: c != '宮城県'))
    out = apply(op, '北海道', ctx)
    assert out in REGION_BLOCKS['東北'] and out != '宮城県'


def test_prefecture_document_matching_uses_normalized_text():
    values = REGION_BLOCKS['関東']
    op = Surrogate(kind='auto', key=b'geo-normalized-document')
    text = 'と'.join(values)
    compact = SurrogateCtx(text=text)
    spaced = SurrogateCtx(text='　'.join(text))
    outputs = {p: apply(op, p, spaced) for p in values}
    assert set(outputs.values()) <= set(PREFECTURES) - set(values)
    assert len(set(outputs.values())) == len(values)
    assert outputs == {p: apply(op, p, compact) for p in values}


@pytest.mark.parametrize('count', [23, 24])
def test_prefecture_document_capacity_across_regions(count):
    values = PREFECTURES[:count]
    op = Surrogate(kind='auto', key=b'geo-capacity')
    ctx = SurrogateCtx(text='と'.join(values))
    outputs = {p: apply(op, p, ctx) for p in values}
    targets = {out for out in outputs.values() if out in PREFECTURES}
    assert len(targets) == min(count, 47 - count)
    assert targets.isdisjoint(values)
    assert len(set(outputs.values())) == count
    assert {p: apply(op, p, SurrogateCtx(text=ctx.text)) for p in reversed(values)} == outputs


@pytest.mark.parametrize('reverse', [False, True])
def test_crowded_prefectures_share_city_address_office_and_station(reverse):
    values = [('LOCATION', p) for p in REGION_BLOCKS['関東'] if p != '東京都'] + VALUES
    ctx = SurrogateCtx(text='、'.join(v for _, v in values))
    outputs = {}
    for kind, value in reversed(values) if reverse else values:
        outputs[value] = apply(Surrogate(kind='auto', key=b'geo-crowded-link'), value, ctx, kind)
    assert outputs['東京都'] not in REGION_BLOCKS['関東']
    assert_linked([outputs[v] for _, v in VALUES])


def test_all_prefectures_in_document_fall_back_to_tokens():
    op, ctx = Surrogate(kind='auto', key=b'geo-full'), SurrogateCtx(text='と'.join(PREFECTURES))
    assert all(apply(op, p, ctx).startswith('<LOCATION_') for p in PREFECTURES)


def test_municipality_safety_skip_does_not_merge_assignments():
    data = gm.load_municipalities()
    sources = [r for r in data.by_prefecture['鳥取県'] if r.kind == '市']
    op = Surrogate(kind='auto', key=b'geo-safe-skip')
    before = [geo.municipality_for(op, r, SurrogateCtx()) for r in sources]
    denied = before[0]
    op._filter = SurrogateFilter(denylist=[denied.name])
    ctx = SurrogateCtx(text=before[1].name)
    after = [geo.municipality_for(op, r, ctx) for r in sources]
    assert all(after) and len(set(after)) == len(sources)
    assert denied not in after and before[1] not in after


@pytest.mark.parametrize('index', range(24))
def test_city_and_designated_city_share_capacity(index):
    data = gm.load_municipalities()
    op = Surrogate(kind='auto', key=f'geo-city-capacity-{index}'.encode())
    # 岩手の14市なら宮城の通常市13+政令市1を別々に割り当てられる。
    table = geo._city_assignments(op, '宮城県', '岩手県', SurrogateCtx())
    targets = [out for src, out in table.items() if src.kind in ('市', '政令市')]
    assert len(targets) == 14 and all(targets)
    assert len(set(targets)) == 14


@pytest.mark.parametrize('station_type', ['LOCATION', 'FACILITY'])
def test_reported_city_station_relationships(station_type):
    op = Surrogate(kind='auto', key=b'replace-with-your-own-secret-key')
    for text, names, station in [
        ('横浜市の横浜駅西口 川崎市', ('横浜市', '川崎市'), '横浜駅西口'),
        ('宮城県仙台市 仙台駅 名取市', ('仙台市', '名取市'), '仙台駅'),
    ]:
        ctx = SurrogateCtx(scope_id=None, text=text)
        targets = [geo.municipality_for(op, gm.load_municipalities().resolve(n), ctx) for n in names]
        assert all(targets) and targets[0] != targets[1]
        assert apply(op, station, ctx, station_type).startswith(targets[0].station_stem + '駅')


@pytest.mark.parametrize('first_type', ['LOCATION', 'FACILITY'])
@pytest.mark.parametrize('sqlite', [False, True])
def test_station_cross_type_vault_roundtrip(tmp_path, first_type, sqlite):
    text = '横浜市の横浜駅西口と横浜駅西口'
    start = text.index('横浜駅西口')
    second = text.rindex('横浜駅西口')
    entities = [Entity('LOCATION', 0, 3, 1.0), Entity(first_type, start, start + 5, 1.0),
                Entity('FACILITY' if first_type == 'LOCATION' else 'LOCATION', second, second + 5, 1.0)]
    store = SQLiteStore(str(tmp_path / 'station.db')) if sqlite else None
    engine = PrivacyEngine(detectors=[FakeDetector(entities)], refiner=None,
                           policy=Policy(default=Surrogate(kind='auto', strategy='vault')),
                           store=store, crypto=FakeCrypto(), secret_key=b'geo-station-roundtrip')
    try:
        result = engine.transform(text, scope_id='station')
        assert result.entities[1].replacement == result.entities[2].replacement
        assert result.entities[1].replacement.endswith('駅西口')
        restored = engine.restore(result.text, scope_id='station')
        assert restored.text == text and restored.restored == 3 and not restored.unresolved
    finally:
        if store:
            store.close()


def test_station_unknown_trailing_phrase_stays_outside_shared_station_path():
    op, ctx = Surrogate(kind='auto', key=b'geo-unknown-affix'), SurrogateCtx()
    from privacy_masker.operators.semantic import facility_shape
    assert facility_shape('羽澄駅待合室')[0] is None
    # LOCATION型の任意文字列までFACILITYへ広げない。
    out = apply(op, '羽澄駅待合室', ctx)
    assert out and not out.endswith('駅待合室')


@pytest.mark.parametrize('strategy', ['random', 'vault'])
@pytest.mark.parametrize('qualified', [False, True])
def test_saved_city_alone_establishes_shared_prefecture(strategy, qualified):
    op, ctx = Surrogate(kind='auto', strategy=strategy), SurrogateCtx()
    original = '神奈川県横浜市' if qualified else '横浜市'
    replacement = '埼玉県さいたま市' if qualified else 'さいたま市'
    ctx.save_token('LOCATION', original, replacement)
    # まだ市を処理していなくても、県単独が保存済み市の先県を使う。
    assert apply(op, '神奈川県', ctx) == '埼玉県'
    assert apply(op, '横浜市役所', ctx, 'FACILITY') == 'さいたま市役所'
    assert apply(op, '横浜駅', ctx, 'LOCATION') == 'さいたま駅'
    assert geo.municipality_for(op, gm.load_municipalities().resolve('横浜市'), ctx).name == 'さいたま市'


@pytest.mark.parametrize('strategy', ['random', 'vault'])
def test_saved_city_reserves_target_prefecture_before_city_is_processed(strategy, monkeypatch):
    op, ctx = Surrogate(kind='auto', strategy=strategy), SurrogateCtx()
    ctx.save_token('LOCATION', '横浜市', 'さいたま市')
    monkeypatch.setattr(geo._RNG, 'sample', lambda choices, size:
                        sorted(choices, key=lambda c: c != '埼玉県'))
    assert geo.prefecture_for(op, '千葉県', ctx) != '埼玉県'


@pytest.mark.parametrize('strategy', ['random', 'vault'])
def test_saved_qualified_city_precedes_ambiguous_short_name(strategy):
    op, ctx = Surrogate(kind='auto', strategy=strategy), SurrogateCtx()
    ctx.save_token('LOCATION', '大阪府', '兵庫県')
    ctx.save_token('LOCATION', '北区', '神戸市東灘区')
    ctx.save_token('LOCATION', '大阪府大阪市北区', '兵庫県神戸市中央区')
    source = gm.load_municipalities().resolve('大阪市北区')
    assert geo.municipality_for(op, source, ctx).name == '神戸市中央区'
