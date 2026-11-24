"""Closed type words and fixed tails; all organization fixtures are synthetic."""
from dataclasses import fields, replace
from itertools import product
import json
import random
import re
from pathlib import Path

import pytest

from privacy_masker.annotators import PublicKnowledge
from privacy_masker.data import ja_pools as jp
from privacy_masker.engine import PrivacyEngine
from privacy_masker.normalize import default_normalize
from privacy_masker.operators import Surrogate
from privacy_masker.operators import semantic as sm
from privacy_masker.operators import surrogate_generators as sg
from privacy_masker.policy import Policy
from privacy_masker.types import Entity
from .conftest import FakeCrypto, FakeDetector, SurrogateCtx


ADDRESS = '東京都新宿区西新宿2丁目8番1号'
EXAMPLES = [
    ('org', '羽澄銀行', '銀行', '羽澄'),
    ('org', '羽澄證券', '證券', '羽澄'),
    ('facility', '桜台小学校', '小学校', '桜台'),
    ('facility', 'ひまわり幼稚園', '幼稚園', 'ひまわり'),
    ('facility', '新宿駅', '駅', '新宿'),
    ('facility', '羽田空港', '空港', '羽田'),
    ('facility', '青葉総合病院', '総合病院', '青葉'),
    ('facility', '中央図書館', '図書館', '中央'),
    ('facility', '渋谷駅前', '駅前', '渋谷'),
    ('facility', '大阪駅北口', '駅北口', '大阪'),
    ('location', '仙台駅前', '駅前', '仙台'),
]


def features(kind, value):
    n = default_normalize(value)
    return sm.extract_features(kind, value, n, sm.RuleCategorizer.categorize(kind, n))


def apply(kind, value, index=0, **kw):
    key = f'type-word-key-{index}'.encode()
    return Surrogate(kind=kind, key=key, **kw).apply(
        value, Entity(kind.upper(), 0, len(value), 1.0), SurrogateCtx(key, text=value),
    )


@pytest.mark.parametrize('kind,value,suffix,core', EXAMPLES)
def test_examples_across_twenty_keys(kind, value, suffix, core):
    for index in range(20):
        out = apply(kind, value, index)
        assert out.endswith(suffix) and core not in out and out != value
        assert out == apply(kind, value, index)
        if kind == 'facility' and suffix in jp.FACILITY_TYPE_WORDS['education']:
            assert not out.startswith(jp.FACILITY_TYPE_PREFIXES)
        if kind == 'location':
            assert '町' not in out and '丁目' not in out


@pytest.mark.parametrize('word', jp.FINANCE_TYPE_WORDS)
@pytest.mark.parametrize('form,pos', [(None, None), ('株式会社', 'pre'), ('株式会社', 'post'), ('（株）', 'pre')])
def test_finance_types_legal_forms_and_directions(word, form, pos):
    body = '羽澄' + word
    value = (form + body if pos == 'pre' else body + form) if form else body
    assert sm.RuleCategorizer.categorize('org', value) == 'finance'
    assert features('org', value).type_word == word
    for index in range(20):
        out = apply('org', value, index)
        got_form, got_pos, got_body = sm.split_legal_form(out)
        assert (got_form, got_pos) == (form, pos)
        assert got_body.startswith(jp.ORG_DIRECTIONS) and got_body.endswith(word)
        assert '羽澄' not in out and out != value


@pytest.mark.parametrize('word', sorted({w for ws in jp.FACILITY_TYPE_WORDS.values() for w in ws}))
def test_every_facility_type(word):
    value = '羽澄' + word
    assert features('facility', value).type_word == word
    for index in range(20):
        out = apply('facility', value, index)
        assert out.endswith(word) and '羽澄' not in out and out != value


@pytest.mark.parametrize('prefix', ('',) + jp.FACILITY_TYPE_PREFIXES)
@pytest.mark.parametrize('word', ['小学校', '高校', '高等学校', '幼稚園'])
def test_school_ownership_only_when_present(prefix, word):
    for index in range(20):
        out = apply('facility', prefix + '桜台' + word, index)
        assert out.endswith(word) and '桜台' not in out
        actual = next((p for p in jp.FACILITY_TYPE_PREFIXES if out.startswith(p)), '')
        assert actual == prefix


@pytest.mark.parametrize('suffix', jp.FACILITY_AFFIXES + tuple(f.format(5) for f in jp.FACILITY_NUMBER_FORMS))
def test_facility_affixes(suffix):
    value = '青葉ビル' + suffix
    for index in range(20):
        out = apply('facility', value, index)
        assert '青葉' not in out and out != value and 'ビル' in out
        tail = out.split('ビル')[1]
        if '5' in suffix:
            assert re.sub(r'\d+', 'N', tail) == suffix.replace('5', 'N')
            assert int(re.search(r'\d+', tail)[0]) != 5
        else:
            assert tail == suffix


@pytest.mark.parametrize('word', jp.DISTRICT_TYPE_WORDS)
def test_district_suffixes(word):
    for index in range(20):
        out = apply('location', '羽澄' + word, index)
        assert out.endswith(word) and '羽澄' not in out
        if word == '駅前':
            assert '町' not in out and '丁目' not in out


@pytest.mark.parametrize('word', jp.BUILDING_TYPE_WORDS)
@pytest.mark.parametrize('pos,suffix,gap', [('post', '3階', ' '), ('post', '5F', ''), ('pre', ' 802号室', ' '), ('post', '', '')])
def test_address_building_shapes_and_region(word, pos, suffix, gap):
    building = ('青葉' + word if pos == 'post' else word + '青葉') + suffix
    value = ADDRESS + gap + building
    f = features('address', value)
    assert f.type_word == word and f.building_position == pos
    for index in range(20):
        out = apply('address', value, index)
        assert not out.startswith('<') and '青葉' not in out and '西新宿' not in out
        assert sm.region_block_of(out) == '関東'
        got_word, _core, attrs = sm.address_building_shape(out)
        assert got_word == word and attrs['building_position'] == pos
        assert attrs['building_separator'] == gap and attrs['number_separator'] == (' ' if suffix.startswith(' ') else '')
        assert attrs.get('affix', '') == f.affix
        if suffix:
            assert int(re.search(r'\d+(?=階$|F$|号室$)', out)[0]) != int(re.search(r'\d+', suffix)[0])
        else:
            assert out.endswith(word)


@pytest.mark.parametrize('kind,value', [
    ('org', '羽澄バンク'), ('org', '羽澄銀行本店'),
    ('facility', '羽澄停留場'), ('facility', '羽澄駅自由通路'), ('facility', '羽澄ビル5階受付'),
    ('location', '羽澄1区'), ('address', ADDRESS + ' 青葉ビル3階受付'),
    ('address', ADDRESS + ' 青葉館802号室'),
])
def test_unknown_types_and_free_tails_use_previous_generator(kind, value):
    f = features(kind, value)
    assert f.type_word is None and not f.affix
    # No preserved-shape feature enters the fallback; random consumption stays identical.
    plain = sm.Features(kind, f.category, f.script, f.length_class, granularity=f.granularity,
                        region_block=f.region_block, has_building=f.has_building, chome_style=f.chome_style)
    for seed in range(20):
        assert sg.generate(f, random.Random(seed)) == sg.generate(plain, random.Random(seed))


@pytest.mark.parametrize('kind,value', [('facility', '青葉ビル５階'), ('facility', '青葉ビル地下５階'),
                                         ('address', ADDRESS + '　メゾン青葉　８０２号室')])
def test_fullwidth_number_format(kind, value):
    for index in range(20):
        out = apply(kind, value, index)
        assert '青葉' not in out
        digits = re.search(r'[０-９]+(?=階$|号室$)', out)[0]
        assert int(digits) != (802 if kind == 'address' else 5)
        if kind == 'address':
            assert '　メゾン' in out and re.search(r'　[０-９]+号室$', out)


def test_closed_features_reject_free_strings_and_large_numbers():
    f = features('facility', '青葉ビル5階')
    for kwargs in ({'type_word': '自由文字列'}, {'affix': '受付'}, {'org_prefix': '自由文字列'},
                   {'excluded_number': 802}, {'building_separator': '自由文字列'}, {'building_position': '自由文字列'}):
        with pytest.raises(ValueError, match='closed vocabulary'):
            replace(f, **kwargs)
    assert features('address', ADDRESS + ' メゾン青葉 802号室').excluded_number is None
    for field in fields(f):
        assert '青葉' not in str(getattr(f, field.name))


@pytest.mark.parametrize('kind,value', [('org', '羽澄銀行'), ('facility', '青葉ビル5階'),
                                         ('location', '羽澄駅前'), ('address', ADDRESS + ' 青葉ビル3階')])
def test_vault_consistency_restore_and_exhaustion(kind, value):
    text = value + '、' + value
    ent = [Entity(kind.upper(), start, start + len(value), 1.0) for start in (0, len(value) + 1)]
    op = Surrogate(kind=kind, strategy='vault')
    engine = PrivacyEngine(policy=Policy(default=op), crypto=FakeCrypto(), secret_key=b'k' * 16,
                           detectors=[FakeDetector(ent)], refiner=None)
    out = engine.transform(text, scope_id='type-words').text
    assert len(set(out.split('、'))) == 1
    assert engine.restore(out, scope_id='type-words').text == text
    op._filter.reject_reason = lambda *a, **kw: 'denylist'
    token = op.apply(value, ent[0], SurrogateCtx())
    assert token.startswith('<' + kind.upper() + '_')


def test_address_keep_city():
    value = ADDRESS + ' 青葉ビル3階'
    for index in range(20):
        out = apply('address', value, index, keep='city')
        assert out.startswith('東京都新宿区') and '西新宿' not in out and '青葉' not in out
        assert re.search(r' ビル', out) is None
        assert re.search(r'ビル(?!3階)[0-9]+階$', out)


@pytest.fixture(scope='module')
def universe():
    return set(sg.enumerate_candidates())


def test_complete_preserved_candidate_space(universe):
    for d, core, word in product(jp.ORG_DIRECTIONS, jp.ORG_CORES, jp.FINANCE_TYPE_WORDS):
        for name in sg._yield_org_forms(d + core + word):
            assert name in universe
    tails = ('',) + jp.FACILITY_AFFIXES + tuple(f.format(i) for f in jp.FACILITY_NUMBER_FORMS for i in jp.TYPE_NUMBERS)
    for category, words in jp.FACILITY_TYPE_WORDS.items():
        prefixes = ('',) + jp.FACILITY_TYPE_PREFIXES if category == 'education' else ('',)
        for pre, core, word, tail in product(prefixes, jp.ORG_CORES + jp.KATA_CORES, words, tails):
            name = pre + core + word + tail
            assert name in universe and name.translate(sg._TO_FULLWIDTH) in universe
    for core, word, pos, sep, tail in product(jp.KATA_CORES, jp.BUILDING_TYPE_WORDS, ('pre', 'post'), ('', ' ', '　'),
                                            ('',) + tuple(f.format(i) for f in jp.BUILDING_NUMBER_FORMS for i in jp.TYPE_NUMBERS)):
        name = (word + core if pos == 'pre' else core + word) + sep + tail
        assert name in universe and name.translate(sg._TO_FULLWIDTH) in universe
    assert {c + w for c, w in product(jp.ORG_CORES + jp.TOWNS, jp.DISTRICT_TYPE_WORDS)} <= universe


def test_bundled_denylist_covers_all_knowledge_collisions(universe):
    data = Path('privacy_masker/data')
    knowledge = PublicKnowledge.load(data / 'public_knowledge_ja.pmk')
    deny = PublicKnowledge.load(data / 'surrogate_denylist.pmk')
    manifest = json.loads((data / 'surrogate_denylist.manifest.json').read_text())
    hits = {c for c in universe if c in knowledge}
    assert all(c in deny for c in hits)
    assert len(hits) == manifest['hits'] and len(universe) == manifest['candidates']
    assert manifest['knowledge_sha256'] == '271d41788998c4bfdca6e385e0e0ab1ae76ea185b4ccec7032e7ba335df2d4a8'


def test_building_public_collision_retries_and_exhausts():
    names = [c + 'ビル' for c in jp.KATA_CORES]
    op = Surrogate(kind='address', key=b'k', knowledge=PublicKnowledge.from_iterable(names))
    value = ADDRESS + ' 青葉ビル3階'
    out = op.apply(value, Entity('ADDRESS', 0, len(value), 1.0), SurrogateCtx())
    assert out.startswith('<ADDRESS_')
    first = apply('address', value)
    word, core, _ = sm.address_building_shape(first)
    out = apply('address', value, knowledge=PublicKnowledge.from_iterable([core + word]))
    assert out != first and not out.startswith('<')


@pytest.mark.parametrize('value', ['青葉ビル3階', 'メゾン青葉 802号室', '青葉ビル'])
def test_address_without_street_numbers_still_preserves_building(value):
    f = features('address', value)
    assert f.type_word in ('ビル', 'メゾン')
    for index in range(20):
        out = apply('address', value, index)
        assert '青葉' not in out and not out.startswith('<')
        word, _core, attrs = sm.address_building_shape(out)
        assert word == f.type_word and attrs.get('affix', '') == f.affix


@pytest.mark.parametrize('core', jp.BUILDING_B)
def test_unparsed_building_still_avoids_original_pool_core(core):
    value = ADDRESS + ' ' + core + 'ビル3階受付'
    assert features('address', value).type_word is None
    assert core in sm.avoid_parts('address', default_normalize(value))
    for index in range(20):
        assert core not in apply('address', value, index)


@pytest.mark.parametrize('number', [0, 1, 2, 12, 13, 802])
def test_number_exclusion_boundaries(number):
    for kind, value in [('facility', f'青葉ビル{number}階'), ('address', ADDRESS + f' 青葉ビル{number}号室')]:
        for index in range(20):
            out = apply(kind, value, index)
            got = int(re.search(r'([0-9]+)(?:階|号室)$', out)[1])
            assert got in jp.TYPE_NUMBERS and got != number


@pytest.mark.parametrize('kind,value', [('org', '羽澄銀行'), ('facility', '青葉ビル5階'),
                                         ('location', '羽澄駅前'), ('address', ADDRESS + ' 青葉ビル3階')])
def test_packaged_collisions_retry_with_preserved_shape(kind, value):
    from privacy_masker.operators.surrogate_filter import SurrogateFilter
    first = apply(kind, value)
    key = b'type-word-key-0'
    op = Surrogate(kind=kind, key=key)
    op._filter = SurrogateFilter(packaged=PublicKnowledge.from_iterable([first]))
    out = op.apply(value, Entity(kind.upper(), 0, len(value), 1.0), SurrogateCtx(key))
    # Full addresses deliberately skip public checks, so deny their building component.
    if kind == 'address':
        word, core, _ = sm.address_building_shape(first)
        op._filter = SurrogateFilter(packaged=PublicKnowledge.from_iterable([core + word]))
        out = op.apply(value, Entity('ADDRESS', 0, len(value), 1.0), SurrogateCtx(key))
    assert out != first and not out.startswith('<')
    assert features(kind, out).type_word == features(kind, value).type_word


@pytest.mark.parametrize('value', ['総合病院', '医療センター', 'バスターミナル', '市立小学校'])
def test_bare_types_do_not_invent_a_proper_core(value):
    assert features('facility', value).type_word is None
