"""Unknown organizations must not acquire an invented industry."""
import hashlib
import json
import random
from itertools import product
from pathlib import Path

import pytest

from privacy_masker.data import ja_pools as jp
from privacy_masker.normalize import default_normalize
from privacy_masker.operators import Surrogate
from privacy_masker.operators import semantic as sm
from privacy_masker.operators import surrogate_generators as sg
from privacy_masker.types import Entity
from .conftest import SurrogateCtx

# Include industry fragments as well as complete category words; neutral corporate
# suffixes (including Partners/Holdings) are deliberately not forbidden.
INDUSTRY_WORDS = set(jp.ORG_INDUSTRIES) | {
    w for words in jp.ORG_INDUSTRY_WORDS.values() for w in words
} | set(jp.ORG_MEDIA_WORDS + jp.ORG_KNOWN_SUFFIXES) | {
    'テクノ', 'ソリュー', 'メディ', 'ロジ', 'システム', 'ネット', 'デジタル',
    'コム', 'ラボ', 'ワークス', '産業', '興業', '商会', '企画',
    '製造', '金融', '小売', '医療', '教育', 'メディア', '物流',
}
ASSOCIATION_WORDS = set('紡織工商電建運薬食印金銀鉄鋼油田学院会党庁局省堂館屋店舗社寺宮製産業技研') | {
    'ツムギ', 'ツムグ', 'オリ', 'オル',
}


def features(script='kanji', **kwargs):
    return sm.Features('org', 'service', script, 'medium', **kwargs)


@pytest.mark.parametrize('script', ['kanji', 'kata'])
def test_service_does_not_invent_an_industry(script):
    for seed in range(3000):
        out = sg.gen_org(features(script), random.Random(seed))
        assert not any(word in out for word in INDUSTRY_WORDS)
        assert not any(word in out for word in ASSOCIATION_WORDS)
        if script == 'kata':
            assert all('ァ' <= c <= 'ヶ' or c == 'ー' for c in out)


# Frozen on f4c96a6 before changing the generator or packaged denylist.
ORG_INPUTS = (
    '株式会社青葉電機', 'アオバシステムズ株式会社', '東雲信用金庫', '若葉商店',
    '株式会社青葉建設', '若葉運輸', '医療法人伊織会', '学校法人若葉学園',
    '青葉開発機構', '青葉協会', '青葉通信社', '青葉党', '営業二課',
)
OTHER_INPUTS = (
    ('person', '山田太郎'), ('person', '山田'), ('person', '佐藤 花子'),
    ('person', 'ミゲル・サントス'), ('person', 'たなか'), ('person', 'John Smith'),
    ('person', 'ｻﾄｳ'), ('person', 'スズキ タロウ'),
    ('facility', '汐見クリニック'), ('facility', '市立若草小学校'), ('facility', '汐見駅'),
    ('location', '大分県'), ('location', '札幌市'), ('location', 'パリ'), ('location', '西新宿2丁目'),
    ('address', '東京都新宿区西新宿1-2-3'),
    ('address', '神奈川県横浜市中区太田町５丁目８番地 サンハイツ301'),
)
KEY = b'service-compatibility-key'


def compatibility_digests():
    """Compare actual output sequences, including RNG consumption and filtering."""
    groups = {}
    def digest(name, outputs):
        groups[name] = hashlib.sha256(json.dumps(outputs, ensure_ascii=False).encode()).hexdigest()

    for category in sm.ORG_CATEGORIES:
        if category in ('service', 'government', 'association', 'education'):
            continue
        outputs = []
        for script, form, pos, seed in product(
            ('kanji', 'kata', 'latin_abbr'), (None,) + sm.LEGAL_FORMS, ('pre', 'post'), range(20)
        ):
            f = sm.Features('org', category, script, 'medium', legal_form=form, legal_form_pos=pos)
            rng = random.Random(seed)
            outputs.append((sg.generate(f, rng), rng.random()))
        digest('generator/' + category, outputs)
    digest('generator/service-latin', [sg.gen_org(features('latin_abbr'), random.Random(i)) for i in range(500)])

    cases = [('org', value) for value in ORG_INPUTS] + list(OTHER_INPUTS)
    for semantic in (True, False):
        inputs = cases if semantic else cases + [('org', '株式会社空澄'), ('org', 'ソラミネ')]
        for kind, value in inputs:
            if semantic and kind == 'org' and sm.RuleCategorizer.categorize(kind, value) in ('government', 'association'):
                continue
            outputs = []
            # auto also exercises the legacy token fallback for FACILITY/LOCATION.
            for i in range(20):
                key = KEY + str(i).encode()
                op = Surrogate(kind='auto', key=key, semantic=semantic)
                outputs.append(op.apply(value, Entity(kind.upper(), 0, len(value), 1.0), SurrogateCtx(key)))
            digest(f'operator/{semantic}/{kind}/{value}', outputs)

    for value, denied in (
        ('青葉通信社', [c + w for c, w in product(jp.ORG_CORES, jp.ORG_MEDIA_WORDS)]),
        ('営業二課', [c + w for c, w in product(jp.DEPT_FUNCTIONS, jp.DEPT_UNITS)]),
    ):
        outputs = []
        for i in range(20):
            key = KEY + str(i).encode()
            op = Surrogate(kind='org', key=key, denylist=denied)
            outputs.append(op.apply(value, Entity('ORG', 0, len(value), 1.0), SurrogateCtx(key)))
        digest('category-fallback/' + value, outputs)
    return groups


def test_non_service_and_legacy_outputs_match_v0_13():
    expected = json.loads((Path(__file__).parent / 'golden/surrogate_compat_v0_13.json').read_text())
    assert compatibility_digests() == expected


@pytest.mark.parametrize('script,pool', [
    ('kanji', jp.ORG_SERVICE_CORES), ('kata', jp.ORG_SERVICE_KATA_CORES),
])
def test_service_candidate_space_and_exhaustive_enumeration(script, pool):
    # Select every pair explicitly, so this tests closure rather than hoping
    # random seeds eventually sample each candidate (including the empty suffix).
    class Choices:
        def __init__(self, i, j):
            self.indices = iter((i, j))

        def choice(self, values):
            return values[next(self.indices)]

    universe = set(sg.enumerate_candidates())
    bodies = {c + s for c, s in product(pool, jp.ORG_SERVICE_SUFFIXES)}
    assert len(set(pool)) == len(pool) >= 200
    assert len(bodies) == len(pool) * len(jp.ORG_SERVICE_SUFFIXES)
    assert not any(w in b for w, b in product(INDUSTRY_WORDS, bodies))
    assert not any(w in c for w, c in product(ASSOCIATION_WORDS, pool))
    for form, pos in product((None,) + sm.LEGAL_FORMS, ('pre', 'post')):
        outputs = set()
        suffixes = jp.ORG_SERVICE_SUFFIXES if form is not None else jp.ORG_SERVICE_SUFFIXES[1:]
        suffix_indices = range(len(suffixes))
        for i, j in product(range(len(pool)), suffix_indices):
            body = pool[i] + suffixes[j]
            if form in ('(株)', '（株）'):
                expected = '（株）' + body
            elif form is None:
                expected = body
            else:
                expected = body + form if pos == 'post' else form + body
            out = sg.gen_org(features(script, legal_form=form, legal_form_pos=pos), Choices(i, j))
            assert out == expected
            assert out in universe
            outputs.add(out)
        assert len(outputs) == len(pool) * len(suffixes)
    assert all(c + s in universe for c, s in product(pool, jp.ORG_SERVICE_SUFFIXES[1:]))
    # Bare strings may still occur in PERSON/LOCATION; service itself never emits them.


@pytest.mark.parametrize('prefixes,endings,exclusions,pool', [
    (jp.ORG_SERVICE_PREFIXES, jp.ORG_SERVICE_ENDINGS, jp.ORG_SERVICE_CORE_EXCLUSIONS, jp.ORG_SERVICE_CORES),
    (jp.ORG_SERVICE_KATA_PREFIXES, jp.ORG_SERVICE_KATA_ENDINGS,
     jp.ORG_SERVICE_KATA_CORE_EXCLUSIONS, jp.ORG_SERVICE_KATA_CORES),
])
def test_service_cores_are_ordered_products_without_exclusions(prefixes, endings, exclusions, pool):
    raw = [a + b for a, b in product(prefixes, endings)]
    assert exclusions <= set(raw)
    assert pool == tuple(c for c in raw if c not in exclusions)


@pytest.mark.parametrize('value', ['北空澄の輪', 'オオソラミネノワ', '株式会社北空澄の輪', 'オオソラミネノワ株式会社'])
def test_service_suffix_condition_and_determinism_through_operator(value):
    norm = default_normalize(value)
    f = sm.extract_features('org', value, norm, 'service')
    assert sm.RuleCategorizer.categorize('org', norm) == 'service'
    pool = jp.ORG_SERVICE_KATA_CORES if f.script == 'kata' else jp.ORG_SERVICE_CORES
    has_suffix = set()
    for i in range(100):
        key = KEY + str(i).encode()
        entity = Entity('ORG', 0, len(value), 1.0)
        op = Surrogate(kind='org', key=key)
        out = op.apply(value, entity, SurrogateCtx(key))
        assert out == op.apply(value, entity, SurrogateCtx(key))
        form, pos, body = sm.split_legal_form(default_normalize(out))
        assert (form, pos) == (f.legal_form, f.legal_form_pos)
        has_suffix.add(any(body.endswith(s) for s in jp.ORG_SERVICE_SUFFIXES[1:]))
        if form is None:
            assert any(body == c + s for c, s in product(pool, jp.ORG_SERVICE_SUFFIXES[1:]))
    assert has_suffix == ({True} if f.legal_form is None else {False, True})


@pytest.mark.parametrize('core', jp.ORG_SERVICE_CORES + jp.ORG_SERVICE_KATA_CORES)
def test_service_never_reuses_original_core(core):
    # Embedded cores must be avoided even when core_split cannot isolate them.
    body = f'オオ{core}ノワ' if core in jp.ORG_SERVICE_KATA_CORES else f'北{core}の輪'
    value = f'株式会社{body}'
    entity = Entity('ORG', 0, len(value), 1.0)
    assert sm.RuleCategorizer.categorize('org', default_normalize(value)) == 'service'
    f = sm.extract_features('org', value, default_normalize(value), 'service')
    assert f.script == ('kata' if core in jp.ORG_SERVICE_KATA_CORES else 'kanji')
    for i in range(100):
        key = KEY + str(i).encode()
        out = Surrogate(kind='org', key=key).apply(value, entity, SurrogateCtx(key))
        assert core not in out
        assert not out.startswith('<ORG_')


@pytest.mark.parametrize('value', ['空澄', 'ソラミネ'])
@pytest.mark.parametrize('blocked_by', ['document', 'denylist', 'packaged', 'knowledge'])
def test_service_safety_filter_retries(value, blocked_by):
    from privacy_masker.annotators import PublicKnowledge
    from privacy_masker.operators.surrogate_filter import SurrogateFilter

    entity = Entity('ORG', 0, len(value), 1.0)
    first = Surrogate(kind='org', key=KEY).apply(value, entity, SurrogateCtx(KEY))
    assert not first.startswith('<ORG_')
    ctx = SurrogateCtx(KEY)
    op = Surrogate(kind='org', key=KEY)
    if blocked_by == 'document':
        ctx.text = f'{value}と{first}が参加した。'
    elif blocked_by == 'denylist':
        op = Surrogate(kind='org', key=KEY, denylist=[first])
    elif blocked_by == 'knowledge':
        op = Surrogate(kind='org', key=KEY, knowledge=PublicKnowledge.from_iterable([first]))
    else:
        op._filter = SurrogateFilter(packaged=PublicKnowledge.from_iterable([first]))
    out = op.apply(value, entity, ctx)
    assert out != first
    assert value not in out
    assert not out.startswith('<ORG_')


@pytest.mark.parametrize('script,value,pool', [
    ('kanji', '空澄', jp.ORG_SERVICE_CORES), ('kata', 'ソラミネ', jp.ORG_SERVICE_KATA_CORES),
])
def test_service_exhausted_pool_falls_back_to_token(script, value, pool):
    denied = [c + s for c, s in product(pool, jp.ORG_SERVICE_SUFFIXES)]
    op = Surrogate(kind='org', key=KEY, denylist=denied)
    out = op.apply(value, Entity('ORG', 0, len(value), 1.0), SurrogateCtx(KEY))
    assert out.startswith('<ORG_') and out.endswith('>')


def test_saved_service_vault_mapping_is_reused():
    value = '株式会社空澄'
    ctx = SurrogateCtx(KEY)
    ctx.save_token('ORG', default_normalize(value), '株式会社若葉工業', original_value=value)
    out = Surrogate(kind='org', strategy='vault', key=KEY).apply(value, Entity('ORG', 0, len(value), 1.0), ctx)
    assert out == '株式会社若葉工業'
