"""ORG suffix preservation and safety, using only synthetic examples and generic vocabulary."""
from dataclasses import replace
from itertools import product
import random

import pytest

from privacy_masker.data import ja_pools as jp
from privacy_masker.normalize import default_normalize
from privacy_masker.operators import Surrogate
from privacy_masker.operators import semantic as sm
from privacy_masker.operators import surrogate_generators as sg
from privacy_masker.types import Entity
from .conftest import SurrogateCtx


EDUCATION = jp.ORG_TYPE_WORDS['education']
GOV_ADDED = ('議院', '会計検査院', '人事院', '県警', '府警', '道警', '警視庁', '検察', '検察庁',
             '地検', '高検', '最高検', '地裁', '高裁', '最高裁', '家裁', '政府', '内閣', '内閣府', '官邸')
ASSOC_ADDED = ('連合', '同盟', '協議会', '労組', '労連')
NEW_RULES = [(cat, word) for cat, words in (
    ('education', EDUCATION), ('government', GOV_ADDED), ('association', ASSOC_ADDED)
) for word in words]
TYPES = [(cat, word) for cat, words in jp.ORG_TYPE_WORDS.items() for word in words]


@pytest.mark.parametrize('category,word', NEW_RULES)
def test_new_rule_is_terminal_and_does_not_override_corporate_forms(category, word):
    value = '羽澄' + word
    assert sm.RuleCategorizer.categorize('org', value) == category
    # Every added word gets a non-terminal counterexample and corporate forms on both sides.
    assert sm.RuleCategorizer.categorize('org', value + '商店') == 'retail'
    assert sm.RuleCategorizer.categorize('org', '株式会社' + value + '商店') == 'retail'
    assert sm.RuleCategorizer.categorize('org', value + '株式会社') == 'service'


@pytest.mark.parametrize('value,expected', [
    ('羽澄大学院大学', 'education'), ('羽澄大学院', 'education'),
    ('学校法人羽澄大学', 'school_corp'), ('羽澄学院', 'school_corp'), ('羽澄学園', 'school_corp'),
    ('羽澄病院', 'service'), ('羽澄寺院', 'service'), ('羽澄美容院', 'service'), ('羽澄院', 'service'),
    ('医療法人羽澄病院', 'medical_corp'), ('羽澄検査院', 'service'),
    ('拡大', 'service'), ('最大', 'service'), ('羽大', 'service'), ('株式会社羽大', 'service'),
    ('羽澄政治協会', 'party'), ('羽澄政治連盟', 'party'), ('羽澄連合党', 'party'),
    ('株式会社羽澄連合', 'service'), ('株式会社羽澄大学', 'service'),
    ('株式会社羽澄県警', 'service'), ('株式会社羽澄政府', 'service'),
    # Existing suffix rules still apply to corporations, including their historical quirks.
    ('株式会社羽澄協会', 'association'), ('株式会社羽澄協議会', 'government'),
    ('株式会社羽澄検察庁', 'government'), ('羽澄連合会', 'association'),
    ('羽澄放送局', 'media'), ('羽澄薬局', 'service'), ('羽澄事務局', 'service'),
    ('羽澄センター', 'service'), ('羽澄都', 'service'), ('羽澄市', 'service'),
])
def test_rule_boundaries_and_priority(value, expected):
    assert sm.RuleCategorizer.categorize('org', value) == expected


def example(category, word, core='空澄'):
    # These words require a public administrative form, rather than a bare generic suffix.
    prefix = '会計' if word == '検査院' else '教育' if word == '委員会' else ''
    return core + prefix + word, core + prefix


@pytest.mark.parametrize('category,word', TYPES)
def test_type_word_is_closed_and_core_is_split_before_longest_suffix(category, word):
    value, core = example(category, word)
    assert sm.RuleCategorizer.categorize('org', value) == category
    f = sm.extract_features('org', value, value, category)
    assert f.type_word == word
    assert sm.RuleCategorizer.core_split('org', value) == len(core)
    assert core in sm.avoid_parts('org', value)
    assert '空澄' in sm.avoid_parts('org', value)


@pytest.mark.parametrize('category,word', TYPES)
@pytest.mark.parametrize('core', ['空澄', 'ソラスミ'])
def test_all_preserved_types_never_leak_original_core_over_400_scopes(category, word, core):
    value, full_core = example(category, word, core)
    entity = Entity('ORG', 0, len(value), 1.0)
    op = Surrogate(kind='org', key=b'org-type-safety')
    for i in range(400):
        ctx = SurrogateCtx(b'org-type-safety', scope_id=str(i), text=value if i % 2 else None)
        out = op.apply(value, entity, ctx)
        assert core not in out and full_core not in out
        assert not out.startswith('<ORG_')
        assert out.endswith(word)


@pytest.mark.parametrize('category,word', TYPES)
def test_all_typed_candidates_are_enumerated_and_have_large_space(category, word, candidate_universe):
    class Pick:
        def __init__(self, index):
            self.index = index

        def choice(self, values):
            return values[self.index]

    for script, pool in [('kanji', jp.ORG_SERVICE_CORES), ('kata', jp.ORG_SERVICE_KATA_CORES)]:
        f = sm.Features('org', category, script, 'medium', type_word=word)
        outputs = {sg.gen_org(f, Pick(i)) for i in range(len(pool))}
        assert len(outputs) == len(pool) >= 100
        assert outputs <= candidate_universe


@pytest.fixture(scope='module')
def candidate_universe():
    return set(sg.enumerate_candidates())


@pytest.mark.parametrize('category', jp.ORG_TYPE_WORDS)
def test_default_type_and_closed_feature_validation(category):
    f = sm.extract_features('org', '羽澄', '羽澄', category)
    assert f.type_word is None
    for i in range(100):
        assert sg.gen_org(f, random.Random(i)).endswith(jp.ORG_TYPE_DEFAULTS[category])
    with pytest.raises(ValueError, match='closed vocabulary'):
        replace(f, type_word='固有の名前')
    with pytest.raises(ValueError, match='closed vocabulary'):
        replace(f, type_word='病院')


@pytest.mark.parametrize('value', ['人事院', '警視庁', '大学院大学', '労働組合'])
def test_whole_type_or_unique_institution_has_no_preserved_suffix(value):
    cat = sm.RuleCategorizer.categorize('org', value)
    assert sm.extract_features('org', value, value, cat).type_word is None


@pytest.mark.parametrize('category,word', [('education', '大学'), ('government', '県警'), ('association', '連合')])
def test_typed_pool_exhaustion_uses_token_and_saved_vault_stays_unchanged(category, word):
    value = '空澄' + word
    entity = Entity('ORG', 0, len(value), 1.0)
    ctx = SurrogateCtx()
    denied = [core + word for core in jp.ORG_SERVICE_CORES]
    out = Surrogate(kind='org', key=b'k', denylist=denied).apply(value, entity, ctx)
    assert out.startswith('<ORG_')
    ctx.save_token('ORG', default_normalize(value), '旧対応', original_value=value)
    assert Surrogate(kind='org', key=b'k', strategy='vault').apply(value, entity, ctx) == '旧対応'


def test_single_character_core_is_also_avoided():
    value = '空大学'
    assert '空' in sm.avoid_parts('org', value)
    entity = Entity('ORG', 0, len(value), 1.0)
    for i in range(400):
        out = Surrogate(kind='org', key=b'k').apply(value, entity, SurrogateCtx(scope_id=str(i)))
        assert '空' not in out and out.endswith('大学')


def test_embedded_neutral_cores_are_avoided_for_typed_organizations():
    for category, word in [('education', '大学'), ('government', '警察署'), ('association', '協会')]:
        for core in jp.ORG_SERVICE_CORES + jp.ORG_SERVICE_KATA_CORES:
            value = '北' + core + 'の輪' + word
            assert core in sm.avoid_parts('org', value)


@pytest.mark.parametrize('category', jp.ORG_TYPE_WORDS)
def test_category_head_only_also_avoids_embedded_neutral_core(category):
    value = '北空澄の輪'
    op = Surrogate(kind='org', key=b'k', category_head=lambda kind, vector: (category, 1.0))
    entity = Entity('ORG', 0, len(value), 1.0, vector=b'vector')
    for i in range(400):
        out = op.apply(value, entity, SurrogateCtx(scope_id=str(i)))
        assert '空澄' not in out
        assert out.endswith(jp.ORG_TYPE_DEFAULTS[category])


def test_hospital_keeps_medical_facility_category():
    assert sm.RuleCategorizer.categorize('facility', '羽澄病院') == 'medical'
