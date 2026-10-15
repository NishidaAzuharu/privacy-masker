"""HandleDetector（チャット・ログのハンドル名、v0.8.1）。盲検評価 v1 の漏れを元にした例。"""
from privacy_masker.detectors import HandleDetector


def spans(text):
    return [text[e.start:e.end] for e in HandleDetector().detect(text)]


def test_slack_style_chat_with_timestamps():
    text = ("#proj\n[09:12] yuki_t: おはようございます\n[09:14] hamada.k: 了解。会場は？\n"
            "[09:14] yuki_t: はい。@nao さんも来られますか\n[09:20] nao: 行けます！")
    assert spans(text) == ["yuki_t", "hamada.k", "yuki_t", "nao", "nao"]


def test_plain_speaker_labels_need_repetition_or_handle_shape():
    text = "Kevin: 今日の飲み会どこだっけ\nじゅん: 名駅\nKevin: 了解\nkurata: OK\nD.Kim: 本番DBの件"
    # Kevin は 2 回出るので話者、kurata は 1 回だけで普通の単語 → 拾わない、D.Kim はハンドルらしい形
    assert spans(text) == ["Kevin", "Kevin", "D.Kim"]


def test_header_words_and_soap_are_not_handles():
    text = "Subject: Re: schedule\nTo: a@example.com\nTel: 03-1234-5678\nS: 痛みあり\nO: 体温 37.1\nA: 良好\nP: 継続\nNOTE: 要確認\nTODO: 返信"
    assert spans(text) == []


def test_at_handle_excludes_email_addresses():
    text = "連絡は @haruka_ig か r.miwa@shinonome-lg.example.jp まで。@sora_kitchen も見てる"
    assert spans(text) == ["haruka_ig", "sora_kitchen"]


def test_log_key_values_and_ticket_agent():
    text = ("2025-09-12 WARN auth login_failed user=echizen.k tenant=midoriya\n"
            "INFO config reloaded by=d.kim from=10.8.0.3\nINFO healthcheck ok checked_by=kurata\n"
            "front-desk account (user: marine_fd01) cannot log in\nAgent (kurata): The account was locked")
    assert spans(text) == ["echizen.k", "d.kim", "kurata", "marine_fd01", "kurata"]


def test_entity_type_score_and_source():
    (e,) = HandleDetector().detect("@taro_1234 さん")
    assert (e.entity_type, e.score, e.source) == ("USERNAME", 0.8, "handle")


def test_yaml_and_list_keys_are_not_speakers():
    text = ("notifications:\n  - channel: mail\n    owner: 根岸貞宏\n  - channel: sms\n    owner: 早瀬春人\n"
            "  - channel: push\n    owner: 谷口蛍\n")
    assert spans(text) == []
    # 会話の話者は拾う（コロンの後が発話）
    chat = "kevin: 今日どこ行く？\nnao: 名駅でいい？\nkevin: いいね\nnao: 了解"
    assert spans(chat) == ["kevin", "nao", "kevin", "nao"]
