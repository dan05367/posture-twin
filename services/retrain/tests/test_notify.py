from app.notify import build_requests, send


class FakeResponse:
    def close(self):
        pass


def test_no_channels_configured_means_no_requests():
    assert build_requests("t", "m", env={}) == []


def test_all_channels_payloads():
    env = {"DISCORD_WEBHOOK_URL": "https://d/x", "SLACK_WEBHOOK_URL": "https://s/y",
           "NTFY_TOPIC": "my-topic", "NTFY_SERVER": "https://ntfy.example/"}
    by_name = {n: (u, b) for n, u, b in build_requests("標題", "內容", env)}
    assert by_name["discord"][1]["content"].startswith("**標題**")
    assert by_name["slack"][1]["text"].startswith("*標題*")
    assert by_name["ntfy"] == ("https://ntfy.example/", {"topic": "my-topic", "title": "標題", "message": "內容"})


def test_send_reports_delivered_and_survives_failures():
    calls = []

    def opener(req, timeout):
        calls.append(req.full_url)
        if "slack" in req.full_url:
            raise OSError("boom")
        return FakeResponse()

    env = {"DISCORD_WEBHOOK_URL": "https://discord/x", "SLACK_WEBHOOK_URL": "https://slack/y"}
    assert send("t", "m", env=env, opener=opener) == ["discord"]
    assert len(calls) == 2
