"""SMTP wiring without network access or actual email delivery."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from sima_vision import alerts


def config(**kwargs):
    return replace(alerts.AlertConfig(
        enable=True, dry_run=False, host="smtp.example.com",
        sender="camera@example.com", recipients=("staff@example.com",),
    ), **kwargs)


@pytest.mark.parametrize("implicit", [False, True])
def test_email_transport_and_cooldown(monkeypatch, implicit):
    smtp = MagicMock()
    smtp.return_value.__enter__.return_value.send_message.return_value = {}
    monkeypatch.setattr(alerts.smtplib, "SMTP_SSL" if implicit else "SMTP", smtp)
    monkeypatch.setenv("FALL_ALERT_SMTP_PASSWORD", "secret")
    sender = alerts.FallAlerts(config(ssl=implicit, starttls=not implicit, username="camera"))
    sender.notify("track #3; frame 10; source time 1.00s")
    sender.worker.join(timeout=5)
    sender.notify("track #4")
    assert smtp.call_count == 1
    connection = smtp.return_value.__enter__.return_value
    assert connection.starttls.call_count == (0 if implicit else 1)
    connection.login.assert_called_once_with("camera", "secret")
    message = connection.send_message.call_args.args[0]
    assert "track #3" in message.get_content()
    assert message["To"] == "staff@example.com"


def test_disabled_and_dry_run_never_connect(monkeypatch):
    smtp = MagicMock()
    monkeypatch.setattr(alerts.smtplib, "SMTP", smtp)
    for cfg in (config(enable=False), config(dry_run=True)):
        sender = alerts.FallAlerts(cfg)
        sender.notify("track #1")
        assert sender.worker is None
    smtp.assert_not_called()


def test_send_failure_is_logged_without_server_response(monkeypatch):
    monkeypatch.setattr(alerts.smtplib, "SMTP", MagicMock(side_effect=OSError("secret")))
    report = MagicMock()
    monkeypatch.setattr(alerts.console, "report", report)
    sender = alerts.FallAlerts(config())
    sender.notify("track #1")
    sender.worker.join(timeout=5)
    assert "OSError" in report.call_args.args[0]
    assert "secret" not in report.call_args.args[0]


@pytest.mark.parametrize("changes", [
    {"host": ""}, {"recipients": ()}, {"ssl": True},
    {"timeout": 0}, {"cooldown_seconds": -1}, {"port": 0},
    {"sender": "a\nb"}, {"username": "camera", "starttls": False},
])
def test_invalid_settings_rejected(changes):
    with pytest.raises(ValueError):
        alerts.validate_alerts(config(**changes))


def test_nested_configuration():
    cfg = alerts.load_alert_config({"alerts": {
        "enable": True, "from": "a@example.com", "to": ["b@example.com"],
        "smtp": {"host": "mail.example.com", "port": 465, "ssl": True, "starttls": False},
    }})
    assert cfg.enable and cfg.ssl and not cfg.starttls
    assert cfg.sender == "a@example.com"
    assert cfg.recipients == ("b@example.com",)
    assert cfg.port == 465


def test_fall_runtime_groups_only_newly_confirmed_tracks():
    from sima_vision.tasks.fall import FallAppConfig, FallPipeline, FallRuntime, Track

    notifier = MagicMock()
    pipeline = FallPipeline(alerts=notifier, frame_h=1080)
    cfg = FallAppConfig()
    runtime = FallRuntime()
    runtime.report_falls(pipeline, cfg, [], 1, 0.0)
    notifier.notify.assert_not_called()
    tracks = [Track(track_id=i, box={"x1": 0, "y1": 0, "x2": 300, "y2": 120})
              for i in (1, 2)]
    runtime.report_falls(pipeline, cfg, tracks, 2, 1.0)
    assert pipeline.falls == 2
    notifier.notify.assert_called_once()
    assert "track #1; track #2" in notifier.notify.call_args.args[0]
