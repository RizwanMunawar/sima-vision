"""Optional SMTP notifications for confirmed falls (standard library only)."""

from __future__ import annotations

import math
import os
import smtplib
import ssl
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage

from .config import _flag, _float, _int, _section, _str, _str_list
from .console import console


@dataclass(frozen=True)
class AlertConfig:
    enable: bool = False
    dry_run: bool = True
    site: str = "Fall detection"
    sender: str = ""
    recipients: tuple[str, ...] = ()
    cooldown_seconds: float = 60.0
    host: str = ""
    port: int = 587
    starttls: bool = True
    ssl: bool = False
    username: str = ""
    password_env: str = "FALL_ALERT_SMTP_PASSWORD"
    timeout: float = 20.0


def load_alert_config(raw: dict) -> AlertConfig:
    a = _section(raw, "alerts")
    s = _section(a, "smtp")
    return AlertConfig(
        enable=_flag(a, "enable", "off") == "on",
        dry_run=_flag(a, "dry_run", "on") == "on",
        site=_str(a, "site", "Fall detection"),
        sender=_str(a, "from"), recipients=_str_list(a, "to", ()),
        cooldown_seconds=_float(a, "cooldown_seconds", 60.0),
        host=_str(s, "host"), port=_int(s, "port", 587),
        starttls=_flag(s, "starttls", "on") == "on",
        ssl=_flag(s, "ssl", "off") == "on",
        username=_str(s, "username"),
        password_env=_str(s, "password_env", "FALL_ALERT_SMTP_PASSWORD"),
        timeout=_float(s, "timeout", 20.0),
    )


def validate_alerts(cfg: AlertConfig) -> None:
    if not cfg.enable:
        return
    if not math.isfinite(cfg.cooldown_seconds) or cfg.cooldown_seconds < 0:
        raise ValueError("alerts.cooldown_seconds must be finite and >= 0")
    if not math.isfinite(cfg.timeout) or cfg.timeout <= 0:
        raise ValueError("alerts.smtp.timeout must be finite and > 0")
    if not 1 <= cfg.port <= 65535:
        raise ValueError("alerts.smtp.port must be in [1, 65535]")
    if cfg.ssl and cfg.starttls:
        raise ValueError("alerts.smtp: choose ssl or starttls, not both")
    for value in (cfg.site, cfg.sender, *cfg.recipients):
        if "\r" in value or "\n" in value:
            raise ValueError("alerts: header values cannot contain newlines")
    if not cfg.dry_run:
        if not cfg.host.strip() or not cfg.sender.strip() or not cfg.recipients:
            raise ValueError("alerts require smtp.host, from and to")
        if any(not recipient.strip() for recipient in cfg.recipients):
            raise ValueError("alerts.to cannot contain empty recipients")
        if cfg.username and not os.environ.get(cfg.password_env):
            raise ValueError(f"alerts: set the {cfg.password_env} environment variable")
        if cfg.username and not (cfg.ssl or cfg.starttls):
            raise ValueError("alerts: SMTP authentication requires TLS")


class FallAlerts:
    """One in-flight email at most; SMTP never blocks the frame-processing thread."""

    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self.last_sent = float("-inf")
        self.worker: threading.Thread | None = None

    def notify(self, details: str) -> None:
        cfg = self.cfg
        now = time.monotonic()
        if not cfg.enable or now - self.last_sent < cfg.cooldown_seconds:
            return
        if self.worker is not None and self.worker.is_alive():
            console.report("[EMAIL] Skipped alert: previous send still in progress")
            return
        message = EmailMessage()
        message["Subject"] = f"[{cfg.site}] Fall detected"
        message["From"] = cfg.sender
        message["To"] = ", ".join(cfg.recipients)
        message.set_content(
            f"Site: {cfg.site}\nDetected at: {datetime.now(timezone.utc).isoformat()}\n"
            f"{details}\nPlease check the camera and person.\n"
        )
        self.last_sent = now
        if cfg.dry_run:
            console.report(f"[EMAIL dry run] {cfg.site}: {details}")
            return
        # Non-daemon: let an accepted send finish on normal process exit.
        self.worker = threading.Thread(target=self._send, args=(message,), daemon=False)
        self.worker.start()

    def _send(self, message: EmailMessage) -> None:
        cfg = self.cfg
        try:
            context = ssl.create_default_context()
            factory = smtplib.SMTP_SSL if cfg.ssl else smtplib.SMTP
            kwargs = {"context": context} if cfg.ssl else {}
            with factory(cfg.host, cfg.port, timeout=cfg.timeout, **kwargs) as smtp:
                if cfg.starttls:
                    smtp.starttls(context=context)
                if cfg.username:
                    smtp.login(cfg.username, os.environ[cfg.password_env])
                refused = smtp.send_message(message)
                console.report("[EMAIL] Some recipients refused" if refused else "[EMAIL] Sent")
        except Exception as exc:
            # SMTP responses may contain addresses or credentials; log only the type.
            console.report(f"[EMAIL] Send failed ({type(exc).__name__}); check SMTP settings")
