"""Fall detection: track people, judge whether one has gone down, email about it.

Three signals, all available from a plain bounding box:

  aspect    a standing person is tall and narrow, a fallen one is wide and
            short. The box aspect ratio crossing 1.0 is the strongest single
            indicator there is without pose keypoints.
  collapse  the box height drops well below what this person's own upright
            height has been, which separates lying down from crouching.
  descent   the centre of the box moves down fast. This is what distinguishes
            a fall from someone lying down deliberately.

Any one of them can fire spuriously for a frame or two, so nothing is reported
until the condition has held for ``fall.confirm_seconds``. That delay is the
whole difference between a useful alert and one nobody reads.
"""

from __future__ import annotations

import os
import queue
import smtplib
import threading
import time
from dataclasses import dataclass, field, replace
from email.message import EmailMessage
from pathlib import Path

from .. import runtime
from ..config import (
    BaseConfig,
    DrawConfig,
    TaskDefaults,
    _flag,
    _float,
    _int,
    _section,
    _str,
    _str_list,
)
from ..console import console
from ..draw import class_color, draw_banner, draw_caption, draw_fps, draw_scale
from ..runloop import TaskRuntime
from ..samples import (
    extract_bbox_payload,
    first_tensor,
    frame_to_bgr,
    joined_field,
    parse_boxes,
    resolve_classes,
)
from ..sinks import Pipeline, load_labels
from .base import Task

UPRIGHT, FALLING, FALLEN, RECOVERING = "upright", "falling", "fallen", "recovering"

#: The class a fallen track is relabelled to, and the colour that class draws
#: in. Red rather than a palette entry: every other box on the frame is an
#: ordinary detection, and this one is the reason the app exists. It is the
#: banner's default fill, so the box and the strip across the bottom agree.
FALL_CLASS = "FALL"
FALL_COLOR = (56, 56, 255)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrackConfig:
    """How detections are followed from frame to frame.

    Attributes:
        classes: Class names or ids that can fall. Empty means every class,
            which is almost never what you want in a warehouse.
        iou_threshold: Minimum overlap to call two boxes the same person.
            Lower it for a low frame rate, where boxes move further per frame.
        max_age: Frames a track survives with no detection before it is dropped.
            This is what carries someone through a brief occlusion.
        min_hits: Frames a track must be seen on before it is reported at all,
            which suppresses one-frame false positives.
        history_seconds: How much per-track history to keep. Must be at least
            ``fall.descent_window``.
        upright_aspect: Aspect ratio at or below which a person counts as
            upright for the purpose of learning their reference height.
    """

    classes: tuple[str, ...] = ("person",)
    iou_threshold: float = 0.30
    max_age: int = 30
    min_hits: int = 3
    history_seconds: float = 3.0
    upright_aspect: float = 0.70


@dataclass(frozen=True)
class FallConfig:
    """When a tracked person counts as fallen.

    Attributes:
        enable: Whether to evaluate falls at all. ``off`` leaves a plain
            person tracker, which is a useful way to tune the tracking first.
        aspect_ratio: Width over height at or above which the box reads as
            lying down. 1.0 is square; 1.2 gives a little margin against a
            crouch or a wide-armed gesture.
        height_drop: Fraction of this person's own learned upright height at or
            below which they count as collapsed. 0.55 means "less than 55% as
            tall as they were".
        descent_rate: Downward speed of the box centre that reads as a fall,
            as a fraction of frame height per second. 0.55 is roughly half the
            frame in one second.
        descent_window: How far back to measure that speed, in seconds.
        confirm_seconds: How long a signal must hold before an alert fires.
            The single most important knob here: too low and a bending forklift
            driver pages someone, too high and the alert is late.
        recover_seconds: How long someone must look upright again before the
            track is re-armed for another alert.
        min_box_height: Ignore boxes shorter than this fraction of the frame,
            which drops distant figures too small to judge.
    """

    enable: bool = True
    aspect_ratio: float = 1.20
    height_drop: float = 0.55
    descent_rate: float = 0.55
    descent_window: float = 0.7
    confirm_seconds: float = 1.5
    recover_seconds: float = 3.0
    min_box_height: float = 0.08


@dataclass(frozen=True)
class AlertConfig:
    """SMTP notification settings.

    There is no password field on purpose. ``config.yaml`` is committed, so a
    password in it is a password on GitHub. The password is read from the
    environment variable named by ``password_env`` instead.

    Attributes:
        enable: Whether to send anything at all.
        dry_run: Compose and log the message without connecting to a server.
            The way to check subject, recipients and attachment wiring.
        host: SMTP server hostname.
        port: SMTP port. 587 for STARTTLS, 465 for implicit SSL, 25 for a relay.
        ssl: Connect with implicit TLS, for port 465.
        starttls: Upgrade a plaintext connection, for port 587.
        username: SMTP login. Empty means an unauthenticated relay.
        password_env: Environment variable holding the password.
        sender: ``From`` address.
        recipients: ``To`` addresses.
        subject: Subject template. ``{label}``, ``{track_id}``, ``{time}`` and
            ``{site}`` are substituted.
        site: A human name for this camera or building, put in the subject and
            the body so an alert says where to go.
        cooldown_seconds: Minimum gap between alerts, across every track. A
            person falling usually produces one event; a camera pointed at a
            busy aisle should not produce forty.
        attach_snapshot: Whether to attach the annotated frame.
        snapshot_dir: Where snapshots are written on the DevKit.
        queue_depth: Pending alerts held for the sender thread. Beyond this,
            alerts are dropped rather than blocking the pipeline.
        timeout: SMTP socket timeout in seconds.
    """

    enable: bool = False
    dry_run: bool = True
    host: str = "smtp.gmail.com"
    port: int = 587
    ssl: bool = False
    starttls: bool = True
    username: str = ""
    password_env: str = "FALL_ALERT_SMTP_PASSWORD"
    sender: str = ""
    recipients: tuple[str, ...] = ()
    subject: str = "[{site}] Fall detected - track #{track_id} at {time}"
    site: str = "Warehouse camera 1"
    cooldown_seconds: float = 60.0
    attach_snapshot: bool = True
    snapshot_dir: str = "alerts"
    queue_depth: int = 8
    timeout: float = 20.0


@dataclass(frozen=True)
class FallAppConfig(BaseConfig):
    """Base config plus the ``tracking``, ``fall`` and ``alerts`` sections."""

    track: TrackConfig = TrackConfig()
    fall: FallConfig = FallConfig()
    alerts: AlertConfig = AlertConfig()


def load_track_config(raw: dict) -> TrackConfig:
    section = _section(raw, "tracking")
    d = TrackConfig()
    return TrackConfig(
        classes=_str_list(section, "classes", d.classes),
        iou_threshold=_float(section, "iou_threshold", d.iou_threshold),
        max_age=_int(section, "max_age", d.max_age),
        min_hits=_int(section, "min_hits", d.min_hits),
        history_seconds=_float(section, "history_seconds", d.history_seconds),
        upright_aspect=_float(section, "upright_aspect", d.upright_aspect),
    )


def load_fall_config(raw: dict) -> FallConfig:
    section = _section(raw, "fall")
    d = FallConfig()
    return FallConfig(
        enable=_flag(section, "enable", "on") == "on",
        aspect_ratio=_float(section, "aspect_ratio", d.aspect_ratio),
        height_drop=_float(section, "height_drop", d.height_drop),
        descent_rate=_float(section, "descent_rate", d.descent_rate),
        descent_window=_float(section, "descent_window", d.descent_window),
        confirm_seconds=_float(section, "confirm_seconds", d.confirm_seconds),
        recover_seconds=_float(section, "recover_seconds", d.recover_seconds),
        min_box_height=_float(section, "min_box_height", d.min_box_height),
    )


def load_alert_config(raw: dict) -> AlertConfig:
    section = _section(raw, "alerts")
    smtp = _section(section, "smtp")
    d = AlertConfig()
    return AlertConfig(
        enable=_flag(section, "enable", "off") == "on",
        dry_run=_flag(section, "dry_run", "on") == "on",
        host=_str(smtp, "host", d.host),
        port=_int(smtp, "port", d.port),
        ssl=_flag(smtp, "ssl", "off") == "on",
        starttls=_flag(smtp, "starttls", "on") == "on",
        username=_str(smtp, "username", d.username),
        password_env=_str(smtp, "password_env", d.password_env),
        timeout=_float(smtp, "timeout", d.timeout),
        sender=_str(section, "from", d.sender),
        recipients=_str_list(section, "to", d.recipients),
        subject=_str(section, "subject", d.subject),
        site=_str(section, "site", d.site),
        cooldown_seconds=_float(section, "cooldown_seconds", d.cooldown_seconds),
        attach_snapshot=_flag(section, "attach_snapshot", "on") == "on",
        snapshot_dir=_str(section, "snapshot_dir", d.snapshot_dir),
        queue_depth=_int(section, "queue_depth", d.queue_depth),
    )


def validate_fall(cfg: FallAppConfig) -> None:
    if not 0.0 <= cfg.track.iou_threshold <= 1.0:
        raise ValueError("tracking.iou_threshold must be in [0.0, 1.0]")
    if cfg.track.max_age < 0:
        raise ValueError("tracking.max_age must be >= 0")
    if cfg.track.min_hits < 1:
        raise ValueError("tracking.min_hits must be >= 1")
    if cfg.track.history_seconds < cfg.fall.descent_window:
        raise ValueError(
            f"tracking.history_seconds ({cfg.track.history_seconds}) is shorter than "
            f"fall.descent_window ({cfg.fall.descent_window}), so the descent test "
            f"would never see far enough back to fire."
        )
    if cfg.fall.aspect_ratio <= 0:
        raise ValueError("fall.aspect_ratio must be > 0")
    if not 0.0 < cfg.fall.height_drop <= 1.0:
        raise ValueError("fall.height_drop must be in (0.0, 1.0]")
    if cfg.fall.descent_rate < 0:
        raise ValueError("fall.descent_rate must be >= 0")
    if cfg.fall.descent_window <= 0:
        raise ValueError("fall.descent_window must be > 0")
    if cfg.fall.confirm_seconds < 0:
        raise ValueError("fall.confirm_seconds must be >= 0")
    if cfg.fall.recover_seconds < 0:
        raise ValueError("fall.recover_seconds must be >= 0")
    if not 0.0 <= cfg.fall.min_box_height < 1.0:
        raise ValueError("fall.min_box_height must be in [0.0, 1.0)")
    if cfg.alerts.enable and not cfg.alerts.dry_run:
        if not cfg.alerts.host:
            raise ValueError("alerts.smtp.host must be set when alerts are enabled")
        if not cfg.alerts.sender:
            raise ValueError("alerts.from must be set when alerts are enabled")
        if not cfg.alerts.recipients:
            raise ValueError("alerts.to must list at least one recipient")
        if cfg.alerts.ssl and cfg.alerts.starttls:
            raise ValueError(
                "alerts.smtp.ssl and starttls are both on. Use ssl for port 465 "
                "or starttls for port 587, not both."
            )
        if not cfg.alerts.password_env:
            raise ValueError("alerts.smtp.password_env must name an environment variable")
    if cfg.alerts.queue_depth < 1:
        raise ValueError("alerts.queue_depth must be >= 1")
    if cfg.alerts.cooldown_seconds < 0:
        raise ValueError("alerts.cooldown_seconds must be >= 0")


def describe_fall(cfg: FallAppConfig) -> str:
    watched = ", ".join(cfg.track.classes) if cfg.track.classes else "every class"
    if not cfg.fall.enable:
        return f"fall: off, tracking {watched} only"
    return (
        f"fall: watching {watched} | aspect>={cfg.fall.aspect_ratio} "
        f"height<={cfg.fall.height_drop:.0%} descent>={cfg.fall.descent_rate:.0%}/s "
        f"| confirm={cfg.fall.confirm_seconds}s recover={cfg.fall.recover_seconds}s"
    )


def describe_alerts(cfg: FallAppConfig) -> str:
    a = cfg.alerts
    if not a.enable:
        return "alerts: off"
    if a.dry_run:
        return (
            f"alerts: DRY RUN, nothing is sent | would mail {len(a.recipients)} "
            f"recipient(s) | cooldown={a.cooldown_seconds}s"
        )
    mode = "ssl" if a.ssl else ("starttls" if a.starttls else "plain")
    return (
        f"alerts: {a.host}:{a.port} {mode} as {a.username or '<anonymous>'} "
        f"-> {', '.join(a.recipients)} | cooldown={a.cooldown_seconds}s "
        f"snapshot={'on' if a.attach_snapshot else 'off'}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tracking
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Track:
    """One person followed across frames, with just enough history to judge them.

    Attributes:
        track_id: Stable id, assigned once and never reused within a run.
        box: Most recent detection, in source-image pixels.
        hits: How many frames this track has been matched on.
        misses: Consecutive frames with no match. Drops the track past a limit.
        history: Recent ``(timestamp_s, centre_y, height, aspect)`` samples,
            oldest first, trimmed to the window the fall rules need.
        upright_height: Rolling reference height from frames where the person
            looked upright. A fallen box is short as well as wide, but only
            relative to how tall *that* person was, which is why this is
            per-track rather than a constant.
        state: One of ``upright``, ``falling``, ``fallen`` or ``recovering``.
        state_since: Timestamp the current state began.
        alerted_at: When an alert was last raised for this track, or 0.0.
    """

    track_id: int
    box: dict
    # 0, not 1: the frame that creates a track also runs it through _advance,
    # so counting it here too would make min_hits mean one frame fewer than it
    # says.
    hits: int = 0
    misses: int = 0
    history: list = field(default_factory=list)
    upright_height: float = 0.0
    state: str = UPRIGHT
    state_since: float = 0.0
    alerted_at: float = 0.0

    @property
    def width(self) -> float:
        return max(1.0, self.box["x2"] - self.box["x1"])

    @property
    def height(self) -> float:
        return max(1.0, self.box["y2"] - self.box["y1"])

    @property
    def aspect(self) -> float:
        """Width over height. Standing is well under 1; lying down is over it."""
        return self.width / self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.box["x1"] + self.box["x2"]) / 2.0,
                (self.box["y1"] + self.box["y2"]) / 2.0)


def box_iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Tracker:
    """Greedy IoU tracker with stable ids.

    Deliberately simple. It has no motion model, so it will swap ids when two
    people cross while overlapping heavily. That costs a duplicate alert at
    worst, which is the right way round for a safety feature: a Kalman filter
    would be more correct and considerably more to get wrong.

    Attributes:
        cfg: Tracking settings.
        tracks: Live tracks.
    """

    def __init__(self, cfg: TrackConfig) -> None:
        self.cfg = cfg
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes: list[dict], now: float) -> list[Track]:
        """Associate this frame's boxes with existing tracks.

        Args:
            boxes: Detections for this frame, already filtered to the classes
                that count as people.
            now: Monotonic timestamp in seconds.

        Returns:
            Every live track seen often enough to be trusted.
        """
        pairs = sorted(
            (
                (box_iou(t.box, b), ti, bi)
                for ti, t in enumerate(self.tracks)
                for bi, b in enumerate(boxes)
            ),
            key=lambda p: p[0],
            reverse=True,
        )
        used_t: set[int] = set()
        used_b: set[int] = set()
        for score, ti, bi in pairs:
            if score < self.cfg.iou_threshold:
                break
            if ti in used_t or bi in used_b:
                continue
            used_t.add(ti)
            used_b.add(bi)
            self._advance(self.tracks[ti], boxes[bi], now)

        for ti, track in enumerate(self.tracks):
            if ti not in used_t:
                track.misses += 1

        for bi, box in enumerate(boxes):
            if bi not in used_b:
                track = Track(track_id=self._next_id, box=box, state_since=now)
                self._next_id += 1
                self._advance(track, box, now)
                self.tracks.append(track)

        self.tracks = [t for t in self.tracks if t.misses <= self.cfg.max_age]
        return [t for t in self.tracks if t.hits >= self.cfg.min_hits and t.misses == 0]

    def _advance(self, track: Track, box: dict, now: float) -> None:
        track.box = box
        track.misses = 0
        track.hits += 1
        _, cy = track.centre
        track.history.append((now, cy, track.height, track.aspect))
        # Keep only what the rules can still look at, so a long run does not
        # accumulate a per-track list the length of the video.
        cutoff = now - self.cfg.history_seconds
        while len(track.history) > 2 and track.history[0][0] < cutoff:
            track.history.pop(0)
        if track.aspect <= self.cfg.upright_aspect:
            # An exponential rather than a max: someone briefly clipped by the
            # frame edge should not raise the bar for the rest of the run.
            track.upright_height = (
                track.height
                if track.upright_height <= 0
                else 0.9 * track.upright_height + 0.1 * track.height
            )


# ─────────────────────────────────────────────────────────────────────────────
# Fall rules
# ─────────────────────────────────────────────────────────────────────────────


def descent_rate(track: Track, window: float) -> float:
    """Downward speed of the box centre, in pixels per second.

    Args:
        track: The track to measure.
        window: How far back to look, in seconds.

    Returns:
        Pixels per second, positive downwards. 0.0 when there is too little
        history to say.
    """
    if len(track.history) < 2:
        return 0.0
    now = track.history[-1][0]
    oldest = track.history[0]
    for sample in track.history:
        if now - sample[0] <= window:
            oldest = sample
            break
    dt = now - oldest[0]
    if dt <= 1e-3:
        return 0.0
    return (track.history[-1][1] - oldest[1]) / dt


def fall_signals(track: Track, fall: FallConfig, frame_h: int) -> dict:
    """Evaluate all three signals for one track, for judging and for reporting."""
    collapsed = (
        track.upright_height > 0
        and track.height <= track.upright_height * fall.height_drop
    )
    rate = descent_rate(track, fall.descent_window)
    return {
        "aspect": track.aspect >= fall.aspect_ratio,
        "collapse": bool(collapsed),
        "descent": rate >= fall.descent_rate * frame_h,
        "aspect_value": round(track.aspect, 2),
        "descent_value": round(rate, 1),
    }


def looks_fallen(track: Track, fall: FallConfig, frame_h: int) -> bool:
    """Whether this track currently satisfies any of the fall signals."""
    s = fall_signals(track, fall, frame_h)
    return s["aspect"] or s["collapse"] or s["descent"]


def update_fall_states(tracks: list[Track], fall: FallConfig, frame_h: int,
                       now: float) -> list[Track]:
    """Advance every track's state machine and return the ones that just fell.

    The machine is::

        upright ──looks_fallen──> falling ──held confirm_seconds──> fallen
           ^                         |                                 |
           |                     recovered                         recovered
           |                         v                                 v
           └──── held recover_seconds ──── recovering <────────────────┘

    Args:
        tracks: Live, confirmed tracks.
        fall: Fall rule settings.
        frame_h: Frame height, so descent thresholds stay resolution independent.
        now: Monotonic timestamp in seconds.

    Returns:
        Tracks that transitioned into ``fallen`` on this frame.
    """
    newly = []
    for track in tracks:
        down = looks_fallen(track, fall, frame_h)
        if track.state in (UPRIGHT, RECOVERING):
            if down:
                track.state, track.state_since = FALLING, now
            elif (
                track.state == RECOVERING
                and now - track.state_since >= fall.recover_seconds
            ):
                track.state, track.state_since = UPRIGHT, now
        elif track.state == FALLING:
            if not down:
                track.state, track.state_since = RECOVERING, now
            elif now - track.state_since >= fall.confirm_seconds:
                track.state, track.state_since = FALLEN, now
                newly.append(track)
        elif track.state == FALLEN:
            if not down:
                track.state, track.state_since = RECOVERING, now
    return newly


# ─────────────────────────────────────────────────────────────────────────────
# SMTP alerts
#
# Sending mail takes anything from tens of milliseconds to a TCP timeout, and
# the run loop cannot afford either. Alerts are handed to a background thread
# through a bounded queue: if the queue is full the alert is dropped and
# counted, which is the correct trade. A stalled pipeline stops detecting the
# next fall, and that is worse than missing one notification.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Alert:
    """One pending notification."""

    track_id: int
    label: str
    box: tuple[int, int, int, int]
    signals: dict
    frame_index: int
    snapshot_path: str = ""


def alert_password(cfg: AlertConfig) -> str:
    """Read the SMTP password from the environment.

    Deliberately not a config key. ``config.yaml`` is committed to the repo, and
    a password in it is a password on GitHub. The variable is read at send time
    so rotating it does not need a restart.
    """
    if not cfg.username:
        return ""
    password = os.environ.get(cfg.password_env, "")
    if not password:
        raise RuntimeError(
            f"alerts.username is set but ${cfg.password_env} is empty.\n"
            f"  export {cfg.password_env}='...' before running, or set "
            f"alerts.dry_run: true to test without sending."
        )
    return password


class AlertSender:
    """Queues fall alerts and sends them as email from a background thread.

    Attributes:
        cfg: Alert settings.
        sent: Successful sends.
        failed: Sends that raised.
        dropped: Alerts discarded because the queue was full.
        suppressed: Alerts skipped by the cooldown.
    """

    def __init__(self, cfg: AlertConfig) -> None:
        self.cfg = cfg
        self.sent = 0
        self.failed = 0
        self.dropped = 0
        self.suppressed = 0
        self._queue: queue.Queue = queue.Queue(maxsize=cfg.queue_depth)
        self._last_global = 0.0
        self._thread = None
        if cfg.enable:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    # ── producer side, called from the run loop ──
    def offer(self, alert: Alert, now: float) -> bool:
        """Queue an alert unless the cooldown or a full queue says otherwise."""
        if not self.cfg.enable:
            return False
        if now - self._last_global < self.cfg.cooldown_seconds:
            self.suppressed += 1
            return False
        try:
            self._queue.put_nowait(alert)
        except queue.Full:
            self.dropped += 1
            return False
        self._last_global = now
        return True

    def close(self, timeout: float = 10.0) -> None:
        """Drain the queue and stop the worker."""
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout)

    # ── consumer side ──
    def _worker(self) -> None:
        while True:
            alert = self._queue.get()
            if alert is None:
                return
            try:
                self.send_now(alert)
                self.sent += 1
            except Exception as exc:  # pragma: no cover - depends on the network
                self.failed += 1
                console.warn(f"alert send failed: {exc}")

    def build_message(self, alert: Alert):
        """Compose the email, attaching the snapshot when there is one."""
        msg = EmailMessage()
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        msg["Subject"] = self.cfg.subject.format(
            label=alert.label, track_id=alert.track_id, time=when, site=self.cfg.site
        )
        msg["From"] = self.cfg.sender
        msg["To"] = ", ".join(self.cfg.recipients)
        x1, y1, x2, y2 = alert.box
        msg.set_content(
            f"A fall was detected.\n\n"
            f"  site      : {self.cfg.site}\n"
            f"  time      : {when}\n"
            f"  track     : #{alert.track_id} ({alert.label})\n"
            f"  frame     : {alert.frame_index}\n"
            f"  box       : x={x1} y={y1} w={x2 - x1} h={y2 - y1}\n"
            f"  aspect    : {alert.signals.get('aspect_value')} "
            f"(triggered: {alert.signals.get('aspect')})\n"
            f"  collapsed : {alert.signals.get('collapse')}\n"
            f"  descent   : {alert.signals.get('descent_value')} px/s "
            f"(triggered: {alert.signals.get('descent')})\n\n"
            f"Sent by the SiMa Modalix fall-detection app.\n"
        )
        if alert.snapshot_path and Path(alert.snapshot_path).is_file():
            data = Path(alert.snapshot_path).read_bytes()
            msg.add_attachment(
                data, maintype="image", subtype="jpeg",
                filename=Path(alert.snapshot_path).name,
            )
        return msg

    def send_now(self, alert: Alert) -> None:
        """Send one alert synchronously. Raises on any SMTP failure."""
        msg = self.build_message(alert)
        if self.cfg.dry_run:
            console.report(
                f"[alert:dry-run] would email {len(self.cfg.recipients)} recipient(s): "
                f"{msg['Subject']}"
            )
            return
        password = alert_password(self.cfg)
        if self.cfg.ssl:
            server = smtplib.SMTP_SSL(self.cfg.host, self.cfg.port, timeout=self.cfg.timeout)
        else:
            server = smtplib.SMTP(self.cfg.host, self.cfg.port, timeout=self.cfg.timeout)
        try:
            if self.cfg.starttls and not self.cfg.ssl:
                server.starttls()
            if self.cfg.username:
                server.login(self.cfg.username, password)
            server.send_message(msg)
        finally:
            try:
                server.quit()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────


def track_label(track: Track, labels: list[str]) -> str:
    """What a track's box calls it.

    A fallen track is relabelled to :data:`FALL_CLASS`; everything else is the
    class the detector reported. The state machine's own words never appear --
    `upright` and `recovering` are this program's internal vocabulary and mean
    nothing to someone watching a corridor.
    """
    if track.state == FALLEN:
        return FALL_CLASS
    class_id = int(track.box.get("class_id", 0))
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return "person"


def track_color(track: Track) -> tuple[int, int, int]:
    """A fallen track's alert colour, or its ordinary class colour."""
    if track.state == FALLEN:
        return FALL_COLOR
    return class_color(int(track.box.get("class_id", 0)))


def track_caption(track: Track, draw, labels: list[str]) -> str:
    """Build the caption for one tracked person.

    The same shape as a detection's: the class, then the score. A fall is a
    change of class, not a different kind of readout -- so a fallen person
    reads `FALL 0.93` where they read `person 0.93` a second earlier, and
    nothing else about the box moves except its colour.
    """
    parts = []
    if draw.show_track_ids:
        parts.append(f"#{track.track_id}")
    if draw.show_labels:
        parts.append(track_label(track, labels))
    if draw.show_scores:
        parts.append(f"{track.box['score']:.{max(0, draw.score_decimals)}f}")
    return " ".join(parts)


def draw_tracks(frame, tracks: list[Track], draw, labels: list[str]) -> None:
    """Draw every tracked person as an ordinary detection, in place.

    Deliberately the same picture `detect` draws: the class colour, a centre
    dot, and a `class score` caption. A fall changes one thing, the class --
    which changes the caption and the colour with it, and nothing else. The
    intermediate states do not paint themselves: someone watching a corridor
    is watching for a fall, and a box that changes colour twice on the way
    there trains them to ignore it.

    Args:
        frame: BGR image, modified in place.
        tracks: Live tracks with their fall state already resolved.
        draw: Visualization settings.
        labels: Class names, so a box says what was detected.
    """
    cv2 = runtime.cv2
    height, width = frame.shape[:2]
    scale = draw_scale(frame, draw)
    thickness = max(1, int(round(draw.box_thickness * scale)))
    radius = max(2, int(round(draw.centre_dot_radius * scale)))

    # Paint larger boxes first, so a small figure in front stays legible.
    ordered = sorted(tracks, key=lambda t: t.width * t.height, reverse=True)
    for track in ordered:
        x1 = max(0, int(round(track.box["x1"])))
        y1 = max(0, int(round(track.box["y1"])))
        x2 = min(width - 1, int(round(track.box["x2"])))
        y2 = min(height - 1, int(round(track.box["y2"])))
        if x2 <= x1 or y2 <= y1:
            continue

        color = track_color(track)
        if draw.centre_dot:
            cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), radius, color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        draw_caption(frame, track_caption(track, draw, labels), (x1, y1),
                     color, draw, scale)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline and run loop
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class FallPipeline(Pipeline):
    """Pipeline plus the tracker, the alert sender and the fall tally.

    Attributes:
        tracker: The :class:`Tracker` following people across frames.
        alerts: The :class:`AlertSender` owning the SMTP thread.
        fall_class_ids: Class ids that can fall, or None for every class.
        falls: Confirmed falls so far this run.
    """

    tracker: object = None
    alerts: object = None
    fall_class_ids: object = None
    falls: int = 0

    def close_extras(self) -> None:
        if self.alerts is None:
            return
        # Drain first: an alert queued microseconds before Ctrl-C is still a
        # fall that happened, and the thread is a daemon so nothing else would
        # wait for it.
        stats = self.alerts
        stats.close()
        if stats.sent or stats.failed or stats.dropped or stats.suppressed:
            console.report(
                f"alerts: sent={stats.sent} failed={stats.failed} "
                f"dropped={stats.dropped} suppressed_by_cooldown={stats.suppressed}"
            )
        self.alerts = None


def person_boxes(cfg: FallAppConfig, pipeline: FallPipeline, boxes: list[dict],
                 frame_h: int) -> list[dict]:
    """Keep only the classes that can fall, and only boxes big enough to judge."""
    floor = cfg.fall.min_box_height * frame_h
    kept = []
    for box in boxes:
        if pipeline.fall_class_ids is not None:
            if int(box["class_id"]) not in pipeline.fall_class_ids:
                continue
        if (box["y2"] - box["y1"]) < floor:
            continue
        kept.append(box)
    return kept


def write_snapshot(cfg: FallAppConfig, track: Track, frame_index: int, frame) -> str:
    """Write the annotated frame that an alert refers to. Returns the path."""
    if frame is None or not cfg.alerts.attach_snapshot:
        return ""
    directory = Path(cfg.alerts.snapshot_dir)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"fall_track{track.track_id:03d}_frame{frame_index:06d}.jpg"
    if not runtime.cv2.imwrite(str(out), frame):
        console.warn(f"failed to write snapshot {out}")
        return ""
    return str(out)


class FallRuntime(TaskRuntime):
    output_label = "detector_output"
    unit = "people"

    def decode(self, pipeline: FallPipeline, cfg: FallAppConfig, sample, index: int):
        # The result field, not the whole bundle -- see DetectRuntime.decode.
        payload, _ = extract_bbox_payload(joined_field(sample, "detections", 1))
        boxes = parse_boxes(payload, pipeline.frame_w, pipeline.frame_h, cfg.max_detections)
        frame = frame_to_bgr(first_tensor(joined_field(sample, "frame", 0)))
        stamp_pts = getattr(sample, "pts_ns", -1)
        # `boxes` and `frame` are copies, so give the decoder its buffer back
        # before anything else. See FrameStamp for why.
        sample = None

        # Track, then judge. Both need a clock, and the source's own timestamps
        # are the honest one: with overflow_policy block the run is slower than
        # realtime, so wall-clock seconds would make every descent look slow.
        # index is 1-based; the elapsed time before the first frame is 0.
        now = (
            stamp_pts / 1e9 if stamp_pts >= 0
            else (index - 1) / float(pipeline.fps or 25)
        )
        self.check_geometry(pipeline, frame)
        people = person_boxes(cfg, pipeline, boxes, pipeline.frame_h)
        tracks = pipeline.tracker.update(people, now)
        fallen_now = (
            update_fall_states(tracks, cfg.fall, pipeline.frame_h, now)
            if cfg.fall.enable else []
        )
        if fallen_now:
            self._raise_alerts(pipeline, cfg, tracks, fallen_now, frame, index, now)
        return frame, tracks, 0.0

    def _raise_alerts(self, pipeline: FallPipeline, cfg: FallAppConfig, tracks: list[Track],
                      fallen_now: list[Track], frame, index: int, now: float) -> None:
        """Log, snapshot and queue an alert for each track that just fell.

        A confirmed fall is rare, so rendering the overlay here -- rather than
        waiting for the sink thread to do it -- costs almost nothing and keeps
        the snapshot attached to the alert that refers to it.
        """
        snapshot_frame = (
            self.render(cfg, pipeline, frame, tracks, float(pipeline.fps or 25))
            if cfg.alerts.attach_snapshot
            else None
        )
        for track in fallen_now:
            pipeline.falls += 1
            signals = fall_signals(track, cfg.fall, pipeline.frame_h)
            snapshot = write_snapshot(cfg, track, index, snapshot_frame)
            class_id = int(track.box["class_id"])
            label = (
                pipeline.labels[class_id]
                if 0 <= class_id < len(pipeline.labels) else "person"
            )
            console.report(
                f"[FALL] track #{track.track_id} ({label}) at frame {index} "
                f"aspect={signals['aspect_value']} descent={signals['descent_value']}px/s"
                + (f" snapshot={snapshot}" if snapshot else "")
            )
            queued = pipeline.alerts.offer(
                Alert(
                    track_id=track.track_id, label=label,
                    box=(int(track.box["x1"]), int(track.box["y1"]),
                         int(track.box["x2"]), int(track.box["y2"])),
                    signals=signals, frame_index=index, snapshot_path=snapshot,
                ),
                now,
            )
            track.alerted_at = now if queued else track.alerted_at

    def render(self, cfg: FallAppConfig, pipeline: FallPipeline, frame, results, fps: float):
        """Draw once per frame and share the result between the video and JPEG sinks."""
        annotated = frame.copy()
        draw_tracks(annotated, results, cfg.draw, pipeline.labels)
        down = [t for t in results if t.state == FALLEN]
        if cfg.draw.banner and down:
            ids = ", ".join(f"#{t.track_id}" for t in down)
            draw_banner(annotated, f"FALL DETECTED - track {ids}", cfg.draw)
        # FPS last, after the banner as well as the tracks. See
        # DetectRuntime.render.
        if cfg.video_hud:
            draw_fps(annotated, fps, cfg.draw)
        return annotated

    def summarise(self, pipeline: FallPipeline, processed: int) -> list[str]:
        return [f"falls={pipeline.falls}"]


# ─────────────────────────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────────────────────────

# Ids and scores off: a fall frame is read at a glance, and `#3 person 0.87`
# is two numbers in front of the one word that matters. Both are still
# config flags for anyone tuning the tracker.
# The same overlay `detect` draws, plus the alert banner. Scores are on
# because a fall box is an ordinary detection box whose class happens to be
# FALL, and `FALL` alone in the place a score usually sits reads as a different
# kind of readout. Track ids stay off: they are this app's bookkeeping, not
# something a detection carries.
FALL_DRAW = DrawConfig(box_thickness=3, centre_dot=True, banner=True,
                       show_track_ids=False, show_scores=True)


class FallTask(Task):
    name = "fall"
    help = "Track people and email when one of them goes down"
    config_class = FallAppConfig
    graph_name = "yolo_detector"
    result_label = "detections"
    output_label = "detector_output"
    defaults = TaskDefaults(
        task="fall",
        family="yolo26",
        save_dir="frames",
        video_path="falls.mp4",
        draw=FALL_DRAW,
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--classes", dest="tracking.classes", nargs="+", metavar="CLASS",
            help="Class names or ids that can fall. Default: person.",
        )
        parser.add_argument(
            "--confirm", dest="fall.confirm_seconds", type=float, metavar="S",
            help="How long a fall signal must hold before an alert fires. Default 1.5.",
        )
        parser.add_argument(
            "--no-fall", dest="fall.enable", action="store_const", const=False,
            help="Track people without judging falls, which is how you tune tracking first.",
        )
        parser.add_argument(
            "--alert-to", dest="alerts.to", nargs="+", metavar="EMAIL",
            help="Recipients for the fall alert. Implies --alerts.",
        )
        parser.add_argument(
            "--alert-from", dest="alerts.from", metavar="EMAIL",
            help="From address for the fall alert.",
        )
        parser.add_argument(
            "--alerts", dest="alerts.enable", action="store_const", const=True,
            help="Enable alerts. Still a dry run unless --send is given.",
        )
        parser.add_argument(
            "--send", dest="alerts.dry_run", action="store_const", const=False,
            help="Actually connect to the SMTP server. Without it alerts are composed "
                 "and logged but never sent.",
        )
        parser.add_argument(
            "--smtp-host", dest="alerts.smtp.host", metavar="HOST",
            help="SMTP server. Default smtp.gmail.com.",
        )
        parser.add_argument(
            "--smtp-port", dest="alerts.smtp.port", type=int, metavar="PORT",
            help="SMTP port. 587 for STARTTLS, 465 for SSL. Default 587.",
        )
        parser.add_argument(
            "--smtp-user", dest="alerts.smtp.username", metavar="USER",
            help="SMTP login. The password is read from $FALL_ALERT_SMTP_PASSWORD, "
                 "never from the config.",
        )
        parser.add_argument(
            "--site", dest="alerts.site", metavar="NAME",
            help="Human name for this camera, put in the alert subject and body.",
        )
        parser.add_argument(
            "--test-alert", action="store_true",
            help="Send one fake alert and exit. Proves the SMTP settings and the "
                 "password environment variable work, without waiting for a fall "
                 "-- or for a board.",
        )

    def link_overrides(self, overrides: dict) -> dict:
        # Naming a recipient or a sender only makes sense if alerts are on, and
        # silently composing nothing would be the worst of both worlds.
        if overrides.get("alerts.to") or overrides.get("alerts.from"):
            overrides.setdefault("alerts.enable", True)
        # --send is meaningless without alerts on, so let it turn them on too.
        if overrides.get("alerts.dry_run") is False:
            overrides.setdefault("alerts.enable", True)
        return overrides

    def extra_sections(self, raw: dict) -> dict:
        return {
            "track": load_track_config(raw),
            "fall": load_fall_config(raw),
            "alerts": load_alert_config(raw),
        }

    def validate(self, cfg: FallAppConfig) -> None:
        super().validate(cfg)
        validate_fall(cfg)

    def describe(self, cfg: FallAppConfig) -> list[str]:
        # Resolving the classes here means a typo in tracking.classes is caught
        # off-board rather than on the DevKit.
        ids = resolve_classes(
            cfg.track.classes, load_labels(cfg.labels_path),
            "tracking.classes", cfg.labels_path,
        )
        lines = [describe_fall(cfg), describe_alerts(cfg)]
        if ids is not None:
            lines.append(f"tracked class ids: {sorted(ids)}")
        return lines

    def make_pipeline(self, cfg: FallAppConfig, labels: list[str]) -> FallPipeline:
        return FallPipeline(
            labels=labels,
            tracker=Tracker(cfg.track),
            alerts=AlertSender(cfg.alerts),
            fall_class_ids=resolve_classes(
                cfg.track.classes, labels, "tracking.classes", cfg.labels_path
            ),
        )

    def prepare(self, cfg: FallAppConfig, pipeline: FallPipeline, step) -> None:
        step.detail(describe_fall(cfg))
        step.detail(describe_alerts(cfg))

    def early_exit(self, cfg: FallAppConfig, args) -> int | None:
        """``--test-alert``: send one fake alert now and report what happened.

        Deliberately synchronous and outside the pipeline. This is the command
        you run when mail is not arriving, so it has to surface the real
        exception rather than queue the work and return 0.
        """
        if not getattr(args, "test_alert", False):
            return None
        if not cfg.alerts.enable:
            console.error(
                "alerts.enable is off, so there is nothing to test.\n"
                "Pass --alerts, or --alert-to somebody@example.com."
            )
            return 1
        # enable=False keeps AlertSender from starting its worker thread; this
        # send happens on this thread so a failure is raised, not logged.
        sender = AlertSender(replace(cfg.alerts, enable=False))
        probe = Alert(
            track_id=0, label="person", box=(100, 100, 400, 700),
            signals={"aspect": True, "collapse": False, "descent": False,
                     "aspect_value": 1.35, "descent_value": 0.0},
            frame_index=0,
        )
        console.info(describe_alerts(cfg))
        sender.send_now(probe)
        console.report(
            "test alert composed (dry_run is on, nothing left the board)."
            if cfg.alerts.dry_run else "test alert sent."
        )
        return 0

    def runtime(self, cfg, pipeline) -> TaskRuntime:
        return FallRuntime()

