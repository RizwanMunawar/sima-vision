"""The README's commands have to be real ones.

A README that drifts from the CLI is worse than no README: every example in it
reads as a promise. These tests take the promises literally. Every
`sima-vision ...` line is fed to the actual parser, and every flag mentioned in
a table has to exist.

The README is now the whole manual, so it also carries commands belonging to
other tools: `wsl`, `docker`, `apt`, `ffmpeg`, `sima-cli`. Those are listed in
FOREIGN_FLAGS one by one rather than waved through by a pattern, and the list is
checked for entries that no longer appear, so it cannot quietly grow into a hole
big enough to hide a real mistake in.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from sima_vision.api import _alias_table
from sima_vision.cli import build_parser
from sima_vision.tasks import TASKS

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"

#: Fragments that stand for something rather than being runnable as written.
PLACEHOLDERS = ("<", ">", "...", "$EDITOR", "EMAIL", "PATH", "NAME", "DIR", "|")

#: Flags in the README that belong to other programs. Every one of these must
#: still appear somewhere, so a section being rewritten does not leave a stale
#: exemption behind.
FOREIGN_FLAGS = {
    "--install": "wsl --install -d Ubuntu",
    "--workspace": "sima-cli sdk setup --workspace",
}

#: Commands the README has to mention by name. The list is short because the
#: command surface is: the three tasks, plus `push` and `pull`. Setup folded
#: into the run, and `watch` and `remote` were removed with it.
COMMANDS = [*TASKS, "push", "pull"]


def command_lines(text: str) -> list[str]:
    """Every `sima-vision ...` invocation in fenced bash blocks, joined and cleaned."""
    found = []
    for body in re.findall(r"```bash\n(.*?)```", text, re.S):
        # Re-join shell line continuations before splitting.
        body = body.replace("\\\n", " ")
        for raw in body.splitlines():
            line = raw.split("#", 1)[0].strip()
            if line.startswith("sima-vision "):
                found.append(line)
    return found


#: The README is deliberately short -- basic usage, not a manual -- so this is
#: a floor on "it still shows real commands", not a target to grow towards.
MINIMUM_EXAMPLES = 8


def test_the_readme_actually_has_examples():
    assert len(command_lines(README.read_text(encoding="utf-8"))) > MINIMUM_EXAMPLES


def test_every_documented_command_parses():
    parser = build_parser()
    checked = 0
    for line in command_lines(README.read_text(encoding="utf-8")):
        argv = shlex.split(line)[1:]
        if any(token.startswith(p) or p in token for token in argv for p in PLACEHOLDERS):
            continue                      # a stand-in, not a real invocation
        if not argv:
            continue
        try:
            parser.parse_args(argv)
        except SystemExit as exc:          # argparse exits on a bad flag
            raise AssertionError(f"`{line}` is not a valid command") from exc
        checked += 1
    assert checked > MINIMUM_EXAMPLES, f"only {checked} commands were actually checked"


def flags_in(text: str) -> set[str]:
    """Every `--flag` in a code span or a fenced block.

    Only those: a badge URL like `sima--vision-3775A9` contains something that
    looks like a flag but is not one.
    """
    spans = re.findall(r"`([^`\n]+)`", text)
    blocks = re.findall(r"```[a-z]*\n(.*?)```", text, re.S)
    return {
        flag
        for chunk in spans + blocks
        for flag in re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]+)", chunk)
    }


def known_flags() -> set[str]:
    """Every option string the CLI accepts, from every subcommand."""
    flags = {"--help", "--version"}

    def collect(parser):
        for action in parser._actions:
            flags.update(action.option_strings)
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):        # a subparsers action
                for sub in choices.values():
                    collect(sub)

    collect(build_parser())
    return flags


def test_the_readme_mentions_no_flag_that_does_not_exist():
    """A table row promising `--video` when the flag is `--video-path` is a bug."""
    documented = flags_in(README.read_text(encoding="utf-8"))
    unknown = documented - known_flags() - set(FOREIGN_FLAGS)
    assert not unknown, (
        f"README documents flags that do not exist: {sorted(unknown)}. "
        "If one belongs to another tool, add it to FOREIGN_FLAGS with the "
        "command it comes from."
    )


def test_no_foreign_flag_exemption_is_stale():
    """An exemption for a flag the README no longer uses hides the next mistake."""
    documented = flags_in(README.read_text(encoding="utf-8"))
    unused = sorted(set(FOREIGN_FLAGS) - documented)
    assert not unused, f"FOREIGN_FLAGS exempts flags the README does not use: {unused}"


def test_the_readme_python_keywords_exist():
    """Every `keyword=` shown in a Python block must be a real setting."""
    text = README.read_text(encoding="utf-8")
    aliases: set[str] = set()
    for task_cls in TASKS.values():
        table, _ = _alias_table(task_cls())
        aliases |= set(table)
    # Arguments of the API functions themselves, which are not config settings.
    aliases |= {"out", "size", "config", "use_config_file", "task"}
    # argparse's own keywords, from the plugin example in "Adding your own app".
    # That block is defining a setting, not using one.
    aliases |= {"dest", "metavar", "action", "default", "type", "nargs", "help"}

    used = set()
    for body in re.findall(r"```python\n(.*?)```", text, re.S):
        used |= set(re.findall(r"[( ,]([a-z_]+)=", body))
    unknown = used - aliases
    assert not unknown, f"README uses Python keywords that do not exist: {sorted(unknown)}"


def test_every_task_and_command_is_documented():
    text = README.read_text(encoding="utf-8")
    for name in COMMANDS:
        assert f"sima-vision {name}" in text, f"{name} is not in the README"


def test_every_environment_variable_is_documented():
    """A variable the code reads but the README never names is unfindable.

    Discovered rather than listed here. Naming two of them by hand is how the
    Environment table came to be missing four of them: nothing broke, the table
    just quietly stopped being the list it says it is.
    """
    from sima_vision import assets, bootstrap, console, devkit

    declared = {
        value
        for module in (assets, bootstrap, console, devkit)
        for name, value in vars(module).items()
        if name.endswith("_ENV") and isinstance(value, str)
    }
    declared.add("FALL_ALERT_SMTP_PASSWORD")

    text = README.read_text(encoding="utf-8")
    missing = sorted(name for name in declared if name not in text)
    assert not missing, f"the README never names: {missing}"


def test_internal_links_resolve():
    text = README.read_text(encoding="utf-8")
    broken = [
        target
        for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text)
        if not target.startswith(("http", "mailto:"))
        and not (README.parent / target).resolve().exists()
    ]
    assert not broken, f"broken links {broken}"


def test_every_image_exists():
    """`<img src>` is not markdown link syntax, so the check above never saw it.

    Six images were deleted from the repo while the README still pointed at all
    six, and every test passed. GitHub renders that as a row of broken-image
    icons at the top of the page, which is the first thing anyone sees.
    """
    text = README.read_text(encoding="utf-8")
    sources = re.findall(r'<img[^>]+src="([^"]+)"', text)
    assert sources, "the header logo at least should be there"
    missing = [
        src
        for src in sources
        if not src.startswith("http") and not (README.parent / src).resolve().is_file()
    ]
    assert not missing, f"README shows images that are not in the repo: {missing}"


def test_no_asset_is_left_unused():
    """An image nothing shows is dead weight in every clone of the repo."""
    text = README.read_text(encoding="utf-8")
    unused = [
        path.relative_to(REPO).as_posix()
        for path in sorted((REPO / "assets").rglob("*"))
        if path.is_file() and path.relative_to(REPO).as_posix() not in text
    ]
    assert not unused, f"assets/ holds files the README never shows: {unused}"


def test_every_section_link_resolves():
    """A `#contents` row pointing at a heading that was renamed is a dead end."""
    text = README.read_text(encoding="utf-8")
    headings = {
        re.sub(r"[^a-z0-9 -]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
        for line in text.splitlines()
        if line.startswith("#")
    }
    anchors = set(re.findall(r"\]\(#([a-z0-9-]+)\)", text))
    assert not anchors - headings, f"links to missing sections: {sorted(anchors - headings)}"


def test_the_readme_is_the_only_guide():
    """docs/ was merged into the README. Nothing may point back at it."""
    text = README.read_text(encoding="utf-8")
    assert not (REPO / "docs").exists(), "docs/ is back; the README is meant to be it"
    assert "docs/" not in text


def test_no_references_to_the_old_layout():
    """The per-app folders and `src/app.py` are gone; nothing may still point at them."""
    text = README.read_text(encoding="utf-8")
    stale = [
        token for token in
        ("object-detection/", "instance-segmentation/", "fall-detection/",
         "src/app.py", "--validate-config", "requirements.txt", "app.py")
        if token in text
    ]
    assert not stale, f"stale references {stale}"


def test_no_em_dashes():
    """Plain ASCII punctuation only, so the README reads the same everywhere."""
    text = README.read_text(encoding="utf-8")
    offenders = {
        f"U+{ord(ch):04X} {ch!r}"
        for ch in text
        if ch in "—–‒―"
    }
    assert not offenders, f"README contains dashes that should be ASCII: {offenders}"


def test_the_version_is_written_in_exactly_one_place():
    """Releasing is triggered by `__version__` changing, so nothing may shadow it.

    pyproject.toml declares the version dynamic and points hatchling at
    `sima_vision/__init__.py`. A literal `version = "..."` creeping back into
    pyproject would build a wheel labelled with whichever of the two was not
    bumped, and that number is permanent once it reaches PyPI.
    """
    import re as _re

    from sima_vision import __version__

    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dynamic = ["version"]' in pyproject
    assert 'path = "sima_vision/__init__.py"' in pyproject
    assert not _re.search(r'(?m)^version\s*=', pyproject), (
        "pyproject.toml has a literal version again; it must stay dynamic"
    )
    # The shape the release workflow greps for, not just any assignment.
    init = (REPO / "sima_vision" / "__init__.py").read_text(encoding="utf-8")
    found = _re.findall(r'(?m)^__version__ = "(.*)"$', init)
    assert found == [__version__], f"expected one __version__ line, got {found}"
    assert _re.fullmatch(r"\d+\.\d+\.\d+([ab]\d+|rc\d+)?", __version__), __version__


def test_the_flags_table_lists_every_flag():
    """The README promises "every flag", so it has to be every flag.

    It was three tables before, one shared and one per app, and the shared one
    held seven of thirty-two. A reader had no way to know which was the case.
    """
    subparsers = [
        action.choices
        for action in build_parser()._actions
        if isinstance(getattr(action, "choices", None), dict)
    ][0]
    real = {
        option
        for task in TASKS
        for action in subparsers[task]._actions
        if action.dest != "help"
        for option in action.option_strings
        if option.startswith("--")
    }

    # Sliced to the next heading, not a named one: the sections have been
    # reordered twice, and a hard-coded end anchor silently swallowed whichever
    # section had moved in between, along with every flag it mentioned.
    text = README.read_text(encoding="utf-8")
    start = text.index("## Apps arguments")
    end = text.index(chr(10) + "## ", start + 1)
    table = text[start:end]
    documented = set(re.findall(r"`(--[a-z][a-z0-9-]*)", table))

    missing = sorted(real - documented)
    assert not missing, f"the flags table is missing: {missing}"
    invented = sorted(documented - real)
    assert not invented, f"the flags table lists flags that do not exist: {invented}"
