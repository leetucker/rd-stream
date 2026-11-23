"""How to run R for the comparisons: r_config.toml at the repository root
(not tracked; see r_config.example.toml), overridden by the environment
variables RD_RSCRIPT (a command, split like a shell would) and
RD_R_LIBS; else `Rscript` on the PATH.

    python tests/r_config.py --shell    # RSCRIPT=... and R_LIBS_USER=... for a shell
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "r_config.toml"


def _file():
    if not CONFIG.exists():
        return {}
    try:
        import tomllib
    except ImportError:                  # Python 3.10
        try:
            import tomli as tomllib
        except ImportError:
            return {}
    return tomllib.loads(CONFIG.read_text())


def rscript_command() -> list[str] | None:
    """The words of the command that runs an R script, or None without R."""
    env = os.environ.get("RD_RSCRIPT")
    if env:
        words = shlex.split(env)
    else:
        value = _file().get("rscript") or "Rscript"
        words = [value] if isinstance(value, str) else [str(v) for v in value]
    words = [os.path.expanduser(w) for w in words]
    return words if shutil.which(words[0]) or Path(words[0]).exists() else None


def r_libs() -> str | None:
    value = os.environ.get("RD_R_LIBS") or _file().get("r_libs") or ""
    return os.path.expanduser(value) or None


def r_environment() -> dict:
    """The environment to run R in: this one, with R_LIBS_USER if configured."""
    env = dict(os.environ)
    libs = r_libs()
    if libs:
        env["R_LIBS_USER"] = libs
    return env


if __name__ == "__main__":
    if "--shell" in sys.argv:
        cmd = rscript_command()
        print(f"RSCRIPT={shlex.quote(shlex.join(cmd)) if cmd else ''}")
        print(f"export R_LIBS_USER={shlex.quote(r_libs() or os.environ.get('R_LIBS_USER', ''))}")
    else:
        print({"rscript": rscript_command(), "r_libs": r_libs()})
