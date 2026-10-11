"""Prompt files (``causetrace/prompts/*.md``): loading, hashing and rendering."""

import hashlib
import string
from dataclasses import dataclass
from importlib import resources

PROMPT_NAMES = ("system", "plan", "check_evidence", "analyze", "verify")


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def render(self, **values: object) -> str:
        """Substitute ``$name`` placeholders; every placeholder must be given."""
        return string.Template(self.text).substitute({k: str(v) for k, v in values.items()})


def load_prompt(name: str) -> Prompt:
    path = resources.files("causetrace").joinpath("prompts", f"{name}.md")
    return Prompt(name=name, text=path.read_text(encoding="utf-8"))


def load_prompts() -> dict[str, Prompt]:
    return {name: load_prompt(name) for name in PROMPT_NAMES}
