"""Optional LLM paraphrasing of template questions. Paraphrases are accepted only if they keep every
content word of the original (so the evidence still supports them); otherwise the template text stays."""

from __future__ import annotations

from video_dataset.llm.base import LLMClient
from video_dataset.schemas.qa import QARecord
from video_dataset.utils.logging import get_logger
from video_dataset.utils.text import containment, content_words
from video_dataset.vision.parsing import extract_json

log = get_logger("questions.paraphrase")


class Paraphraser:
    def __init__(self, client: LLMClient, min_containment: float = 0.8):
        self.client = client
        self.min_containment = min_containment

    def paraphrase(self, records: list[QARecord], batch: int = 20) -> int:
        changed = 0
        for i in range(0, len(records), batch):
            chunk = records[i : i + batch]
            listing = "\n".join(f"{k}. {r.question}" for k, r in enumerate(chunk))
            prompt = (
                "Rewrite each question below in natural, varied English without changing its meaning, its timestamps, "
                "or any named entity/action. Keep questions self-contained.\n\n" + listing +
                '\n\nReturn JSON: {"questions": [{"index": <int>, "question": "<rewritten>"}]}'
            )
            try:
                resp = self.client.complete(prompt, max_tokens=2000, temperature=0.4)
                data = extract_json(resp.text) or {}
            except Exception as exc:
                log.warning("paraphrase batch failed: %s", exc)
                continue
            for item in data.get("questions", []):
                try:
                    idx = int(item["index"])
                    new_q = str(item["question"]).strip()
                except (KeyError, TypeError, ValueError):
                    continue
                if not (0 <= idx < len(chunk)) or not new_q:
                    continue
                orig = chunk[idx]
                if containment(content_words(orig.question), content_words(new_q)) >= self.min_containment:
                    orig.question = new_q
                    orig.generator = f"{orig.generator}+paraphrase:{self.client.model}"
                    changed += 1
        return changed
