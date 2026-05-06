from __future__ import annotations

import time

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test

WAR_AND_PEACE_PASSAGE = (
    "Oh, don't speak to me of Austria. Perhaps I don't understand things, but Austria "
    "never has wished, and does not wish, for war. She is betraying us! Russia alone must "
    "save Europe. Our gracious sovereign recognizes his high vocation and will be true to "
    "it. That is the one thing I have faith in! Our good and wonderful sovereign has to "
    "perform the noblest role on earth, and he is so virtuous and noble that God will not "
    "forsake him. He will fulfill his vocation and crush the hydra of revolution, which has "
    "become more terrible than ever in the person of this murderer and villain! We alone "
    "must avenge the blood of the just one. Whom, I ask you, can we rely on? England with "
    "her commercial spirit will not and cannot understand the Emperor Alexander's loftiness "
    "of soul. She has refused to evacuate Malta. She wanted to find, and still seeks, some "
    "secret motive in our actions. What answer did Novosiltsev get? None. The English have "
    "not understood and cannot understand the self-abnegation of our Emperor who wants "
    "nothing for himself, but only desires the good of mankind. And what have they promised? "
    "Nothing! And what little they have promised they will not perform! Prussia has always "
    "declared that Buonaparte is invincible, and that all Europe is powerless before him. "
    "And I don't believe a word that Hardenburg says, or Haugwitz either. This famous "
    "Prussian neutrality is just a trap. I have faith only in God and the lofty destiny of "
    "our adored monarch. He will save Europe!"
)

LONG_INPUT_KEYWORDS = ("russia", "austria", "europe", "prussia", "england", "france")


@register_test("long_input")
class LongInputTest(AbstractValidationTest):
    """Validates that the model can comprehend and reason about long inputs."""

    def test_id(self) -> str:
        return "long_input"

    def test_name(self) -> str:
        return "Long Input Comprehension"

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        checks: list[CheckResult] = []
        start = time.time()

        prompt = WAR_AND_PEACE_PASSAGE + " What country is this passage primarily discussing?"

        try:
            with self._check_scope(engine, "long_input"):
                response = engine.chat(
                    [{"role": "user", "content": prompt}],
                    max_tokens=256,
                )
        except Exception as e:
            checks.append(CheckResult(name="long_input_error", passed=False, detail=str(e)))
            return TestResult(
                test_id=self.test_id(),
                test_name=self.test_name(),
                engine_id=engine.engine_id(),
                model=model,
                checks=checks,
                elapsed_seconds=time.time() - start,
            )

        content = response.get("content", "") or ""
        checks.append(
            CheckResult(
                name="long_input_has_content",
                passed=bool(content),
                expected="non-empty content",
                actual=content[:200],
            )
        )

        lower = content.lower()
        found = [kw for kw in LONG_INPUT_KEYWORDS if kw in lower]
        checks.append(
            CheckResult(
                name="long_input_relevant_keyword",
                passed=len(found) > 0,
                expected=f"at least one of {LONG_INPUT_KEYWORDS}",
                actual=f"found: {found}" if found else content[:200],
            )
        )

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )
