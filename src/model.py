from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field


TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class NaiveBayesSpamModel:
    spam_messages: int = 0
    ham_messages: int = 0
    spam_word_counts: Counter[str] = field(default_factory=Counter)
    ham_word_counts: Counter[str] = field(default_factory=Counter)
    spam_total_words: int = 0
    ham_total_words: int = 0
    vocabulary: set[str] = field(default_factory=set)

    def fit(self, texts: list[str], labels: list[str]) -> None:
        for text, label in zip(texts, labels):
            tokens = tokenize(text)
            self.vocabulary.update(tokens)

            if label == "spam":
                self.spam_messages += 1
                self.spam_word_counts.update(tokens)
                self.spam_total_words += len(tokens)
            else:
                self.ham_messages += 1
                self.ham_word_counts.update(tokens)
                self.ham_total_words += len(tokens)

    def predict(self, text: str) -> str:
        label, _ = self.predict_with_confidence(text)
        return label

    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        spam_log = self._log_probability(text, "spam")
        ham_log = self._log_probability(text, "ham")
        max_log = max(spam_log, ham_log)
        spam_prob = math.exp(spam_log - max_log)
        ham_prob = math.exp(ham_log - max_log)
        total = spam_prob + ham_prob
        normalized_spam_prob = spam_prob / total if total else 0.5

        label = "spam" if normalized_spam_prob >= 0.5 else "ham"
        confidence = normalized_spam_prob if label == "spam" else 1 - normalized_spam_prob
        return label, round(confidence, 4)

    def _log_probability(self, text: str, label: str) -> float:
        total_messages = self.spam_messages + self.ham_messages
        if total_messages == 0:
            raise ValueError("Model has not been trained.")

        vocab_size = max(len(self.vocabulary), 1)
        tokens = tokenize(text)

        if label == "spam":
            prior = self.spam_messages / total_messages
            word_counts = self.spam_word_counts
            total_words = self.spam_total_words
        else:
            prior = self.ham_messages / total_messages
            word_counts = self.ham_word_counts
            total_words = self.ham_total_words

        score = math.log(prior if prior > 0 else 1e-12)
        denominator = total_words + vocab_size

        for token in tokens:
            token_count = word_counts.get(token, 0)
            score += math.log((token_count + 1) / denominator)

        return score
