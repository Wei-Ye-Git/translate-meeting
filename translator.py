import os
import re
import time
from contextvars import ContextVar
from functools import wraps

import torch
from transformers import MarianMTModel, MarianTokenizer
from i18n import data, t
from translation_segmentation import (
    DEFAULT_MAX_TOKENS,
    join_translated_segments,
    split_long_text,
)

_active_terminology = ContextVar("active_terminology", default=None)


def _with_terminology_snapshot(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        active = _active_terminology.get()
        if active is not None:
            return method(self, *args, **kwargs)
        snapshot = self._terminology_snapshot()
        token = _active_terminology.set(snapshot)
        try:
            return method(self, *args, **kwargs)
        finally:
            _active_terminology.reset(token)

    return wrapped


def translation_quality_issues(source, translated):
    issues = []
    if not translated or not translated.strip():
        return ["empty_translation"]
    source_numbers = re.findall(r"(?<!\w)\d+(?:[.,/]\d+)*(?!\w)", source)
    translated_numbers = re.findall(r"(?<!\w)\d+(?:[.,/]\d+)*(?!\w)", translated)
    if any(number not in translated_numbers for number in source_numbers):
        issues.append("missing_numbers")
    source_words = re.findall(r"[A-Za-z]{2,}", source)
    chinese_range = data("translation.chinese_characters", r"\u4e00-\u9fff")
    chinese_chars = len(re.findall(f"[{chinese_range}]", translated))
    latin_words = re.findall(r"[A-Za-z]{2,}", translated)
    if len(source_words) >= 4 and not chinese_chars and len(latin_words) >= len(source_words) // 2:
        issues.append("mostly_untranslated")
    translated_units = len(re.findall(f"[{chinese_range}]|[A-Za-z0-9]+", translated))
    chinese_chars = len(re.findall(f"[{chinese_range}]", translated))
    source_units = len(source_words) + len(source_numbers)
    substantive_source_units = len([
        word for word in source_words
        if word.lower() not in {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for",
            "is", "are", "was", "were", "be", "we", "i", "you", "it", "this",
            "that", "please", "will", "would", "can", "could", "should",
            "before", "after", "by", "with", "from", "as", "at", "do", "does",
            "did", "has", "have", "had", "not", "no", "if", "then", "than",
        }
    ]) + len(source_numbers)
    if (
        source_units >= 5
        and substantive_source_units >= 3
        and (
            translated_units < max(2, source_units // 2)
            or (chinese_chars > 0 and chinese_chars < max(4, substantive_source_units * 1.5))
        )
    ):
        issues.append("possible_content_omission")
    if re.search(r"\b(?:not|no|never|without|neither|nor)\b", source, re.IGNORECASE):
        negation_pattern = "|".join(
            re.escape(marker)
            for marker in data("translation.negation_markers", [])
        )
        if not negation_pattern or not re.search(negation_pattern, translated):
            issues.append("possible_missing_negation")
    return issues


def set_review_reason(metadata, reason):
    """Add one independent reason for manual review to a record."""
    result = dict(metadata or {})
    reasons = list(result.get("review_reasons", []))
    legacy_reason = result.get("review_reason")
    if legacy_reason and legacy_reason not in reasons:
        reasons.append(legacy_reason)
    if reason not in reasons:
        reasons.append(reason)
    result["review_reasons"] = reasons
    result["review"] = True
    return result


def clear_review_reason(metadata, reason):
    """Clear one review reason without removing unrelated review requirements."""
    result = dict(metadata or {})
    reasons = list(result.get("review_reasons", []))
    legacy_reason = result.get("review_reason")
    if legacy_reason and legacy_reason not in reasons:
        reasons.append(legacy_reason)
    reasons = [item for item in reasons if item != reason]
    result.pop("review_reason", None)
    if reasons:
        result["review_reasons"] = reasons
        result["review"] = True
    else:
        result.pop("review_reasons", None)
        if not result.get("translation_review") and not result.get("asr_anomaly"):
            result.pop("review", None)
    return result


def apply_translation_quality_status(metadata, issues):
    """Replace translation review reasons while preserving ASR review state."""
    result = dict(metadata or {})
    has_asr_review = bool(result.get("asr_anomaly") or result.get("asr_suspect"))
    has_asr_reason = any(
        reason.startswith("asr_") for reason in result.get("review_reasons", [])
    )
    reasons = list(result.get("review_reasons", []))
    had_translation_review = result.get("translation_review") or any(
        reason.startswith("translation_") or reason == "translation_retry_failed"
        for reason in reasons
    )
    legacy_reason = result.get("review_reason")
    if legacy_reason and legacy_reason not in reasons:
        reasons.append(legacy_reason)
    elif (
        result.get("review")
        and not reasons
        and not had_translation_review
        and not has_asr_review
    ):
        reasons.append("legacy_review")
    reasons = [
        reason for reason in reasons
        if not (
            reason.startswith("translation_")
            or reason in {"translation_retry_failed", "translation_error"}
        )
    ]
    reasons.extend(
        issue
        if issue in {"translation_retry_failed", "translation_error"}
        else f"translation_{issue}"
        for issue in issues
    )
    result.pop("review_reason", None)
    result.pop("translation_review", None)
    result.pop("translation_issues", None)
    if issues:
        result["translation_review"] = True
        result["translation_issues"] = list(issues)
    if reasons:
        result["review_reasons"] = reasons
        result["review"] = True
    else:
        result.pop("review_reasons", None)
        if had_translation_review and not has_asr_review and not has_asr_reason:
            result.pop("review", None)
        elif has_asr_review:
            result["review"] = True
    return result


def can_use_context_translation(texts, translated_parts):
    if not texts or translated_parts and len(texts) != len(translated_parts):
        return False
    word_counts = [len(text.split()) for text in texts]
    if min(word_counts) < 4:
        return False
    return all(count >= 4 for count in word_counts)


def _safe_print(msg):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="backslashreplace").decode("ascii"))


class Translator:
    def __init__(
        self,
        model_path=None,
        device="cpu",
        terminology=None,
        intra_threads=2,
        inter_threads=1,
    ):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "models", "opus_mt_en_zh")
        if device == "cpu":
            torch.set_num_threads(min(4, max(2, (os.cpu_count() or 4) // 2)))
        self.device = device
        self.terminology = terminology
        self.cache = {}
        self.last_latency_seconds = 0.0
        self.ct2_model = None
        self.model = None

        ct2_dir = None
        if os.path.isfile(os.path.join(model_path, "model.bin")):
            ct2_dir = model_path
        else:
            alt_ct2 = model_path.rstrip("/\\") + "_ct2"
            if os.path.isfile(os.path.join(alt_ct2, "model.bin")):
                ct2_dir = alt_ct2

        if ct2_dir is not None:
            try:
                import ctranslate2
                _safe_print(f"[translate] Loading CTranslate2 English-to-Chinese model: {ct2_dir} (intra_threads={intra_threads}, inter_threads={inter_threads})")
                self.ct2_model = ctranslate2.Translator(
                    ct2_dir,
                    device=device,
                    compute_type="int8",
                    intra_threads=intra_threads,
                    inter_threads=inter_threads,
                )
                tok_dir = ct2_dir if os.path.isfile(os.path.join(ct2_dir, "vocab.json")) else model_path
                self.tokenizer = MarianTokenizer.from_pretrained(tok_dir, local_files_only=True)
            except Exception as error:
                _safe_print(f"[translate] CTranslate2 loading failed; falling back to PyTorch: {error}")
                self.ct2_model = None

        if self.ct2_model is None:
            _safe_print(f"[translate] Loading PyTorch English-to-Chinese model: {model_path}")
            self.tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = MarianMTModel.from_pretrained(model_path, local_files_only=True).to(device)
            self.model.eval()

    @staticmethod
    def quality_issues(source, translated):
        return translation_quality_issues(source, translated)

    @staticmethod
    def should_retranslate(source, translated):
        if not source or not translated:
            return True
        source_words = re.findall(r"[A-Za-z0-9']+", source)
        chinese_range = data("translation.chinese_characters", r"\u4e00-\u9fff")
        translated_words = re.findall(f"[{chinese_range}A-Za-z0-9']+", translated)
        if not translated_words:
            return True
        chinese_chars = len(re.findall(f"[{chinese_range}]", translated))
        english_chars = len(re.findall(r"[A-Za-z]", translated))
        if len(source_words) >= 4 and chinese_chars == 0 and english_chars > 0:
            return True
        translated_units = len(
            re.findall(f"[{chinese_range}]|[A-Za-z0-9']+", translated)
        )
        if translated_units < max(2, len(source_words) // 2):
            return True
        return False

    @staticmethod
    def normalize_output(text):
        result = re.sub(r"\s+", " ", text.strip())
        punctuation = re.escape(data("translation.chinese_punctuation", "，。！？"))
        result = re.sub(rf"([{punctuation}])\1+", r"\1", result)
        result = re.sub(rf"\s+([{punctuation}])", r"\1", result)
        result = re.sub(rf"([{punctuation}])([A-Za-z])", r"\1 \2", result)
        return result.strip()

    def _terminology_snapshot(self):
        active = _active_terminology.get()
        if active is not None:
            return active
        snapshot = getattr(self.terminology, "snapshot", None)
        return snapshot() if callable(snapshot) else self.terminology

    def _apply_terms(self, source, translated, terminology=None):
        if terminology is None:
            terminology = self._terminology_snapshot()
        if terminology is None:
            return translated
        return terminology.apply(source, translated)

    def _token_count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def _translate_long_text(self, text, quality):
        segments = split_long_text(
            text,
            self._token_count,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        translated = self.translate_batch(segments, quality=True if quality else False)
        return join_translated_segments(translated)

    def _generate_ct2(self, text, num_beams=1, max_new_tokens=256):
        encoded = self.tokenizer.encode(text)
        tokens = self.tokenizer.convert_ids_to_tokens(encoded)
        results = self.ct2_model.translate_batch(
            [tokens],
            beam_size=num_beams,
            max_decoding_length=max_new_tokens,
        )
        out_tokens = results[0].hypotheses[0]
        return self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(out_tokens), skip_special_tokens=True
        ).strip()

    def _generate_batch_ct2(self, texts, num_beams=1, max_new_tokens=256):
        token_batches = [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(t))
            for t in texts
        ]
        results = self.ct2_model.translate_batch(
            token_batches,
            beam_size=num_beams,
            max_decoding_length=max_new_tokens,
        )
        return [
            self.tokenizer.decode(
                self.tokenizer.convert_tokens_to_ids(res.hypotheses[0]),
                skip_special_tokens=True,
            ).strip()
            for res in results
        ]

    def _generate(self, inputs, num_beams, max_new_tokens=256):
        output = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, num_beams=num_beams
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()

    def _generate_context(self, text, num_beams=4, max_new_tokens=384):
        if getattr(self, "ct2_model", None) is not None:
            return self._generate_ct2(
                text, num_beams=num_beams, max_new_tokens=max_new_tokens
            )
        inputs = self.tokenizer(
            [text], return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        return self._generate(inputs, num_beams, max_new_tokens)

    @torch.no_grad()
    @_with_terminology_snapshot
    def translate(self, text: str, quality=False) -> str:
        if not text.strip():
            return ""
        text = text.strip()
        source_text = text
        terminology = self._terminology_snapshot()
        glossary_version = getattr(terminology, "version", 0)
        cache_key = (source_text, quality, glossary_version)
        if cache_key in self.cache:
            self.last_latency_seconds = 0.0
            return self.cache[cache_key]
        text = source_text
        started = time.monotonic()
        token_count = self._token_count(text)
        if token_count > 512:
            translated = self._translate_long_text(source_text, quality)
            self.last_latency_seconds = time.monotonic() - started
            return translated
        if getattr(self, "ct2_model", None) is not None:
            translated = self._generate_ct2(text, num_beams=4 if quality else 1)
            issues = self.quality_issues(source_text, translated)
            if issues and quality is False:
                quality_translation = self._generate_ct2(text, num_beams=4)
                quality_issues = self.quality_issues(source_text, quality_translation)
                if len(quality_issues) < len(issues):
                    translated = quality_translation
        else:
            inputs = self.tokenizer(
                [text], return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            translated = self._generate(inputs, num_beams=4 if quality else 1)
            issues = self.quality_issues(source_text, translated)
            if issues and quality is False:
                quality_translation = self._generate(inputs, num_beams=4)
                quality_issues = self.quality_issues(source_text, quality_translation)
                if len(quality_issues) < len(issues):
                    translated = quality_translation
        translated = self.normalize_output(
            self._apply_terms(source_text, translated, terminology)
        )
        if self.should_retranslate(source_text, translated) and quality is False:
            quality_result = self.translate(source_text, quality=True)
            if quality_result and quality_result != translated:
                translated = quality_result
        if len(self.cache) >= 2000:
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = translated
        self.last_latency_seconds = time.monotonic() - started
        return translated

    @torch.no_grad()
    @_with_terminology_snapshot
    def translate_batch(self, texts, quality=True):
        texts = [text.strip() for text in texts if text.strip()]
        if not texts:
            return []
        started = time.monotonic()
        terminology = self._terminology_snapshot()
        glossary_version = getattr(terminology, "version", 0)
        if any(self._token_count(text) > 512 for text in texts):
            short_texts = [text for text in texts if self._token_count(text) <= 512]
            if short_texts:
                if getattr(self, "ct2_model", None) is not None:
                    short_outputs = self._generate_batch_ct2(
                        short_texts, num_beams=4 if quality else 1
                    )
                else:
                    short_inputs = self.tokenizer(
                        short_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    ).to(self.device)
                    short_outputs = self.model.generate(
                        **short_inputs,
                        max_new_tokens=256,
                        num_beams=4 if quality else 1,
                    )
                    short_outputs = [
                        self.tokenizer.decode(item, skip_special_tokens=True).strip()
                        for item in short_outputs
                    ]
                translated_short = iter(
                    self.normalize_output(self._apply_terms(text, translated, terminology))
                    for text, translated in zip(short_texts, short_outputs)
                )
            else:
                translated_short = iter(())
            output = [
                next(translated_short)
                if self._token_count(text) <= 512
                else self._translate_long_text(text, quality)
                for text in texts
            ]
            self.last_latency_seconds = time.monotonic() - started
            return output
        if getattr(self, "ct2_model", None) is not None:
            output_texts = self._generate_batch_ct2(texts, num_beams=4 if quality else 1)
        else:
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self.device)
            output = self.model.generate(
                **inputs, max_new_tokens=256, num_beams=4 if quality else 1
            )
            output_texts = [
                self.tokenizer.decode(item, skip_special_tokens=True).strip()
                for item in output
            ]
        results = []
        for text, item in zip(texts, output_texts):
            translated = self._apply_terms(text, item, terminology)
            translated = self.normalize_output(translated)
            if self.should_retranslate(text, translated) and quality is False:
                translated = self.translate(text, quality=True)
            results.append(translated)
        for text, translated in zip(texts, results):
            while len(self.cache) >= 2000:
                self.cache.pop(next(iter(self.cache)))
            self.cache[(text, quality, glossary_version)] = translated
        self.last_latency_seconds = time.monotonic() - started
        return results

    @_with_terminology_snapshot
    @torch.no_grad()
    def translate_with_context(self, text, context, quality=True):
        text = text.strip()
        context = [item.strip() for item in context if item and item.strip()]
        if not text:
            return ""
        terminology = self._terminology_snapshot()
        if context and len(text.split()) >= 4:
            translated = self.translate_context(context[-2:] + [text])
            if translated:
                final_translation = self._apply_terms(text, translated[-1], terminology)
                if self.quality_issues(text, final_translation) == []:
                    return self.normalize_output(final_translation)
        return self.translate(text, quality=quality)

    @_with_terminology_snapshot
    @torch.no_grad()
    def translate_context(self, texts):
        texts = [text.strip() for text in texts if text.strip()]
        if not texts:
            return []
        started = time.monotonic()
        terminology = self._terminology_snapshot()
        if any(self._token_count(text) > 512 for text in texts):
            results = []
            for text in texts:
                if self._token_count(text) > 512:
                    results.append(self.translate(text, quality=True))
                    continue
                if getattr(self, "ct2_model", None) is not None:
                    translated = self._generate_ct2(text, num_beams=4)
                else:
                    inputs = self.tokenizer(
                        [text], return_tensors="pt", truncation=True, max_length=512
                    ).to(self.device)
                    translated = self._generate(inputs, num_beams=4)
                translated = self.normalize_output(
                    self._apply_terms(text, translated, terminology)
                )
                results.append(translated)
            self.last_latency_seconds = time.monotonic() - started
            return results
        if len(texts) > 1:
            if can_use_context_translation(texts, ["" for _ in texts]):
                source = "\n".join(texts)
                if self._token_count(source) <= 384:
                    translated = self._generate_context(source)
                    parts = [
                        part.strip()
                        for part in re.findall(r".+?[。！？!?](?:\s*|$)|.+$", translated)
                        if part.strip()
                    ]
                    if len(parts) == len(texts):
                        results = []
                        for text, part in zip(texts, parts):
                            part = self.normalize_output(
                                self._apply_terms(text, part, terminology)
                            )
                            results.append(part)
                        if self._context_results_usable(texts, results):
                            self.last_latency_seconds = time.monotonic() - started
                            return results
            return self.translate_batch(texts, quality=True)
        source = "\n".join(texts)
        if self._token_count(source) > 512:
            return self.translate_batch(texts, quality=True)
        translated = self._generate_context(source)
        self.last_latency_seconds = time.monotonic() - started
        parts = [
            part.strip() for part in re.findall(r".+?[。！？!?](?:\s*|$)|.+$", translated)
            if part.strip()
        ]
        if can_use_context_translation(texts, parts):
            results = [
                self._apply_terms(text, self.normalize_output(part), terminology)
                for text, part in zip(texts, parts)
            ]
            if self._context_results_usable(texts, results):
                return results
        return self.translate_batch(texts, quality=True)

    def _context_results_usable(self, texts, translated_parts):
        if not can_use_context_translation(texts, translated_parts):
            return False
        return all(
            not self.quality_issues(text, translated)
            for text, translated in zip(texts, translated_parts)
        )
