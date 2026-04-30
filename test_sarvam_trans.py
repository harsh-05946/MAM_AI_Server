import argparse
import json
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SARVAM_MODEL_NAME = "sarvamai/sarvam-translate"
DEFAULT_MAX_LENGTH = 1024

SUPPORTED_LANGUAGES = {
    "assamese": "Assamese",
    "bengali": "Bengali",
    "bodo": "Bodo",
    "dogri": "Dogri",
    "gujarati": "Gujarati",
    "english": "English",
    "hindi": "Hindi",
    "kannada": "Kannada",
    "kashmiri": "Kashmiri",
    "konkani": "Konkani",
    "maithili": "Maithili",
    "malayalam": "Malayalam",
    "manipuri": "Manipuri",
    "marathi": "Marathi",
    "nepali": "Nepali",
    "odia": "Odia",
    "punjabi": "Punjabi",
    "sanskrit": "Sanskrit",
    "santali": "Santali",
    "sindhi": "Sindhi",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "urdu": "Urdu",
}


class SarvamTranslator:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.model_name = SARVAM_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        elif self.device == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        if self.verbose:
            print(f"Loading tokenizer: {self.model_name}", file=sys.stderr)
        self.tokenizer: Any = AutoTokenizer.from_pretrained(self.model_name)
        self.model: Any = None

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return

        if self.verbose:
            print(f"Loading model on {self.device}: {self.model_name}", file=sys.stderr)
        model_kwargs: dict[str, Any] = {"torch_dtype": self.dtype}

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs).to(self.device)
        self.model.eval()

    @staticmethod
    def _mostly_latin(text: str) -> bool:
        letters = [ch for ch in text if ch.isalpha()]
        if not letters:
            return False
        latin = sum(1 for ch in letters if "a" <= ch.lower() <= "z")
        return (latin / len(letters)) >= 0.70

    def _run_generation(self, messages: list[dict[str, str]], max_length: int, temperature: float) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=max(temperature, 0.01),
                num_return_sequences=1,
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def translate(
        self,
        text: str,
        target_lang: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        temperature: float = 0.01,
    ) -> str:
        normalized_target = target_lang.lower().strip()
        if normalized_target in SUPPORTED_LANGUAGES:
            resolved_target = SUPPORTED_LANGUAGES[normalized_target]
        elif target_lang in SUPPORTED_LANGUAGES.values():
            resolved_target = target_lang
        else:
            supported = ", ".join(sorted(SUPPORTED_LANGUAGES.values()))
            raise ValueError(f"Unsupported target language '{target_lang}'. Supported: {supported}")

        self._ensure_model_loaded()

        first_pass = [
            {"role": "system", "content": f"Translate the text below to {resolved_target}."},
            {"role": "user", "content": text},
        ]
        output = self._run_generation(first_pass, max_length=max_length, temperature=temperature)

        # Sarvam-only recovery for code-mixed text: if a non-English target still comes out mostly
        # Latin, retranslate the first-pass output into the target language.
        if resolved_target != "English" and self._mostly_latin(output):
            second_pass = [
                {
                    "role": "system",
                    "content": (
                        f"Translate the text below to {resolved_target}. "
                        f"Return only {resolved_target} text, no notes."
                    ),
                },
                {"role": "user", "content": output},
            ]
            output = self._run_generation(second_pass, max_length=max_length, temperature=temperature)

        return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Sarvam-Translate model")
    parser.add_argument("--text", required=True, help="Text to translate")
    parser.add_argument("--target-lang", required=True, help="Target language name")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max generated tokens")
    parser.add_argument("--temperature", type=float, default=0.01, help="Sampling temperature")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--quiet", action="store_true", help="Silence progress logs")
    args = parser.parse_args()

    translator = SarvamTranslator(verbose=not args.quiet)
    started = time.time()
    translated = translator.translate(args.text, args.target_lang, args.max_length, args.temperature)
    elapsed = time.time() - started

    result = {
        "input": args.text,
        "output": translated,
        "target_language": args.target_lang,
        "model": SARVAM_MODEL_NAME,
        "device": translator.device,
        "timing_seconds": round(elapsed, 4),
    }

    if args.json:
        if args.quiet:
            print(json.dumps(result, ensure_ascii=False))
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(f"Input: {args.text}")
    print(f"Target language: {args.target_lang}")
    print(f"Translation: {translated}")
    print(f"Timing: {elapsed:.4f}s")


if __name__ == "__main__":
    main()
