import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class LocalLLM:
    MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

        if self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.MODEL_NAME
            )

            self.model.to(self.device)

        self.model.eval()

    def generate(
        self,
        prompt,
        max_new_tokens=512,
        temperature=0.1,
    ):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a strict evidence-grounded technical "
                    "assistant.\n\n"

                    "Your answer must be based ONLY on the supplied "
                    "project evidence.\n\n"

                    "Rules:\n"
                    "1. Never invent implementation details.\n"
                    "2. Never infer behavior that is not explicitly "
                    "supported by the evidence.\n"
                    "3. Do not use outside knowledge.\n"
                    "4. If the evidence does not establish a fact, "
                    "say that the indexed project evidence does not "
                    "confirm it.\n"
                    "5. Every important technical claim must include "
                    "one or more evidence references such as [E1] "
                    "or [E2].\n"
                    "6. Evidence references must correspond to the "
                    "provided evidence blocks.\n"
                    "7. Do not create evidence references that do "
                    "not exist.\n"
                    "8. Distinguish clearly between what the source "
                    "code contains and what it does not establish.\n"
                    "9. Do not use words such as 'likely', "
                    "'probably', 'appears', or 'presumably' to fill "
                    "missing information.\n"
                    "10. If multiple sources support a claim, cite "
                    "all relevant evidence references.\n\n"

                    "Answer concisely and technically."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = output[
            0
        ][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return answer.strip()

    def close(self):
        if hasattr(self, "model"):
            del self.model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()