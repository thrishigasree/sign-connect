import os
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Part-of-speech and semantic categories for the 21 Sign Language classes
PRONOUNS = {
    "I": {"be": "am", "have": "have", "wear": "wear", "obj": "me", "poss": "my"},
    "YOU": {"be": "are", "have": "have", "wear": "wear", "obj": "you", "poss": "your"},
    "HE": {"be": "is", "have": "has", "wear": "wears", "obj": "him", "poss": "his"},
    "SHE": {"be": "is", "have": "has", "wear": "wears", "obj": "her", "poss": "her"},
    "IT": {"be": "is", "have": "has", "wear": "has", "obj": "it", "poss": "its"},
}

ADJECTIVES = {"ALRIGHT", "GOOD", "RICH", "SAD"}
TIME_ADVERBS = {"TODAY", "MORNING", "AFTERNOON"}
SEASONS = {"MONSOON", "SUMMER", "WINTER"}
CLOTHING_OBJECTS = {"DRESS": "a dress", "T-SHIRT": "a T-shirt", "TELEPHONE": "the telephone"}
INTERJECTIONS = {"HELLO", "PLEASE", "THANK YOU"}


class LocalNLPEngine:
    """
    Advanced Linguistic Gloss-to-Text (G2T) Natural Language Processing Engine.
    
    Transforms sign language gloss sequences into grammatically coherent,
    syntactically polished spoken English sentences using linguistic dependency parsing,
    copula verb synthesis, and semantic role labeling.
    """

    @classmethod
    def process(cls, tokens: List[str]) -> str:
        if not tokens:
            return ""

        words = [w.strip().upper() for w in tokens if w and w.strip()]
        if not words:
            return ""

        # Step 1: Pre-process compound signs & social formulas
        processed = cls._resolve_compounds(words)
        
        # Step 2: Extract sentence components (Interjections, Time, Subject, Predicate, Objects)
        sentence = cls._synthesize_grammar(processed)

        # Step 3: Polish capitalization and punctuation
        return cls._finalize_punctuation(sentence)

    @classmethod
    def _resolve_compounds(cls, words: List[str]) -> List[str]:
        """Resolves multi-word sign compounds like 'GOOD MORNING' or 'THANK YOU'."""
        result = []
        i = 0
        n = len(words)
        while i < n:
            if i + 1 < n and words[i] == "GOOD" and words[i + 1] in ["MORNING", "AFTERNOON"]:
                result.append(f"GOOD_{words[i + 1]}")
                i += 2
            elif i + 1 < n and words[i] == "THANK" and words[i + 1] == "YOU":
                result.append("THANK_YOU")
                i += 2
            else:
                result.append(words[i])
                i += 1
        return result

    @classmethod
    def _synthesize_grammar(cls, words: List[str]) -> str:
        parts: List[str] = []
        i = 0
        n = len(words)

        has_interjection = False
        temporal_prefix = None

        while i < n:
            curr = words[i]

            # 1. Greetings & Courtesy
            if curr == "HELLO":
                parts.append("Hello," if i + 1 < n else "Hello")
                has_interjection = True
                i += 1
                continue
            elif curr == "GOOD_MORNING":
                parts.append("Good morning," if i + 1 < n else "Good morning")
                has_interjection = True
                i += 1
                continue
            elif curr == "GOOD_AFTERNOON":
                parts.append("Good afternoon," if i + 1 < n else "Good afternoon")
                has_interjection = True
                i += 1
                continue
            elif curr in ["THANK_YOU", "THANK YOU"]:
                if i + 1 < n and words[i + 1] == "YOU":
                    parts.append("Thank you very much")
                    i += 2
                else:
                    parts.append("Thank you")
                has_interjection = True
                continue
            elif curr == "PLEASE":
                parts.append("Please")
                i += 1
                continue

            # 2. Time Topic (e.g. TODAY)
            if curr == "TODAY" and i == 0 and n > 1:
                temporal_prefix = "Today"
                i += 1
                continue

            # 3. Pronoun + State/Action Synthesizer
            if curr in PRONOUNS:
                # Capitalize only if beginning of sentence and not following a greeting
                if not has_interjection and not temporal_prefix and len(parts) == 0:
                    subj = "I" if curr == "I" else curr.capitalize()
                else:
                    subj = "I" if curr == "I" else curr.lower()

                pronoun_info = PRONOUNS[curr]

                # Check following word
                if i + 1 < n:
                    nxt = words[i + 1]
                    
                    # Pronoun + Adjective -> Subject + Copula + Adjective (e.g., "I am sad", "she is rich")
                    if nxt in ADJECTIVES:
                        parts.append(f"{subj} {pronoun_info['be']} {nxt.lower()}")
                        i += 2
                        continue

                    # Pronoun + Clothing -> Subject + is wearing + Clothing (e.g. "he is wearing a dress")
                    if nxt in CLOTHING_OBJECTS:
                        parts.append(f"{subj} is wearing {CLOTHING_OBJECTS[nxt]}")
                        i += 2
                        continue

                    # Pronoun + Telephone -> "I will call on the telephone"
                    if nxt == "TELEPHONE":
                        parts.append(f"{subj} will call on the telephone")
                        i += 2
                        continue

                    # Pronoun + Season/Weather -> "it is winter"
                    if nxt in SEASONS:
                        if curr == "IT":
                            parts.append(f"it is {nxt.lower()}")
                        else:
                            parts.append(f"{subj} is in {nxt.lower()}")
                        i += 2
                        continue

                parts.append(subj)
                i += 1
                continue

            # 4. Standalone Adjectives
            if curr in ADJECTIVES:
                parts.append(curr.lower())
                i += 1
                continue

            # 5. Seasons / Weather
            if curr in SEASONS:
                parts.append(f"the {curr.lower()} season")
                i += 1
                continue

            # 6. Clothing / Entities
            if curr in CLOTHING_OBJECTS:
                parts.append(CLOTHING_OBJECTS[curr])
                i += 1
                continue

            # Fallback
            parts.append(curr.capitalize() if i == 0 else curr.lower())
            i += 1

        # Combine with temporal prefix if present
        joined = " ".join(parts)
        if temporal_prefix:
            if joined:
                if joined.lower().startswith("is ") or joined.lower().startswith("the "):
                    joined = f"{temporal_prefix} is {joined}"
                else:
                    joined = f"{temporal_prefix}, {joined}"
            else:
                joined = temporal_prefix

        return joined

    @classmethod
    def _finalize_punctuation(cls, text: str) -> str:
        text = text.strip()
        if not text:
            return ""

        # Clean multiple spaces and comma spacing
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+,', ',', text)

        # Capitalize first character
        text = text[0].upper() + text[1:]

        # Appropriate closing punctuation
        if not text.endswith((".", "!", "?")):
            lower = text.lower()
            if any(k in lower for k in ["hello", "thank you", "good morning", "good afternoon"]):
                text += "!"
            elif any(lower.startswith(q) for q in ["are you", "is it", "who", "what", "where"]):
                text += "?"
            else:
                text += "."

        return text


class SentenceService:
    """
    Multimodal Sentence Prediction Service combining:
    1. Generative AI Model (Groq LLaMA-3.1 / LLaMA-3.3) for context-rich natural language generation & translation.
    2. Local Computational NLP Engine for offline, instantaneous syntactic G2T reconstruction.
    3. Raw sign gloss concatenation.
    """

    def __init__(self, default_api_key: Optional[str] = None):
        self.default_api_key = default_api_key or os.environ.get("GROQ_API_KEY")

    def _get_groq_client(self, api_key: Optional[str] = None):
        key = api_key or self.default_api_key or os.environ.get("GROQ_API_KEY")
        if not key or key == "YOUR_GROQ_API_KEY" or len(key.strip()) < 10:
            return None
        try:
            from groq import Groq
            return Groq(api_key=key.strip())
        except Exception as e:
            logger.warning(f"Could not initialize Groq client: {e}")
            return None

    def build_raw(self, words: List[str]) -> str:
        cleaned = [w.strip().upper() for w in words if w and w.strip()]
        return " ".join(cleaned)

    def build_local_nlp(self, words: List[str]) -> str:
        """Runs the Local Computational NLP Gloss-to-Text Engine."""
        return LocalNLPEngine.process(words)

    def generate_ai_model(
        self,
        words: List[str],
        target_language: str = "English",
        custom_api_key: Optional[str] = None
    ) -> Optional[str]:
        """
        Uses an advanced Generative AI model (LLaMA-3.1-8B on Groq) to predict
        a fluent, contextually natural sentence from sign language keywords.
        """
        client = self._get_groq_client(custom_api_key)
        if not client:
            return None

        prompt_words = " ".join([w.strip().upper() for w in words if w and w.strip()])
        if not prompt_words:
            return ""

        try:
            if target_language.lower() == "english":
                system_prompt = (
                    "You are an expert Sign Language Translator AI. "
                    "Your task is to convert a sequence of recognized Sign Language Gloss keywords "
                    "into a single, highly natural, grammatically correct spoken English sentence. "
                    "Rules:\n"
                    "- Do NOT add conversational filler or explanations.\n"
                    "- Return ONLY the final predicted sentence.\n"
                    "- Correctly infer subject-verb agreement, tense, and auxiliary verbs.\n"
                    "- Example: 'TODAY I SAD' -> 'Today, I am sad.'\n"
                    "- Example: 'HELLO YOU ALRIGHT' -> 'Hello! Are you alright?'\n"
                    "- Example: 'GOOD MORNING YOU PLEASE TELEPHONE' -> 'Good morning! Please call on the telephone.'"
                )
            else:
                system_prompt = (
                    f"You are an expert Sign Language Translator AI. "
                    f"Convert the following Sign Language keywords into a natural, meaningful, grammatically correct sentence in {target_language}. "
                    f"Return ONLY the translated sentence in {target_language}. Do NOT include English translation or notes."
                )

            chat = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Sign Keywords: {prompt_words}"}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.15,
                max_tokens=64
            )

            response = chat.choices[0].message.content.strip()
            # Clean possible markdown or quote wrapping
            response = response.strip('"\'')
            return response

        except Exception as e:
            logger.error(f"AI Model prediction error: {e}")
            return None

    def build_sentence(
        self,
        words: List[str],
        mode: str = "nlp",
        target_language: str = "English",
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Dispatches sentence prediction across selected engines:
          - 'ai_model' (or 'groq'): Uses Generative AI LLM (LLaMA 3.1) with fallback to Local NLP
          - 'nlp' (or 'local_formatted'): Uses Local Computational NLP G2T Engine
          - 'raw': Direct token concatenation
        """
        if not words:
            return {
                "sentence": "",
                "mode": mode,
                "method": "empty",
                "words_count": 0,
                "language": target_language
            }

        # 1. Raw mode
        if mode == "raw":
            return {
                "sentence": self.build_raw(words),
                "mode": "raw",
                "method": "raw_concatenation",
                "words_count": len(words),
                "language": target_language
            }

        # 2. Generative AI Model mode
        if mode in ["ai_model", "groq"]:
            ai_sentence = self.generate_ai_model(words, target_language, custom_api_key=api_key)
            if ai_sentence:
                return {
                    "sentence": ai_sentence,
                    "mode": "ai_model",
                    "method": "llama_3.1_generative_ai",
                    "words_count": len(words),
                    "language": target_language
                }
            # Fallback to local NLP engine if AI key not configured or network error
            local_sentence = self.build_local_nlp(words)
            return {
                "sentence": local_sentence,
                "mode": "ai_model",
                "method": "local_nlp_fallback",
                "words_count": len(words),
                "language": "English",
                "notice": "AI model requires a valid Groq API key. Using local NLP engine."
            }

        # 3. Default: Local Computational NLP Engine
        local_sentence = self.build_local_nlp(words)
        return {
            "sentence": local_sentence,
            "mode": "nlp",
            "method": "computational_nlp_engine",
            "words_count": len(words),
            "language": "English"
        }
