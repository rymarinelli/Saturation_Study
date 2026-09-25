"""
lexicon.py — the single source of truth for every word list the CVE pipeline uses.

  * keyword (regex) relevance filter:  TIER_1_PATTERNS, TIER_2_FRAMEWORKS,
    TIER_2_CONTEXT, DENY_PRODUCTS  ->  keyword_relevance()
  * BM25 query:                         BM25_QUERY_SOURCE_TERMS -> LLM_QUERY_TERMS
  * OWASP Top 10 for LLM Applications (2025) mapping: OWASP_LLM -> categorize_owasp()

The patterns are carried over verbatim from the exploratory notebook
(notebooks/llm_cve_dynamics.ipynb, "Filter rules" and "OWASP mapping" cells).
All rules are hand-written; no LLM is used to label or map any CVE.
"""

from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────────────
# 1. Keyword relevance filter
#    Tier 1 ("adversarial"): explicit LLM / GenAI lexicon and LLM-specific tools.
#    Tier 2 ("supply_chain"): classical ML frameworks, only together with an
#    ML-context word. A deny-list suppresses known common-noun collisions.
# ─────────────────────────────────────────────────────────────────────────────
TIER_1_PATTERNS = [
    r"\blarge language model", r"\bllms?\b", r"\bgenerative ai\b", r"\bgenai\b",
    r"\bchatgpt\b", r"\bopenai\b", r"\banthropic\b",
    r"\bllama[\s-]?(2|3|4|cpp|index|file)", r"\bllama[- ]?guard\b",
    r"\bgpt-?(3\.5|4|4o|5)",
    r"\b(google )?gemini (1|1\.5|2|pro|flash|ultra|nano|model|ai|llm)",
    r"\banthropic claude\b", r"\bclaude (3|3\.5|haiku|sonnet|opus|instant|2|1)",
    r"\bmistral (7b|small|medium|large|nemo)", r"\bmixtral\b",
    r"\bqwen ?\d", r"\bdeepseek\b", r"\bgrok\b", r"\bcohere (command|embed)",
    r"\bprompt injection\b", r"\bjailbreak(ing)?\b",
    r"\bsystem prompt (leak|disclosure|injection)", r"\bprompt leakage\b",
    r"\brag (system|pipeline|application)", r"\bretrieval[- ]augmented\b",
    r"\bhallucinat", r"\bfine[- ]?tun", r"\bhugging\s?face\b", r"\btransformer model",
    r"\blangchain\b", r"\bllamaindex\b", r"\bllama[- ]index\b",
    r"\bvllm\b", r"\bollama\b", r"\bbentoml\b", r"\btriton inference\b",
    r"\bmlflow\b", r"\bkubeflow\b", r"\bautogpt\b",
    r"\bgradio\b", r"\btext[- ]generation[- ]webui\b",
    r"\bvector (db|database|store)\b", r"\bembedding model\b",
    r"\bopen[- ]?webui\b", r"\bcomfyui\b", r"\binvoke[- ]?ai\b",
    r"\banything[- ]?llm\b", r"\bflowise\b", r"\bdify\b",
    r"\bhaystack\b.*\b(llm|ai|nlp)", r"\bnvidia nemo\b", r"\bguardrails (ai|llm)",
    r"\bllamafile\b", r"\btext-generation-inference\b", r"\bopenai[- ]?api\b",
    r"\binference server\b",
    r"\bchromadb\b", r"\bpinecone\b", r"\bweaviate\b", r"\bmilvus\b",
    r"\bqdrant\b", r"\bfaiss\b",
    r"\blunary\b", r"\bgiskard\b", r"\bvanna\b", r"\binstructlab\b",
    r"\bnemo[- ]guardrails\b", r"\bsafetensors\b", r"\bpickle.*model\b",
    r"\bdspy\b", r"\blitellm\b", r"\bcrewai\b",
    r"\bautogen\b.*\b(ai|llm|model)", r"\bmem ?gpt\b",
    r"\bopen[- ]assistant\b", r"\bjan[- ]ai\b", r"\blm studio\b", r"\bkoboldcpp\b",
]
_T1 = re.compile("|".join(TIER_1_PATTERNS), re.I)

TIER_2_FRAMEWORKS = [
    r"\bpytorch\b", r"\btensorflow\b", r"\bkeras\b", r"\bscikit[- ]learn\b",
    r"\bonnx\b", r"\btransformers\b", r"\bdatasets\b",
    r"\bxgboost\b", r"\blightgbm\b", r"\bopenvino\b", r"\bmlx\b",
]
_T2 = re.compile("|".join(TIER_2_FRAMEWORKS), re.I)
TIER_2_CONTEXT = (r"\b(model|\bai\b|ml\b|machine learning|deep learning|neural network|"
                  r"inference|training|llm|nlp|tensor)\b")
_T2_CONTEXT = re.compile(TIER_2_CONTEXT, re.I)

DENY_PRODUCTS = [
    r"resi[- ]gemini", r"gemini[- ]net",
    r"crowdstrike falcon", r"falcon logscale", r"falconpro",
    r"\bphi[- ]node\b",
]
_DENY = re.compile("|".join(DENY_PRODUCTS), re.I)


def keyword_relevance(text: str) -> str:
    """'adversarial' (Tier 1), 'supply_chain' (Tier 2) or '' (not LLM-related).
    `text` is the English description plus the CPE match strings."""
    if _DENY.search(text):
        return ""
    if _T1.search(text):
        return "adversarial"
    if _T2.search(text) and _T2_CONTEXT.search(text):
        return "supply_chain"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# 2. BM25 tokenization and query
#    Multi-word key phrases are folded into single tokens so each phrase gets its
#    own IDF weight. The query is the keyword lexicon above in plain-token form:
#    BM25 therefore RE-WEIGHTS the same vocabulary (graded, rarity-weighted,
#    length-normalized); it is not an independent operationalization.
# ─────────────────────────────────────────────────────────────────────────────
KEY_PHRASES = [
    "large language model", "retrieval augmented", "retrieval-augmented",
    "prompt injection", "prompt leakage", "system prompt", "generative ai",
    "machine learning", "deep learning", "neural network", "hugging face",
    "vector database", "vector store", "embedding model", "transformer model",
    "inference server", "inference engine", "fine tune", "fine-tune", "fine-tuning",
    "open webui", "text generation webui", "lm studio",
]
_PHRASE_SUBS = [
    (re.compile(re.escape(p), re.I), p.replace("-", " ").replace(" ", "_"))
    for p in sorted(KEY_PHRASES, key=len, reverse=True)
]
_TOKEN_RX = re.compile(r"[a-z0-9]+(?:_[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase, fold key phrases to single tokens, split into alphanumerics."""
    if not text:
        return []
    t = text.lower()
    for rx, repl in _PHRASE_SUBS:
        t = rx.sub(repl, t)
    return _TOKEN_RX.findall(t)


BM25_QUERY_SOURCE_TERMS = [
    # LLM / GenAI core
    "large language model", "llm", "llms", "generative ai", "genai",
    "chatgpt", "openai", "anthropic", "claude", "gpt", "gemini",
    "llama", "llama cpp", "llama guard", "mistral", "mixtral", "qwen",
    "deepseek", "grok", "cohere",
    # attacks / LLM concepts
    "prompt injection", "jailbreak", "jailbreaking", "system prompt",
    "prompt leakage", "rag", "retrieval augmented", "hallucination",
    "fine tune", "hugging face", "transformer model", "embedding model",
    # LLM tooling / frameworks
    "langchain", "llamaindex", "vllm", "ollama", "bentoml", "triton",
    "mlflow", "kubeflow", "autogpt", "gradio", "text generation webui",
    "open webui", "comfyui", "invokeai", "anythingllm", "flowise", "dify",
    "haystack", "nemo", "guardrails", "llamafile", "litellm", "dspy",
    "crewai", "autogen", "memgpt", "koboldcpp", "jan ai", "lm studio",
    "inference server", "safetensors",
    # vector databases
    "vector database", "vector store", "chromadb", "pinecone", "weaviate",
    "milvus", "qdrant", "faiss",
    # tier-2 ML frameworks
    "pytorch", "tensorflow", "keras", "scikit learn", "onnx", "transformers",
    "xgboost", "lightgbm", "openvino", "mlx", "machine learning",
    "deep learning", "neural network",
]
# Flattened through the same tokenizer so phrases fold exactly like documents.
LLM_QUERY_TERMS = sorted({tok for term in BM25_QUERY_SOURCE_TERMS for tok in tokenize(term)})
_QUERY_SET = frozenset(LLM_QUERY_TERMS)

# Unambiguously LLM/GenAI terms (used only to triage disagreements in the notebook).
HIGH_SIGNAL_TERMS = frozenset({
    "chatgpt", "gpt", "claude", "gemini", "llama", "mistral", "mixtral", "qwen",
    "deepseek", "grok", "cohere", "openai", "anthropic",
    "llm", "large_language_model", "generative_ai", "genai", "prompt_injection",
    "jailbreak", "prompt_leakage", "system_prompt", "rag", "retrieval_augmented",
    "hallucination",
    "langchain", "llamaindex", "vllm", "ollama", "llamafile", "litellm", "dspy",
    "crewai", "autogpt", "autogen", "memgpt", "koboldcpp", "flowise", "dify",
    "anythingllm", "comfyui", "guardrails", "safetensors",
    "chromadb", "pinecone", "weaviate", "milvus", "qdrant", "faiss",
})


def matched_query_terms(tokens) -> list[str]:
    """Which query terms occur in a document (why BM25 scored it at all)."""
    return sorted(set(tokens) & _QUERY_SET)


def has_high_signal(tokens) -> bool:
    return bool(set(tokens) & HIGH_SIGNAL_TERMS)


# ─────────────────────────────────────────────────────────────────────────────
# 3. OWASP Top 10 for LLM Applications (2025) mapping
#    Description keywords OR CWE class per category; a CVE can hit several.
#    Tier-2 (supply_chain) CVEs with no other hit default to LLM03.
# ─────────────────────────────────────────────────────────────────────────────
OWASP_LLM = {
    "LLM01_Prompt_Injection": (
        r"prompt injection|prompt[- ]inject|jailbreak|indirect (prompt|injection)|instruction injection",
        set(),
    ),
    "LLM02_Sensitive_Info_Disclosure": (
        r"information disclosure|data leak|sensitive (data|information)|\bpii\b|credential (leak|exposure)|secret leak|training data (leak|extraction)|membership inference",
        {"CWE-200", "CWE-209", "CWE-532", "CWE-538", "CWE-540", "CWE-359", "CWE-201"},
    ),
    "LLM03_Supply_Chain": (
        r"supply[- ]chain|malicious (model|package|dependency)|pickle (deserializ|injection)|safetensors|deserializ|model file|backdoor.*model|trojan.*model",
        {"CWE-502", "CWE-94", "CWE-78", "CWE-829", "CWE-915", "CWE-1188", "CWE-1395"},
    ),
    "LLM04_Data_Model_Poisoning": (
        r"data poison|model poison|training data poison|backdoor (attack|in.*model)|fine[- ]?tun.*malic|adversarial (training|example)",
        set(),
    ),
    "LLM05_Improper_Output_Handling": (
        r"output handling|insecure output|unsanitized output|render(ed)? (markdown|html).*model",
        {"CWE-79", "CWE-80", "CWE-918", "CWE-601"},
    ),
    "LLM06_Excessive_Agency": (
        r"excessive (agency|permissions|functionality)|agent.*unauthorized|tool (use|call).*unauthor|function call.*injection|plugin.*unauthor",
        {"CWE-269", "CWE-732", "CWE-285"},
    ),
    "LLM07_System_Prompt_Leakage": (
        r"system prompt (leak|disclosure|exposure|extract)|prompt leakage",
        set(),
    ),
    "LLM08_Vector_Embedding_Weakness": (
        r"vector (db|database|store)|embedding.*(poison|leak|inject)|\brag\b.*(poison|attack|inject)|chroma(db)?|pinecone|weaviate|milvus|qdrant|faiss",
        set(),
    ),
    "LLM09_Misinformation": (
        r"hallucinat|misinformation|package hallucinat",
        set(),
    ),
    "LLM10_Unbounded_Consumption": (
        r"denial of service|\bdos\b|resource exhaustion|unbounded (consum|generation)|infinite loop.*model|token.*flood|context.*overflow",
        {"CWE-400", "CWE-770", "CWE-674", "CWE-789", "CWE-834", "CWE-835"},
    ),
}
OWASP_ORDER = list(OWASP_LLM)
_OWASP_KW = {k: re.compile(p, re.I) for k, (p, _) in OWASP_LLM.items()}
_OWASP_CWE = {k: cwes for k, (_, cwes) in OWASP_LLM.items()}


def categorize_owasp(text: str, cwes, relevance: str) -> list[str]:
    cwes_set = set(cwes)
    hits = set()
    for cat in OWASP_LLM:
        if _OWASP_KW[cat].search(text):
            hits.add(cat)
        if cwes_set & _OWASP_CWE[cat]:
            hits.add(cat)
    if relevance == "supply_chain" and not hits:
        hits.add("LLM03_Supply_Chain")
    return sorted(hits)
