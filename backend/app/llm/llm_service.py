"""LLM provider orchestration: Groq → Gemini → Ollama → safe unavailable."""

from typing import Any, Optional

from app.core.config import get_settings
from app.llm.gemini_service import GeminiService
from app.llm.groq_service import GroqService
from app.llm.ollama_service import OllamaService
from app.utils.logging import get_logger

logger = get_logger(__name__)

UNAVAILABLE_MESSAGE = "AI Assistant is temporarily unavailable."


class LLMService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.providers = {
            "groq": GroqService(),
            "gemini": GeminiService(),
            "ollama": OllamaService(),
        }

    def _ordered_providers(self) -> list[str]:
        order = [
            self.settings.llm_provider,
            self.settings.llm_fallback,
            self.settings.llm_offline,
        ]
        seen: set[str] = set()
        result: list[str] = []
        for name in order:
            key = (name or "").strip().lower()
            if key and key not in seen and key in self.providers:
                seen.add(key)
                result.append(key)
        return result

    def generate(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        for name in self._ordered_providers():
            provider = self.providers[name]
            if not provider.is_available():
                logger.info("LLM provider unavailable: %s", name)
                continue
            try:
                result = provider.generate(prompt, **kwargs)
                if result.get("status") == "ok":
                    result["provider"] = name
                    return result
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM provider %s failed: %s", name, exc)
                continue
        return {
            "status": "unavailable",
            "message": UNAVAILABLE_MESSAGE,
            "provider": None,
            "content": None,
        }

    def generate_assistant_reply(
        self,
        user_message: str,
        assessment_context: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Generate a grounded reply; never invent triage/providers/findings."""
        from app.ai.rag_service import RAGService

        rag = RAGService()
        rag_result = rag.retrieve(user_message, top_k=2)
        citations = rag_result.get("chunks", [])

        context_lines = []
        if assessment_context:
            context_lines.append(f"Assessment status: {assessment_context.get('status')}")
            context_lines.append(
                f"Triage Pathway (decided ONLY by rules engine): {assessment_context.get('pathway') or 'not determined'}"
            )
            context_lines.append(
                f"Suggested specialty: {assessment_context.get('specialty') or 'not determined'}"
            )
            context_lines.append(f"Inputs provided: {assessment_context.get('input_types')}")
            if assessment_context.get("symptoms"):
                context_lines.append(f"Structured symptoms: {assessment_context.get('symptoms')}")
            if assessment_context.get("report_findings"):
                context_lines.append(f"Report findings: {assessment_context.get('report_findings')}")
            if assessment_context.get("xray"):
                context_lines.append(f"X-ray model result: {assessment_context.get('xray')}")
        else:
            context_lines.append("No prior assessment context available for this user.")

        knowledge_lines = []
        if citations:
            knowledge_lines.append("TRUSTED MEDICAL REFERENCE KNOWLEDGE (RAG):")
            for c in citations:
                knowledge_lines.append(f"- [{c['source']}] {c['topic']}: {c['content']}")

        system_constraints = (
            "You are MediSense AI Assistant, an academic healthcare educational assistant.\n"
            "STRICT GUIDELINES:\n"
            "1. You explain, educate, and summarize based ONLY on provided assessment context and trusted reference guidelines.\n"
            "2. NEVER invent a medical diagnosis or disease certainty.\n"
            "3. NEVER prescribe medicines, change dosages, or recommend specific pharmaceuticals.\n"
            "4. NEVER alter or override the triage pathway.\n"
            "5. If information is missing, state clearly that it is unavailable.\n"
            "6. Always recommend that the patient consult a qualified healthcare provider for clinical evaluation."
        )

        prompt_parts = [system_constraints, "\nUSER ASSESSMENT CONTEXT:"]
        prompt_parts.extend(context_lines)
        if knowledge_lines:
            prompt_parts.append("\n" + "\n".join(knowledge_lines))
        prompt_parts.append(f"\nUSER QUESTION: {user_message}\n")
        prompt_parts.append("ASSISTANT ANSWER:")

        prompt = "\n".join(prompt_parts)

        result = self.generate(prompt)
        if result.get("status") != "ok":
            # Safe grounded response using available context and RAG when offline/unavailable
            fallback_answer = (
                "MediSense AI Assistant is operating in safe educational mode. "
                + (
                    f"Your assessment pathway is currently '{assessment_context.get('pathway')}' with suggested specialty '{assessment_context.get('specialty')}'. "
                    if assessment_context and assessment_context.get("pathway")
                    else "No completed assessment is currently active. "
                )
                + "For medical evaluations or questions regarding symptoms and lab tests, please consult a qualified healthcare professional."
            )
            return {
                "status": "ok",
                "answer": fallback_answer,
                "provider": "grounded_rule_fallback",
                "citations": citations,
                "message": "ok",
                "grounded": True,
            }

        return {
            "status": "ok",
            "answer": result.get("content"),
            "provider": result.get("provider"),
            "citations": citations,
            "message": "ok",
            "grounded": True,
        }
