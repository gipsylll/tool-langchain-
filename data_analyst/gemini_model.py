"""Адаптер Gemini для сохранения подписей Tool Calling между вызовами."""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


class GeminiChatOpenAI(ChatOpenAI):
    """Сохраняет opaque thought_signature из OpenAI-совместимого ответа Gemini."""

    def _create_chat_result(self, response: Any, generation_info: dict | None = None) -> Any:
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump()
        choices = raw.get("choices") or []
        for generation, choice in zip(result.generations, choices, strict=False):
            message = generation.message
            if not isinstance(message, AIMessage) or not isinstance(choice, dict):
                continue
            response_message = choice.get("message") or {}
            signatures = {}
            for call in response_message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                extra = call.get("extra_content")
                google = extra.get("google") if isinstance(extra, dict) else None
                signature = google.get("thought_signature") if isinstance(google, dict) else None
                if isinstance(call.get("id"), str) and isinstance(signature, str):
                    signatures[call["id"]] = signature
            if signatures:
                message.additional_kwargs["gemini_tool_signatures"] = signatures
        return result

    def _get_request_payload(
        self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        messages = self._convert_input(input_).to_messages()
        for message, payload_message in zip(messages, payload.get("messages", []), strict=False):
            if not isinstance(message, AIMessage):
                continue
            signatures = message.additional_kwargs.get("gemini_tool_signatures", {})
            if not isinstance(signatures, dict):
                continue
            for call in payload_message.get("tool_calls", []):
                signature = signatures.get(call.get("id"))
                if isinstance(signature, str):
                    call["extra_content"] = {"google": {"thought_signature": signature}}
        return payload
