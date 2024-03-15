import importlib
from typing import Optional

# from huggingface.chat_models.base import BaseChatModel
# from huggingface.llms import BaseLLM
# from huggingface.schema.messages import (
    # AIMessage,
    # BaseMessage,
    # ChatMessage,
    # FunctionMessage,
    # HumanMessage,
    # SystemMessage,
# )

from evals.api import CompletionFn, CompletionResult
from evals.prompt.base import CompletionPrompt, is_chat_prompt
from evals.record import record_sampling

# start stub classes

class BaseChatModel: pass


class BaseLLM: pass


class BaseMessage: pass


class AIMessage(BaseMessage):
    def __init__(self, content, additional_kwargs) -> None:
        super().__init__()
        self.content=content
        self.additional_kwargs=additional_kwargs


class ChatMessage(BaseMessage):
    def __init__(self, content, role) -> None:
        self.content = content
        self.role = role

class FunctionMessage(BaseMessage):
    def __init__(self, content, name) -> None:
        self.content = content
        self.name = name


class HumanMessage(BaseMessage):
    def __init__(self, content) -> None:
        self.content = content


class SystemMessage(BaseMessage):
    def __init__(self, content) -> None:
        self.content = content

# end stub classes


class HuggingFaceLLMCompletionResult(CompletionResult):
    def __init__(self, response) -> None:
        self.response = response

    def get_completions(self) -> list[str]:
        return [self.response.strip()]


class HuggingFaceLLMCompletionFn(CompletionFn):
    def __init__(self, llm: str, llm_kwargs: Optional[dict] = None, **kwargs) -> None:
        # Import and resolve self.llm to an instance of llm argument here,
        # assuming it's always a subclass of BaseLLM
        if llm_kwargs is None:
            llm_kwargs = {}
        module = importlib.import_module("huggingface.llms")
        llm_class = getattr(module, llm)

        if issubclass(llm_class, BaseLLM):
            self.llm = llm_class(**llm_kwargs)
        else:
            raise ValueError(f"{llm} is not a subclass of BaseLLM")

    def __call__(self, prompt, **kwargs) -> HuggingFaceLLMCompletionResult:
        prompt = CompletionPrompt(prompt).to_formatted_prompt()
        response = self.llm(prompt)
        record_sampling(prompt=prompt, sampled=response)
        return HuggingFaceLLMCompletionResult(response)


def _convert_dict_to_huggingface_message(_dict) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    
    if role == "assistant":
        content = _dict["content"] or ""  # OpenAI returns None for tool invocations
        if _dict.get("function_call"):
            additional_kwargs = {"function_call": dict(_dict["function_call"])}
        else:
            additional_kwargs = {}
        return AIMessage(content=content, additional_kwargs=additional_kwargs)

    if role == "system":
        return SystemMessage(content=_dict["content"])

    if role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])

    return ChatMessage(content=_dict["content"], role=role)


class HuggingFaceChatModelCompletionFn(CompletionFn):

    def __init__(self, llm: str, chat_model_kwargs: Optional[dict] = None, **kwargs) -> None:
        # Import and resolve self.llm to an instance of llm argument here,
        # assuming it's always a subclass of BaseLLM
        if chat_model_kwargs is None:
            chat_model_kwargs = {}
        module = importlib.import_module("huggingface.chat_models")
        llm_class = getattr(module, llm)

        if issubclass(llm_class, BaseChatModel):
            self.llm = llm_class(**chat_model_kwargs)
        else:
            raise ValueError(f"{llm} is not a subclass of BaseChatModel")

    def __call__(self, prompt, **kwargs) -> HuggingFaceLLMCompletionResult:
        if is_chat_prompt(prompt):
            messages = [_convert_dict_to_huggingface_message(message) for message in prompt]
        else:
            messages = [HumanMessage(content=prompt)]
        response = self.llm(messages).content
        record_sampling(prompt=prompt, sampled=response)
        return HuggingFaceLLMCompletionResult(response)
