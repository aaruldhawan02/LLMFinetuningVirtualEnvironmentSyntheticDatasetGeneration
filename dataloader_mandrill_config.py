# This is the configuration file when you run a model (taken from HuggingFace/github) on Mandrill. If your data-
# set does not follow the specefied format written in dataclass you can create a new custom dataclass and 
# change the parameters based on your dataset. You would also need to add the custom dataclass name in NAME2CLS.

import jsonlines
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
from preprocess.chat import llama_tokenizer, llama_dialog2tokens
from preprocess.prompts import SYSTEM_PROMPT
from llama.generation import Dialog

@dataclass(frozen=True, kw_only=True)
class Chat(ABC):
    id: Optional[str] = None
    system_prompt: Optional[str] = None
    dataset_name: Optional[str] = None
    system_prompt: Optional[str] = SYSTEM_PROMPT

    @abstractmethod
    def to_llama_prompt(self) -> Dialog:
        pass
    
    @abstractmethod
    def to_llama_target(self) -> Dialog:
        '''
        returns: singleton list of format 
        [ {'role': 'assistant', 'content': <content>} ]
        '''
        pass
    
    def to_llama_dialog(self) -> Dialog:
        # return singleton list containing final assistant response
        return self.to_llama_prompt() + self.to_llama_target()
    
    def to_llama_input_with_labels(self):
        prompt_tokens = llama_dialog2tokens(self.to_llama_prompt())
        response_tokens = llama_tokenizer(
            f"{self.to_llama_target()[0]['content']} {llama_tokenizer.eos_token}"
        )['input_ids']
        input_ids = prompt_tokens + response_tokens
        attention_mask = [1] * len(input_ids)
        labels = [-100]*len(prompt_tokens) + response_tokens
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    

@dataclass(frozen=True, kw_only=True)
class Alpaca(Chat):
    instruction: str
    input: str
    output: Optional[str]
    text: str
    
    def to_llama_prompt(self) -> Dialog:
        return [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': self.instruction + self.input},
        ]
    
    def to_llama_target(self) -> Dialog:
        return [
            {'role': 'assistant', 'content': self.output}
        ]
    
@dataclass(frozen=True, kw_only=True)
class OpenChat(Chat):
    user: str
    assistant: str
    formatted: str
    
    def to_llama_prompt(self, system_prompt=SYSTEM_PROMPT) -> Dialog:
        return [
            {'role': 'system', 'content': self.system_prompt or system_prompt},
            {'role': 'user', 'content': self.user},
        ]

    def to_llama_target(self) -> Dialog:
        return [
            {'role': 'assistant', 'content': self.assistant}
        ]
    
@dataclass(frozen=True, kw_only=True)
class PromptResponse(Chat):
    prompt: str
    response: str
    
    def to_llama_prompt(self, system_prompt=SYSTEM_PROMPT) -> Dialog:
        return [
            {'role': 'system', 'content': self.system_prompt or system_prompt},
            {'role': 'user', 'content': self.prompt},
        ]

    def to_llama_target(self) -> Dialog:
        return [
            {'role': 'assistant', 'content': self.response}
        ]
            
@dataclass(frozen=True, kw_only=True)
class InputOutput(Chat):
    instruction: str
    input: str
    output: str
    
    def to_llama_prompt(self, system_prompt=SYSTEM_PROMPT) -> Dialog:
        return [
            {'role': 'system', 'content': self.system_prompt or system_prompt},
            {'role': 'user', 'content': self.input},
        ]

    def to_llama_target(self) -> Dialog:
        return [
            {'role': 'assistant', 'content': self.output}
        ]
@dataclass(frozen=True, kw_only=True)
class QuestionResponse(Chat):
    question: str
    response: str
    
    def to_llama_prompt(self, system_prompt=SYSTEM_PROMPT) -> Dialog:
        return [
            {'role': 'system', 'content': self.system_prompt or system_prompt},
            {'role': 'user', 'content': self.question},
        ]

    def to_llama_target(self) -> Dialog:
        return [
            {'role': 'assistant', 'content': self.response}
        ]
    
@dataclass(frozen=True, kw_only=True)
class QuestionAnswerMovies(Chat):
    question: str
    answer: str
    
    def to_llama_prompt(self, system_prompt=SYSTEM_PROMPT) -> Dialog:
        return [
            {'role': 'system', 'content': self.system_prompt or system_prompt},
            {'role': 'user', 'content': self.question},
        ]

    def to_llama_target(self) -> Dialog:
        return [
            {'role': 'assistant', 'content': self.answer}
        ]
@dataclass(frozen=True, kw_only=True)
class QuestionAnswerYahoo(Chat):
    id: str
    question: str
    answer: str
    nbestanswers: list
    main_category: str
    
    def to_llama_prompt(self, system_prompt=SYSTEM_PROMPT) -> Dialog:
        return [
            {'role': 'system', 'content': self.system_prompt or system_prompt},
            {'role': 'user', 'content': self.question},
        ]

    def to_llama_target(self) -> Dialog:
        return [
            {'role': 'assistant', 'content': self.answer}
        ]
@dataclass(frozen=True, kw_only=True)
class QuestionAnswerKAIST_COT(Chat):
    source: str
    target: str
    rationale: str
    task: str
    type: str
    
    def to_llama_prompt(self, system_prompt=SYSTEM_PROMPT) -> Dialog:
        return [
            {'role': 'system', 'content': self.system_prompt or system_prompt},
            {'role': 'user', 'content': self.source},
        ]

    def to_llama_target(self) -> Dialog:
        return [
            {'role': 'assistant', 'content': self.target + "." + self.rationale}
        ]
@dataclass(frozen=True, kw_only=True)
class Platypus(Chat):
    input: str
    output: str
    instruction: str
    data_source: str
    
    def to_llama_prompt(self, system_prompt=SYSTEM_PROMPT) -> Dialog:
        return [
            {'role': 'system', 'content': self.system_prompt or system_prompt},
            {'role': 'user', 'content': self.instruction},
        ]

    def to_llama_target(self) -> Dialog:
        return [
            {'role': 'assistant', 'content': self.output}
        ]
    
@dataclass(frozen=True, kw_only=True)
class TinySeries(Chat):
    prompt: str
    main_topic: str
    subtopic: str
    adjective: str
    action_verb: str
    scenario: str
    target_audience: str
    programming_language: str
    common_sense_topic: str
    idx: str
    response: str
    
    def to_llama_prompt(self, system_prompt=SYSTEM_PROMPT) -> Dialog:
        return [
            {'role': 'system', 'content': self.system_prompt or system_prompt},
            {'role': 'user', 'content': self.prompt},
        ]

    def to_llama_target(self) -> Dialog:
        return [
            {'role': 'assistant', 'content': self.response}
        ]
    
@dataclass(frozen=True, kw_only=True)
class Llama2Turn:
    instruction: Dialog
    response: Dialog
    dataset_name: Optional[str]
    
NAME2CLS = {
    'prompt-response': PromptResponse,
    'question-response': QuestionResponse,
    'tinyseries': TinySeries,
    'platypus': Platypus,
    'alpaca': Alpaca,
    'openchat': OpenChat,
    'input-out': InputOutput,
    'q-a':QuestionAnswerMovies,
    'q-a-yah':QuestionAnswerYahoo,
    'kaist':QuestionAnswerKAIST_COT
}