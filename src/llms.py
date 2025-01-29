"""Module providing a LangChain Wrapper around various local models which require specialized input schemes."""

# Imports for ShowUI
import ast
from typing import Any, Dict, List, Optional

import torch
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate
from pydantic import Field
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from src.utils import decode_base64_to_pil

SHOWUI_NAV_SYSTEM = """You are an assistant trained to navigate the {_APP} screen. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
{_ACTION_SPACE}
"""

SHOWUI_NAV_FORMAT = """
Format the action as a json string with the following keys:
{{
    "action": "ACTION_TYPE", 
    "value": "element", 
    "position": [x,y]
}}
and for multiple actions return a list of such json strings. 

If value or position is not applicable, set it as `None`.
Position might be [[x1,y1], [x2,y2]] if the action requires a start and end position.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

SHOWUI_ACTIONS = """
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
2. `INPUT`: Type a string into an element, value is a string to type and the position [x,y] is required. 
3. `SELECT`: Select a value for an element, value is not applicable and the position [x,y] is required. 
4. `HOVER`: Hover on an element, value is not applicable and the position [x,y] is required.
5. `ANSWER`: Answer the question, value is the answer and the position is not applicable.
6. `ENTER`: Enter operation, value and position are not applicable.
7. `SCROLL`: Scroll the screen, value is the direction to scroll and the position is not applicable.
8. `SELECT_TEXT`: Select some text content, value is not applicable and position [[x1,y1], [x2,y2]] is the start and end position of the select operation.
9. `COPY`: Copy the text, value is the text to copy and the position is not applicable.
"""

system_prompt = SHOWUI_NAV_SYSTEM.format(
    _APP="web", _ACTION_SPACE=SHOWUI_ACTIONS, _NAV_FORMAT=SHOWUI_NAV_FORMAT
)

SHOWUI_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("user", system_prompt),
        ("user", "Task: {query}"),
        (
            "user",
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,{img}"},
                }
            ],
        ),
    ]
)


class ChatShowUI(BaseChatModel):
    """A custom chat model using ShowUI"""

    model_name: str = Field(default="showlab/ShowUI-2B", alias="model")
    model: Optional[Qwen2VLForConditionalGeneration] = None
    processor_name: str = Field(default="Qwen/Qwen2-VL-2B-Instruct", alias="processor")
    processor: Optional[AutoProcessor] = None
    max_tokens: Optional[int] = 256
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    device: Optional[str] = None  # Declare the device as a field

    # Prompts and related
    split: str = "web"
    system_prompt: str = SHOWUI_NAV_SYSTEM
    format_prompt: str = SHOWUI_NAV_FORMAT
    action_map: Dict = {"web": SHOWUI_ACTIONS}

    def __init__(self, min_pixels=256 * 28 * 28, max_pixels=1344 * 28 * 28, **kwargs):
        super().__init__(**kwargs)  # Initialize Pydantic fields

        # Check for device and assign to the declared field
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            raise ValueError("ChatShowUI should only be run with a GPU...")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained(
            self.processor_name, min_pixels=min_pixels, max_pixels=max_pixels
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response from ShowUI"""

        assert all(
            isinstance(m, HumanMessage) for m in messages
        ), "Messages must all be HumanMessages"

        # convert from langchain format to ShowUI format
        user_input = self._to_chat_prompt(messages)
        text = self.processor.apply_chat_template(
            user_input, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(user_input)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(
                inputs.input_ids, generated_ids
            )  # removes text from the input
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Create a ChatResult object, expected by LangChain
        ai_message = AIMessage(content=output_text)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronous generation for ShowUI."""

        return self._generate(messages=messages)

    # This method is required by the BaseChatModel class.
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "ShowUI-GUI-navigation-model"

    def _to_chatml_format(self, message: BaseMessage) -> Dict:
        """Convert LangChain message object to a ChatML format"""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            raise ValueError(f"Unknown message type: {type(message)}")

        return {"role": role, "content": message.content}

    def _to_chat_prompt(self, messages: List[BaseMessage]) -> List[Dict]:
        """Convert a list of messages into a prompt format appropriate for ShowUI"""
        if not messages:
            raise ValueError("At least one Message must be provided")
        if not all([isinstance(m, HumanMessage) for m in messages]):
            raise ValueError("ShowUI doesn't accept non-Human Messages currently ...")

        contents = []
        for m in messages:
            converted_m = self._to_chatml_format(m)

            if isinstance(converted_m["content"], list):
                # We have decoded image we should convert into  PIL image object
                encoded_img = converted_m["content"][0]["image_url"]["url"].split(",")[
                    -1
                ]  # strips prefix

                # convert encoded image string to PIL object
                image = decode_base64_to_pil(encoded_img)

                converted_content = {"type": "image_url", "image": image}
                converted_m["content"] = converted_content
            elif isinstance(converted_m["content"], str):
                converted_m["content"] = {
                    "type": "text",
                    "text": converted_m["content"],
                }

            contents.append(converted_m["content"])

        return [{"role": "user", "content": contents}]
