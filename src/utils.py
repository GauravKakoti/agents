import ast
import asyncio
import base64
import io
import json
import os
import re
from io import BytesIO
from typing import Union

import numpy as np
import requests
import torch
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain as chain_decorator
from PIL import Image, ImageDraw
from torchvision.ops import box_convert

from src.custom_types import Action, Plan
from src.prompts import (
    LAVAGUE_HEADER,
    LAVAGUE_HEADER_NAV_ONLY,
    LAVAGUE_ORDER,
    WEBNAVIGATOR_HEADER,
    WEBNAVIGATOR_ORDER,
)

# Get the absolute path to the current directory (src)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to `mark_page.js`
js_file_path = os.path.join(current_dir, "mark_page.js")

# some javascript we will run on each step to take a screenshot of the page, select the elements to annotate, and add bounding boxes
with open(js_file_path) as f:
    mark_page_script = f.read()


async def js_tag_labels(image, raw_bboxes, page, output_coord_in_ratio=False, **kwargs):
    """
    Method to generate text on each bbox with javascript-related things,
    specifically a bbox type based on tagName, and text from outerText
    Args:
        image: the image, really only used here to get the dimensions
        page: PlayWright page object, to run javascript
        raw_bboxes: bboxes found by omniparser's get_bboxes
        output_coord_in_ratio: If true, then the bboxes are as a fraction of the
            screen, which won't do, so we'll have to fix. if not for this
            possibility, we wouldn't need the image either
    returns:
        List[bboxes]: A list of bboxes with text- and image-only elements cut
    """
    labels = []
    xywh = np.array(
        [
            [
                bbox["shape"]["x"],
                bbox["shape"]["y"],
                bbox["shape"]["width"],
                bbox["shape"]["height"],
            ]
            for bbox in raw_bboxes
        ]
    )
    if output_coord_in_ratio:
        # get the associated PIL image object to get dimensions of the image
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        else:
            raise ValueError("image_input must be a string or bytes.")

        w, h = image.size
        xywh *= np.array([w, h, w, h])[None, :]

    # Ensure the page content is fully loaded
    await page.wait_for_load_state("domcontentloaded")

    for x, y, w, h in xywh:
        center_x, center_y = x + w / 2, y + h / 2
        # print(f"Evaluating element at: ({center_x}, {center_y})")  # Debugging log

        js_data = await page.evaluate(
            f"""
            (function() {{
                var element = document.elementFromPoint({center_x}, {center_y});
                if (!element) {{
                    return [null, null, null, null];
                }}
                return [
                    element.tagName, 
                    element.onclick, 
                    window.getComputedStyle(element).cursor,
                    element.textContent.trim().replace(/\\s{{2,}}/g, " ")
                ];
            }})()
        """
        )

        if (
            js_data[0] is None
        ):  # Check if the element is null; if it is, add a generic "unknown type" label and skip further processing
            labels.append(
                ["unknown type", ""]
            )  # Append a generic label for unidentifiable elements
            continue  # Skip the rest of the logic for this element

        tag_name = (
            js_data[0].lower() if js_data[0] else "unknown"
        )  # Convert the tag name to lowercase; if there's no tag name, set it to "unknown"

        onclick = js_data[1]
        cursor = js_data[2].lower()
        text = js_data[3].lower()

        if tag_name in ["input", "textarea", "select"]:
            typ = "text input"
        elif (
            tag_name in ["button", "a", "iframe"]
            or onclick is not None
            or cursor == "pointer"
        ):
            typ = "clickable"
        elif tag_name in ["video"]:
            typ = "video"
        elif tag_name in ["p", "h1", "h2", "h3", "h4", "h5", "h6"]:
            typ = "text box"
        elif tag_name in ["img"]:
            typ = "image"
        else:
            typ = "unknown type, tag " + tag_name

        labels.append([typ, (text if text and len(re.split("[><]", text)) < 3 else "")])

    return labels


@chain_decorator
async def mark_page(input):
    page = input["page"]
    parser = input.get("parser", None)
    tag_with_js = input.get("tag_with_js", True)
    # kwargs = {k: v for k, v in input.items() if k not in ["page", "parser"]}
    kwargs = input

    # Wait until the html is loaded, between 0.5 and 5 seconds (arbitrary)
    await asyncio.sleep(0.5)
    await page.wait_for_load_state("domcontentloaded", timeout=4500)
    # Check if parser is provided; if so, use it instead of JavaScript

    # Capture the unmarked screenshot before any annotations are applied
    unmarked_screenshot = await page.screenshot()

    # Check if parser is provided; if so, use it instead of JavaScript
    if parser:
        screenshot = await page.screenshot()

        # Get the box locations, sizes, and potential captions
        raw_bboxes = parser.get_bboxes(screenshot, **kwargs)

        # Generate types from javascript
        if tag_with_js:
            js_labels = await js_tag_labels(screenshot, raw_bboxes, **kwargs)
            for (label, text), bbox in zip(js_labels, raw_bboxes):
                bbox["type"] = label
                if text:
                    bbox["text"] = text

        # Mark up the screenshot
        encoded_image = parser.annotate(
            image_source=screenshot, raw_bboxes=raw_bboxes, **kwargs
        )
        # Formatting and centering coordinates
        formatted_bboxes = [
            {
                "x": bbox["shape"]["x"] + bbox["shape"]["width"] / 2,
                "y": bbox["shape"]["y"] + bbox["shape"]["height"] / 2,
                "width": bbox["shape"]["width"],
                "height": bbox["shape"]["height"],
                "text": bbox.get("text", ""),
                "type": bbox.get("type", ""),
            }
            for bbox in raw_bboxes
        ]

    else:
        # Default to JavaScript-based annotation if no parser is provided
        await page.evaluate(mark_page_script)
        for _ in range(10):
            try:
                formatted_bboxes = await page.evaluate("markPage()")
                break
            except Exception:
                await asyncio.sleep(3)
        screenshot = await page.screenshot()
        await page.evaluate("unmarkPage()")
        encoded_image = base64.b64encode(screenshot).decode("utf-8")
    return {
        "img": encoded_image,
        "bboxes": formatted_bboxes,
        "unannotated_img": base64.b64encode(unmarked_screenshot).decode("utf-8"),
    }


async def annotate(state, parser=None, **kwargs):
    marked_page = await mark_page.with_retry().ainvoke(
        {"page": state["page"], "parser": parser, **kwargs}
    )
    return {**state, **marked_page}


def format_descriptions(state):
    labels = []
    for i, bbox in enumerate(state["bboxes"]):
        text = bbox.get("ariaLabel") or ""
        if not text.strip():
            text = bbox["text"]
        el_type = bbox.get("type")
        if text:
            labels.append(f'{i} ({el_type}): "{text}"')
        else:
            labels.append(f"{i} ({el_type})")
    bbox_descriptions = "\nValid Bounding Boxes:\n" + "\n".join(labels)
    return {**state, "bbox_descriptions": bbox_descriptions}


def parse_formatter_string(text: str) -> dict:
    text = text.replace("json", "").replace(
        "```", ""
    )  # Some models start the json string as '```json' which causes json decoding errors.
    try:
        dictionary = json.loads(text)
    except json.JSONDecodeError:
        # TODO(dominic): probably should log something here ...?
        # TODO(dominic): Is this really what we want to do, or raise?
        print(text)
        dictionary = {
            "thought": "could not find the thought...",
            "tasks": ["No tasks right now... "],
        }

    return Plan(**dictionary)


def parse(text: str) -> dict:
    # set up regex for prefixes for our two points of interest
    thought_prefix = r"(?i)thoughts*:*[\*\- \n]*"
    action_prefix = (
        r"(?i)\n *\**action:*[\*\- \n]*"  # Just combining all prefixes seen so far
    )
    # print("THE TEXT WE ARE PARSING IS", text)  # Uncomment to see raw-ish output from the llm

    # Extract the thought
    text_by_thought = re.split(thought_prefix, text, 1)
    if len(text_by_thought) > 1:
        put_back_in_action = " action "  # In case the thoughts included the word action and we accidentally cut it out
        # The logic below is that we want everything from "Thought:" until the last "Action:", so we split by "Action:" and take every block
        # but the last, but since the word "action" might be in the thoughts, everywhere we split we add back in the word "action", and cut it
        # back off at the end
        thought = "".join(
            phrase + put_back_in_action
            for phrase in re.split(action_prefix, text_by_thought[1])[:-1]
        )[: -len(put_back_in_action)].strip()
    else:
        thought = "cannot find the thought"

    # Extract the action
    text_by_action = re.split(action_prefix, text)
    if len(text_by_action) < 2:
        return {
            "action": "retry",
            "args": f"Could not parse LLM Action in: {text.strip()},\nWe will mark this as the thought",
            "thought": text.strip(),
        }
    action_block = text_by_action[-1].strip()
    split_output = action_block.split(" ", 1)

    # Extract the parameters for the function, and what function to use
    if len(split_output) == 1:
        action, action_input = (
            split_output[0],
            None,
        )  # If there's no parameters, just set the action
    else:
        action, action_input = split_output
    action = action.strip("\"'[]();:*")  # Clean up the action
    if action_input is not None:
        if (
            action == "Type"
        ):  # Special cleaning for type since anything could be in text
            action_input = re.split(
                ";[\t ]*|,[\t ]*|[\t ][\t ]*", action_input.strip(), 1
            )  # Split box id and the rest
            action_input = [
                inp.strip("*\"'[]-") for inp in action_input
            ]  # Clean it up, remove leading or trailing spaces, line ends, *,',", or []
        else:
            action_input = [
                inp.strip("*\"'[]")
                for inp in re.split(
                    ";[\t ]*|,[\t ][\t ]*|[\t ][\t ]*", action_input.strip()
                )  # Cut the parameters string into individual parameters
            ]
        # if len(action_input) == 1 and len(action_input[0].split(" ")) == 2:
        #     action_input = [inp.strip("*\"'[]") for inp in action_input[0].split(" ")]
    return {"action": action, "args": action_input, "thought": thought}


def generate_prompt_template(
    objective="Here is the objective:\n{objective}",
    header_text= LAVAGUE_HEADER_NAV_ONLY,
    few_shot="",
    previous_actions="scratchpad",
    bbox_full_desc="{bbox_descriptions}",
    order=LAVAGUE_ORDER,
    **kwargs,
):
    """
    Generate the prompt from sub-prompts, this still outputs a prompt that can take in
    updated inputs along the way, like bbox_descriptions and img
    inputs:
     - objective: The objective from the user
     - header_text: this is all the instructions and such we include to direct the agent
     - few_shot: a block of text with examples of inputs and outputs for few-shot prompting
     - previous_actions: a list of previous actions. This may require more work to change, if left as default
                         value ("scratchpad"), it uses format_previous_actions in utils via the code
                         MessagesPlaceholder("scratchpad", optional=True)
     - bbox_full_desc: if you want a fancier input than default for the bbox_descriptions, by default it is
                       generated by utils/format_descriptions, which says "valid bounding boxes are:\n" then
                       line separated descriptions
     - order: the order these are entered into the prompt, separated by a blank line
    """
    # First off set up a dictionary with all the prompt elements, so we can call them through order
    entries = kwargs
    entries["objective"] = objective
    entries["header_text"] = header_text
    entries["bbox_full_desc"] = bbox_full_desc
    entries["previous_actions"] = previous_actions
    entries["few_shot"] = few_shot
    # image is a bit special
    entries["image"] = {"url": "data:image/png;base64,{img}"}

    def type(entry):
        if entry == "image":
            return "image_url"
        return "text"

    # Build the prompt
    prompts = []
    for entry in order:
        if entries[entry] == "scratchpad":
            prompts.append(
                MessagesPlaceholder("scratchpad", optional=True)
            )  # It would be good to do similar optional toggles on more of these
            continue
        elif entry == "header_text":  # Comment out this clause to make it more uniform
            prompts.append(
                ("human", entries[entry] + """\n""")
            )  # Ideally this is system, not human, but for groq/llama we need human
            continue
        elif entry != "image":
            # The next two lines are for simple text prompting, if you replace them with the two lines
            # commented out afterward, you get
            # prompts.append(entries[entry] + "\n")
            # continue
            text = entries[entry] + "\n\n"
            entry_type = "text"
        else:
            text = entries[entry]
            entry_type = "image_url"
        prompts.append(("user", [{"type": entry_type, entry_type: text}]))

    prompt_format = ChatPromptTemplate(messages=prompts)
    return prompt_format


def encode_image(image_input: Union[Image.Image, bytes]) -> str:
    """Utility function to take various image formats and convert into a base64 encoded string"""
    if isinstance(image_input, Image.Image):
        buffer = io.BytesIO()
        image_input.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    elif isinstance(image_input, bytes):
        # In this case, image_input is byte string from playwright screenshots
        encoded_image = base64.b64encode(image_input).decode("utf-8")
    else:
        raise ValueError(
            f"image_input should be a PIL Image or bytes, got type {type(image_input)}"
        )

    return encoded_image


def decode_base64_to_pil(encoded_image: str):
    """Decode a base64 encoded string to a PIL object."""
    image_data = base64.b64decode(encoded_image)

    buffer = io.BytesIO(image_data)
    pil_image = Image.open(buffer)
    return pil_image


def parse_llm_output(llm_string):
    # Split the string on dictionary endings to locate all dictionaries
    chunks = llm_string.split("},")

    # Add back in the closing braces for each dictionary string.
    parsed_dicts = []
    for chunk in chunks:
        # Ensure proper formatting
        if not chunk.endswith("}"):
            chunk += "}"
        # Safely evaluate string as a dictionary
        parsed_dicts.append(Action(**ast.literal_eval(chunk)))

    return parsed_dicts


def draw_point(image_input, point=None, radius=5):
    if isinstance(image_input, str):
        image = (
            Image.open(BytesIO(requests.get(image_input).content))
            if image_input.startswith("http")
            else Image.open(image_input)
        )
    elif isinstance(image_input, bytes):
        image = Image.open(BytesIO(image_input)).convert("RGB")
    else:
        image = image_input

    if point:
        x, y = point[0] * image.width, point[1] * image.height
        ImageDraw.Draw(image).ellipse(
            (x - radius, y - radius, x + radius, y + radius), fill="red"
        )
    # display(image)
    return image
