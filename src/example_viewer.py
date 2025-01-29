import asyncio
import json
import base64
import io
import os
import sys

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from PIL import Image
from playwright.async_api import async_playwright

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.agents import WebAgent
from src.omniparser import OmniParser, OmniParserConfig
from src.prompts import LAVAGUE_HEADER_NAV_ONLY

# Load .env file from root directory
load_dotenv(os.path.join(project_root, ".env"))
    
def update_file():
    ''' Loads in the file at Examples/new_file '''
    with open(os.path.join(st.session_state.examples_dir,
                           st.session_state["examples_file"]), "r") as f:
        st.session_state.examples = json.load(f)
    # Clear out the bad examples
    for i in range(len(st.session_state.examples) - 1, -1, -1):
        example = st.session_state.examples[i]
        if (example.get("screenshot", None) is None
            or example.get("action", None) is None):
                st.session_state.examples.pop(i)  # could be more efficient:
    st.session_state.example_no = 0
    st.session_state.changes_made = False

def update_example():
    ''' If you change the action, action args, thought, or objective, we should save that '''
    example = st.session_state.examples[st.session_state.example_no]
    example["action"] = st.session_state["action"]
    args = [st.session_state["arg1"], st.session_state["arg2"]]
    if not args[1]:
        args.pop()
    if not args[0]:
        args.pop()
    example["action_args"] = args
    example["thoughts"] = st.session_state["thoughts"]
    example["objective"] = st.session_state["objective"]
    st.session_state.changes_made = True

def delete_example(confirmed):
    if not confirmed:
        def state_toggle():
            st.session_state.remove_confirm = True
        return state_toggle
    else:
        def true_delete():
            st.session_state.examples.pop(st.session_state.example_no)
            for example_no in range(st.session_state.example_no, len(st.session_state.examples)):
                st.session_state.examples[example_no]["step"] -= 1
            if st.session_state.example_no >= len(st.session_state.examples):
                st.session_state.example_no = len(st.session_state.examples) - 1
            st.session_state.changes_made = True
        return true_delete

def save_changes(confirmed):
    if not confirmed:
        def state_toggle():
            st.session_state.save_confirm = True
        return state_toggle
    else:
        def true_save():
            with open(os.path.join(st.session_state.examples_dir,
                                   st.session_state["examples_file"]), "w") as f:
                json.dump(st.session_state.examples, f, indent=4)
            st.session_state.changes_made = False
        return true_save

# Set up the logs directory as a streamlit state variable before it gets used
if "examples_dir" not in st.session_state:
    st.session_state.examples_dir = os.path.join(project_root, "Examples")
    st.session_state.examples = []
    st.session_state.example_no = 0
    st.session_state.remove_confirm = False
    st.session_state.save_confirm = False
    st.session_state.changes_made = False
    st.session_state["examples_file"] = os.listdir(st.session_state.examples_dir)[0]
    update_file()

# Streamlit app layout
st.title("Example Verification Interface")
st.selectbox(
    "examples_file", os.listdir(st.session_state.examples_dir),
    on_change = update_file,
    key="examples_file"
)

col1, col2, col3 = st.columns([1, 20, 1])

def left_button():
    st.session_state.example_no = max(0, st.session_state.example_no - 1)
def right_button():
    st.session_state.example_no = min(len(st.session_state.examples) - 1,
                                      st.session_state.example_no + 1)
with col1:
    left_button = st.button("<",
        on_click=left_button)
with col3:
    right_button = st.button("\\>",
        on_click=right_button)

# Now to go through and update all the example-related containers appropriately
if st.session_state.examples:
    # Full description of the event
    with col2:
        
        # The delete button changes depending on status
        if st.session_state.changes_made:
            if not st.session_state.save_confirm:
                st.button("Save changes made", on_click=save_changes(False))
            else:
                remove_col1, remove_col2, remove_col3 = st.columns([3, 1, 1])
                with remove_col1:
                    st.markdown("### Confirm saving changes\nthis will overwrite data")
                with remove_col2:
                    st.button("no")
                with remove_col3:
                    st.button("yes", on_click=save_changes(True))

        example = st.session_state.examples[st.session_state.example_no]
        st.markdown(f"""### Step {example.get("step")} """)
        st.image(
            Image.open(io.BytesIO(base64.b64decode(example["screenshot"])))
            # caption = f""" Action: {example.get("action")}   {example.get("action_args")}"""
        )
        # The action is a little special, since it should update with the event
        # and be able to change the event
        sub_col_1, sub_col_2 = st.columns(2)
        with sub_col_1:
            st.text_input(
                "Action",
                example["action"],
                on_change=update_example,
                key="action"
            )
        with sub_col_2:
            arg_def = example.get("action_args", None)
            st.text_input(
                "Arg 1",
                arg_def[0] if arg_def is not None and len(arg_def) > 0 else "",
                on_change=update_example,
                key="arg1"
            )
            st.text_input(
                "Arg 2",
                arg_def[1] if arg_def is not None and len(arg_def) > 1 else "",
                on_change=update_example,
                key="arg2"
            )
        # st.markdown(f"""### Thoughts:\n{example["thoughts"].replace("$", "\\$")}""")
        st.text_area(
            "Thoughts",
            example.get("thoughts", [""]),
            on_change=update_example,
            key="thoughts"
        )
        # st.markdown(f"""### Objective:\n{example["objective"].replace("$", "\\$")}""")
        st.text_area(
            "Objective",
            example.get("objective", [""]),
            on_change=update_example,
            key="objective"
        )

        # The delete button changes depending on status
        if not st.session_state.remove_confirm:
            st.button("Delete Example", on_click=delete_example(False))
        else:
            remove_col1, remove_col2, remove_col3 = st.columns([3, 1, 1])
            with remove_col1:
                st.markdown("### Confirm deletion:")
            with remove_col2:
                st.button("no")
            with remove_col3:
                st.button("yes", on_click=delete_example(True))
        
    st.session_state.remove_confirm = False
    st.session_state.save_confirm = False
else:
    with col2:
        step_container = st.markdown("""# No data""")

    



