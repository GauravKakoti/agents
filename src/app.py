import asyncio
import json
import base64
import io
import os
import sys
import platform
import streamlit as st
import traceback
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from PIL import Image
from playwright.async_api import async_playwright
from pathlib import Path

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(project_root)
sys.path.append(project_root)

from src.agents import WebAgent
from src.omniparser import OmniParser, OmniParserConfig
from src.prompts import LAVAGUE_HEADER_NAV_ONLY

# Load .env file from root directory
load_dotenv(os.path.join(project_root, ".env"))

# Set up the logs as a streamlit state variable
if "structured_logger" not in st.session_state:
    st.session_state.structured_logger = None
    st.session_state.op_config = None

def verify_paths():
    """Verify that required model paths exist"""
    paths = {
        "som_model": os.path.join(
            project_root, "src", "weights", "omniparser", "icon_detect", "best.pt"
        ),
        "caption_model": os.path.join(
            project_root, "src", "weights", "omniparser", "icon_caption_blip2"
        ),
    }

    missing = []
    for name, path in paths.items():
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")

    if missing:
        st.error("Missing required model files:")
        for m in missing:
            st.error(m)
        return False
    return True


# Initialize the agent (do this once)
@st.cache_resource
def init_agent(model, caption_model, device):
    try:
        if caption_model == "java":
            omniparser = None
        else:
            if not verify_paths():
                return None
            # Create explicit config with full paths
            st.session_state.op_config = OmniParserConfig(
                caption_model=caption_model,
                som_model_path=os.path.join(
                    project_root, "src", "weights", "omniparser", "icon_detect", "best.pt"
                ),
                device=device,
                caption_model_path=os.path.join(
                    project_root, "src", "weights", "omniparser", "icon_caption_blip2"
                ),
            )

            omniparser = OmniParser.from_config(st.session_state.op_config)

        # Define the llm
        if model.startswith("gpt"):
            llm = ChatOpenAI(model=model)
        else:
            llm = ChatGroq(model=model)

        return WebAgent(
            image_parser=omniparser,
            llm=llm,
            prompt_kwargs={"header_text": LAVAGUE_HEADER_NAV_ONLY},
            project_root = project_root
        )
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        import traceback
        st.error(f'this is the project root: {project_root}')
        st.error(traceback.format_exc())  # Show full error traceback
        return None


# Streamlit app layout
st.title("Web Agent Interface")

# User inputs
start_url = st.text_input("Enter the start URL", "https://www.bankrate.com/")
question = st.text_input(
    "Enter the question",
    "Please find the best mortgage rate available to me using bankrate.com. My zip code is 90210 and the purchase price is $400,000 with a down payment of $85,000. My credit score is 800.",
)
model = st.selectbox(
    "Underlying model", ("gpt-4o", "gpt-4o-mini", "llama-3.2-90b-vision-preview")
)
caption_model = st.selectbox("caption model", ("llava", "blip2", "java")) 
device = st.selectbox("Device", ("cpu", "mps", "cuda"))
max_steps = st.number_input("Max steps", min_value=1, max_value=150, value=50)
headless = st.checkbox("Run in headless mode", value=True)

with st.container():
    agent_button = st.button("Run Agent")
    file_name = st.text_input(
    "File name for saved examples",
    "new_example.json",
)
    save_button = st.button("Save Run")


# Initialize containers for dynamic content
status_container = st.empty()
screenshot_container = st.empty()
thought_container = st.empty()
result_container = st.empty()
    
# Button to start the agent
if agent_button:
    web_agent = init_agent(model, caption_model, device)
    if web_agent is None:
        st.error("Could not initialize the web agent.")
    else:
        web_agent.structured_logger.clear_log()
        async def run_agent_with_ui():
            async with async_playwright() as playwright:
                # Initialize browser
                status_container.write("Starting browser...")
                browser = await playwright.chromium.launch(headless=headless)
                
                page = await browser.new_page()
                await page.set_viewport_size({"width": 1280, "height": 720})
                try:
                    # Navigate to start URL
                    status_container.write(f"Navigating to {start_url}")
                    await page.goto(start_url)
                    # Initialize agent state
                    state = {
                        "step": 1,
                        "page": page,
                        "objective": question,
                        "scratchpad": [],
                    }
                    # Create event stream
                    event_stream = web_agent.graph.astream(
                        state, {"recursion_limit": max_steps}
                    )

                    # Process events
                    async for event in event_stream:
                        if "agent" not in event:
                            continue

                        state = event["agent"]
                        current_step = state.get("step", 0)

                        # Update UI with current state
                        status_container.write(f"Step {current_step}")

                        # Show current URL
                        url = page.url
                        st.text(f"Current URL: {url}")

                        # Show screenshot if available
                        if state.get("img"):
                            try:
                                # Convert base64 string to image
                                image_bytes = base64.b64decode(state["img"])
                                image = Image.open(io.BytesIO(image_bytes))
                                screenshot_container.image(
                                    image, caption=f"Browser view - Step {current_step}"
                                )
                            except Exception as img_error:
                                st.error(f"Error displaying image: {str(img_error)}")
                        else:
                            screenshot_container.empty()

                        # Show thought process
                        if state.get("prediction", None):
                            thought_container.markdown(
                                f"""
                                ### Current Action
                                **Action:** {state['prediction'].get('action', 'Unknown')}
                                
                                **Arguments:** {state['prediction'].get('args', [])}
                                
                                **Thought:** {state['prediction'].get('thoughts', 'No reasoning provided')}
                                """
                            )
                        else:
                            thought_container.empty()

                        # Check for final answer
                        if(
                            state.get("prediction", {}).get("action", "").lower()
                            == "answer"
                        ):
                            final_answer = state["prediction"]["args"][0]
                            result_container.markdown(
                                "### Final Answer\n" + str(final_answer)
                            )
                            st.session_state.structured_logger = web_agent.structured_logger
                            break

                except Exception as e:
                    status_container.error(f"An error occurred: {str(e)}")
                    screenshot_container.error(traceback.format_exc())
                    st.write("Initialized OmniParser with config:", st.session_state.op_config)  # Debug output
                finally:
                        # Clean up
                        await browser.close()
        # Run the async function
        if sys.platform.startswith('win'):
            loop = asyncio.ProactorEventLoop()
            loop.run_until_complete(run_agent_with_ui())
        else:
            asyncio.run(run_agent_with_ui())

if save_button:
    if st.session_state.structured_logger is None:
        with status_container:
            st.error("Have not run agent yet")
    else:
        st.session_state.structured_logger.save_to_json(Path(project_root + "/Examples/" + file_name))