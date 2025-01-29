import sys
import os
import asyncio
import json

from src import omniparser

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import WebAgent  # Import WebAgent and necessary modules
from tools import click, type_text, scroll, wait, go_back, to_google  # Import tools
from omniparser import OmniParser, OmniParserConfig

# Define tools dictionary
tools = {
    "Click": click,
    "Type": type_text,
    "Scroll": scroll,
    "Wait": wait,
    "GoBack": go_back,
    "Google": to_google,
}

# initialize Omniparser
print(f"Initializing OmniParser...")
omniparser = OmniParser.from_config(OmniParserConfig())

# Initialize WebAgent with tools
web_agent = WebAgent(tools)

async def main(start_url, question, max_steps, headless=True):
    result = await web_agent.run(start_url, question, max_steps, headless)
    return result

if __name__ == "__main__":
    import sys
    start_url = sys.argv[1]
    question = sys.argv[2]
    max_steps = int(sys.argv[3])
    headless = sys.argv[4].lower() == "true"

    # Run the async function and print the result as JSON
    result = asyncio.run(main(start_url, question, max_steps, headless))
    print(json.dumps(result))  # Output result as JSON for Streamlit to capture
