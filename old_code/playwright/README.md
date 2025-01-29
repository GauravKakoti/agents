# Wikipedia Login Detection with Playwright and oLlama

This script uses [Playwright](https://playwright.dev/) to open a Chromium browser and navigate to Wikipedia. It checks for the presence of a "Log in" button by extracting visible text from the page and sending it to [oLlama](https://ollama.com/) for analysis. If oLlama confirms the button’s presence, the script clicks on it.

## Requirements

- Python 3.x
- [Playwright](https://playwright.dev/python/docs/intro) (`pip install playwright`)
- [oLlama](https://ollama.com/) account with API access

## Setup

1. **Install Dependencies**:
   ```bash
   pip install playwright ollama
   playwright install

2. oLlama model: Ensure your oLlama model is configured in your environment.

## Usage
Run the script with:
python main.py

## Code Details
The script navigates to Wikipedia, waits for the "Log in" button, and retrieves page text.
The visible text is sent to oLlama with a prompt to confirm the button’s presence.
If oLlama returns "Yes," the script clicks the "Log in" button.

## Important Notes
Playwright Headless: The script runs with a visible browser window (headless=False) for demonstration purposes.
Error Handling: Ensure oLlama response handling is configured for reliability.

## License
MIT License

This `README.md` provides a quick overview, setup instructions, usage guide, and additional details for running and understanding the script.
