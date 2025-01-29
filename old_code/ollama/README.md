# OLlama Automation Project

This repository contains scripts designed to automate interactions with web pages using Selenium. The main files, `main_selenium_manual.py` and `main_selenium_web_agent.py`, offer distinct approaches for performing automation tasks.

## Project Files

1. **main_selenium_manual.py**: A script that performs manual, customized web automation tasks.
2. **main_selenium_web_agent.py**: A script using a web agent to perform intelligent interactions with webpage elements.

## Features

- Automates web actions such as clicking buttons, entering text, and extracting information.
- Uses Selenium for browser automation.
- Tailored for interactions that require adaptive element selection.

---

### Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/fellowship/web-agent.git
   ```

2. **Install Dependencies:**
   Ensure you have Python installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Selenium WebDriver:**
   Download the WebDriver corresponding to your browser (e.g., ChromeDriver for Chrome) and add it to your system's PATH.

---

### Usage

Each script can be run from the command line. Modify the script parameters as needed to interact with the desired web page.

#### `main_selenium_manual.py`

This script allows for manual control over element selection and action specification. It is useful when precise, static elements on a webpage need interaction.

**Run the script:**
```bash
python main_selenium_manual.py
```

**Expected Output**:
```
Navigating to URL: https://en.wikipedia.org/ and analyzing button with text: 'Log in'
DevTools listening on ws://127.0.0.1:64635/devtools/browser/efa3dac7-5bca-4542-b771-a072d0f653d7
Created TensorFlow Lite XNNPACK delegate for CPU.
Attempting to use a delegate that only supports static-sized tensors with a graph that has dynamic-sized tensors (tensor#58 is a dynamic-sized tensor).
Redirected page title is: Log in - Wikipedia
Successfully clicked the button 'Log in' using selector '//a[span[contains(text(), 'Log in')]]'. Title: Log in - Wikipedia. HTML saved to 'redirected_page.html' and screenshot saved to 'redirected_page_screenshot.png'.
```

#### `main_selenium_web_agent.py`

This script utilizes a more intelligent, dynamic approach, using a language model for enhanced element selection. It’s particularly useful when element names or paths may vary or when the HTML structure is complex.

**Run the script:**
```bash
python main_selenium_web_agent.py
```

**Expected Output**:
```
> Entering new AgentExecutor chain...
Thought: To solve this, I need to navigate to the Wikipedia website and then interact with the login button.
Action: SmartWebNavigator
Action Input: navigate_and_click|https://en.wikipedia.org/|Log in
DevTools listening on ws://127.0.0.1:64880/devtools/browser/a5efe03c-5bfb-44d1-b20a-1de2b0796c70
Successfully navigated to https://en.wikipedia.org/
Attempting to click button with text: Log in
Created TensorFlow Lite XNNPACK delegate for CPU.
Observation:Successfully navigated to https://en.wikipedia.org/ and clicked 'Log in' button. New page title: Log in - Wikipedia
Thought: To solve this, I need to navigate to the login page and then interact with the login form.
Action: SmartWebNavigator
Action Input: navigate_and_click|https://en.wikipedia.org/wiki/Log_in_(user_account)|Log in
DevTools listening on ws://127.0.0.1:64941/devtools/browser/7c176fb7-98e1-4fcc-bc2d-6a7251af3efc
Successfully navigated to https://en.wikipedia.org/wiki/Log_in_(user_account)
Attempting to click button with text: Log in
Created TensorFlow Lite XNNPACK delegate for CPU.
Observation:Successfully navigated to https://en.wikipedia.org/wiki/Log_in_(user_account), but could not click 'Log in' button.
Final Answer: The website https://en.wikipedia.org/ was opened, but the login button couldn't be clicked due to its dynamic nature and potential anti-clicking measures.
> Finished chain.
The website https://en.wikipedia.org/ was opened, but the login button couldn't be clicked due to its dynamic nature and potential anti-clicking measures.
```

---

### Troubleshooting

- **Element Not Found**: Verify that the XPath or CSS selector is correct and accessible. Use `WebDriverWait` and `ExpectedConditions` to handle dynamic loading.
- **Deprecation Warnings**: Update deprecated functions as required in each script.

---

### License

This project is licensed under the MIT License.

---