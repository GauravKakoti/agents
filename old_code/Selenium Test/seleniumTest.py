import re
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, TimeoutException

class SmartWebNavigator:
    @staticmethod
    def execute_code(code: str):
        """Execute dynamically generated Selenium code."""
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)

        try:
            # Set up a safe environment for the code execution
            exec(code, {'driver': driver, 'webdriver': webdriver, 'By': webdriver.common.by.By, 'time': time})
        except Exception as e:
            return f"Error executing the code: {str(e)}"
        finally:
            driver.quit()

class WebAgent:
    LLM_API_URL = "http://127.0.0.1:1234/v1/completions"

    @staticmethod
    def get_llm_response(prompt: str) -> str:
        """Get response from the LLM based on the prompt."""
        formatted_prompt = (
            f"Generate only Python code using Selenium for the following task, without any explanations or comments.\n"
            f"Task: {prompt}\n"
            f"Output only runnable Python code."
        )
        try:
            response = requests.post(
                WebAgent.LLM_API_URL,
                json={"prompt": formatted_prompt}
            )
            response.raise_for_status()
            response_data = response.json()
            llm_text = response_data.get("choices", [{}])[0].get("text", "").strip()
            return llm_text
        except requests.exceptions.RequestException as e:
            print(f"Error calling LLM API: {e}")
            return ""

    @staticmethod
    def extract_code_from_response(response_text: str) -> str:
        """Extract code from the LLM response if it's wrapped in a code block."""
        # Use regex to find the Python code block between ```python and ```
        code_match = re.search(r"```python\n(.*?)```", response_text, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return response_text.strip()  # Fallback to return the whole text if no code block is found

    def run(self, prompt: str) -> str:
        """Process prompt and execute Selenium code based on LLM guidance."""
        llm_response = self.get_llm_response(prompt)
        
        # Extract code from the response
        extracted_code = self.extract_code_from_response(llm_response)
        
        # Validate the response contains essential Selenium code components before execution
        if "webdriver" in extracted_code and "driver.get" in extracted_code:
            result = SmartWebNavigator.execute_code(extracted_code)
        else:
            result = "The LLM did not generate executable Selenium code. Please refine your prompt."
        
        return result

# Test the WebAgent with a user prompt
user_prompt = "Open the website https://en.wikipedia.org/, click on the button 'Log in', and search for the term 'Dogs'."
web_agent = WebAgent()
result = web_agent.run(user_prompt)
print(result)
