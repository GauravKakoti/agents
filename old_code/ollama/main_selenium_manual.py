from langchain_ollama import OllamaLLM
from langchain.tools import BaseTool
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException
import time

# Initialize the Ollama model
llm = OllamaLLM(model="llama3.2")

# Define a custom tool for intelligent navigation, button detection, HTML retrieval, and screenshot capture
class SmartWebNavigatorTool(BaseTool):
    name: str = "smart_web_navigator"
    description: str = "Navigates to a web page, dynamically identifies a button by its text, clicks it, and retrieves the redirected page's title, full HTML content, and a screenshot."

    def _run(self, url: str, button_text: str):
        return self.navigate_and_analyze(url, button_text)

    def navigate_and_analyze(self, url: str, button_text: str):
        options = Options()
        options.add_argument("--headless")  # Run in headless mode
        driver = webdriver.Chrome(options=options)
        
        try:
            # Step 1: Open the specified URL and capture the HTML content
            driver.get(url)
            time.sleep(3)  # Wait for page to load
            page_html = driver.page_source
            
            # Step 2: Use LLM to analyze HTML and generate a dynamic selector for the button
            model_prompt = (
                f"Identify the most reliable XPath selector specifically in the format `//a[span[contains(text(), '{button_text}')]]` "
                f"or similar, that would locate a button containing the exact text '{button_text}' in this HTML content. "
                f"Only return the XPath in that exact format without additional explanation. HTML snippet:\n\n"
                f"{page_html[:5000]}... (truncated for brevity)"
            )
            selector = llm.invoke(model_prompt).strip()
            
            # Validate if selector looks like an XPath in the specified format
            if not selector.startswith("//a[span") and "contains(text()" not in selector:
                return "Error: The model did not return a valid selector in the required format."
            
            # Step 3: Use the generated selector to locate and click the button
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, selector))
            )
            button = driver.find_element(By.XPATH, selector)
            button.click()
            time.sleep(3)  # Wait for redirection to complete
            
            # Step 4: Capture the title, HTML, and screenshot of the redirected page
            redirected_title = driver.title
            redirected_html = driver.page_source
            print(f"Redirected page title is: {redirected_title}")
            
            # Save HTML content to a file
            with open("redirected_page.html", "w", encoding="utf-8") as file:
                file.write(redirected_html)
            
            # Capture a screenshot of the redirected page
            screenshot_path = "redirected_page_screenshot.png"
            driver.save_screenshot(screenshot_path)
            
            return f"Successfully clicked the button '{button_text}' using selector '{selector}'. Title: {redirected_title}. HTML saved to 'redirected_page.html' and screenshot saved to '{screenshot_path}'."
        
        except TimeoutException:
            return f"Error: The button with text '{button_text}' could not be found on {url} after waiting."
        except WebDriverException as e:
            if "net::ERR_NAME_NOT_RESOLVED" in str(e):
                return f"Error: Unable to resolve the domain name for {url}. The website may not exist."
            return f"An unexpected error occurred: {str(e)}"
        finally:
            driver.quit()

# Function to process the prompt and use the smart tool if a URL and button name are detected
def intelligent_navigation_prompt(prompt: str):
    if "http" in prompt and "button" in prompt:
        # Extract URL and button text from the prompt
        url = prompt.split("http", 1)[1].split()[0]
        url = "http" + url
        button_text = prompt.split("button '")[1].split("'")[0]
        
        print(f"Navigating to URL: {url} and analyzing button with text: '{button_text}'")
        
        # Initialize and run the SmartWebNavigatorTool
        navigator = SmartWebNavigatorTool()
        response = navigator._run(url, button_text)
        print(response)
    else:
        # Default response for general prompts
        response = llm.invoke(prompt)
        print("Model Response:", response)

# Test the function with a prompt containing a URL and button text
user_prompt = "Please open the website https://en.wikipedia.org/ and click the button 'Log in' to get the redirected page title, full HTML content, and a screenshot."
intelligent_navigation_prompt(user_prompt)