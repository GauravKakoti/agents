from playwright.sync_api import sync_playwright
import ollama

def main():
    with sync_playwright() as p:
        # Launch a browser instance
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        
        # Navigate to the Wikipedia page
        page.goto("https://en.wikipedia.org")
        
        # Wait for the 'Log in' button to be visible on the page
        page.wait_for_selector("text=Log in")
        
        # Extract the visible text from the page's body for analysis
        page_content = page.inner_text("body")

        # Send the visible text to oLlama to confirm if it detects the 'Log in' button
        response = ollama.chat(model="llama3.2", messages=[
            {
                "role": "user",
                "content": (
                    f"Based on the following visible text:\n\n{page_content}\n\n"
                    "Please confirm, with and only Yes or Not if you find a button labeled 'Log in' in the top right corner of the page."
                )
            }
        ])
        
        # Capture the response content from oLlama
        instruction = response['message']['content']

        # Click on the 'Log in' button if oLlama confirms its presence
        if "Yes" in instruction:
            page.click("text=Log in")
        else:
            # Close the browser instance
            browser.close()

if __name__ == "__main__":
    main()
