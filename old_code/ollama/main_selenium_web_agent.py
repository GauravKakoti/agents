from langchain_ollama import OllamaLLM
from langchain.tools import BaseTool
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException, TimeoutException, NoSuchElementException
from pydantic import Field
import time
import re

class SmartWebNavigatorTool(BaseTool):
    name: str = "SmartWebNavigator"
    description: str = "Navigates to a web page and interacts with elements. Use format: navigate_and_click|url|button_text"
    llm: OllamaLLM = Field(default_factory=lambda: OllamaLLM(model="llama3.2"))

    def _run(self, tool_input: str):
        parts = tool_input.split('|')
        if len(parts) != 3 or parts[0] != "navigate_and_click":
            return "Invalid input. Use format: navigate_and_click|url|button_text"
        
        url = parts[1]
        button_text = parts[2]

        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)

        try:
            driver.get(url)
            time.sleep(3)  # Wait for page to load
            print(f"Successfully navigated to {url}")

            print(f"Attempting to click button with text: {button_text}")
            selectors = [
                f"//button[contains(text(), '{button_text}')]",
                f"//a[contains(text(), '{button_text}')]",
                f"//*[contains(@class, 'button') and contains(text(), '{button_text}')]",
                f"//*[contains(@class, 'btn') and contains(text(), '{button_text}')]",
                f"//*[contains(text(), '{button_text}')]"
            ]

            for selector in selectors:
                try:
                    element = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, selector))
                    )
                    element.click()
                    time.sleep(3)  # Wait for any redirection
                    return f"Successfully navigated to {url} and clicked '{button_text}' button. New page title: {driver.title}"
                except:
                    continue

            return f"Successfully navigated to {url}, but could not click '{button_text}' button."

        except WebDriverException as e:
            return f"Error during web navigation: {str(e)}"
        
        finally:
            driver.quit()

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: list[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action}\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> AgentAction | AgentFinish:
        # Check for Final Answer
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Check for Answer (as in your error message)
        if llm_output.startswith("Answer:"):
            return AgentFinish(
                return_values={"output": llm_output.strip()},
                log=llm_output,
            )
        
        # Try to parse different output formats
        action_match = re.search(r"Action:?\s*(.*?)(?:\n|$)", llm_output, re.IGNORECASE | re.DOTALL)
        input_match = re.search(r"(?:Action Input|tool_input)[:=]?\s*(.*?)(?:\n|$)", llm_output, re.IGNORECASE | re.DOTALL)
        
        if action_match and input_match:
            action = action_match.group(1).strip()
            action_input = input_match.group(1).strip()
            
            # Remove quotes if present
            action_input = action_input.strip("'\"")
            
            # If action is SmartWebNavigator, ensure the input is in the correct format
            if "SmartWebNavigator" in action and "|" not in action_input:
                parts = action_input.split()
                if len(parts) >= 2:
                    action_input = f"navigate_and_click|{parts[0]}|{' '.join(parts[1:])}"
            
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)
        
        # If we can't parse the output, return it as a finish action
        return AgentFinish(
            return_values={"output": llm_output.strip()},
            log=llm_output,
        )

    @property
    def _type(self) -> str:
        return "custom_output_parser"

class WebAgent:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2")
        self.navigator_tool = SmartWebNavigatorTool()
        self.tools = [
            Tool(
                name="SmartWebNavigator",
                func=self.navigator_tool._run,
                description="Navigates to a web page and interacts with elements. Use format: navigate_and_click|url|button_text"
            )
        ]
        self.agent_executor = self.create_agent_executor()

    def create_agent_executor(self):
        prompt = CustomPromptTemplate(
            template="""You are a web navigation agent. Your task is to navigate websites and interact with elements as requested.

Available tools:
{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: SmartWebNavigator
Action Input: navigate_and_click|url|button_text
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

If you have completed the task, respond with:
Final Answer: [Your answer here]

Question: {input}
{agent_scratchpad}""",
            tools=self.tools,
            input_variables=["input", "intermediate_steps"]
        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=CustomOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        return AgentExecutor.from_agent_and_tools(agent=agent, tools=self.tools, verbose=True)

    def run(self, prompt: str):
        return self.agent_executor.run(prompt)

# Test the WebAgent with a user prompt
user_prompt = "Please open the website https://en.wikipedia.org/ and click the button 'Log in'."
web_agent = WebAgent()
result = web_agent.run(user_prompt)
print(result)