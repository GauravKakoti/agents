# Web Interaction Automation with Ollama, Lavague, and Playwright

This project showcases the integration of **Ollama**, **Lavague**, and **Playwright** to automate and enhance web interactions using AI-driven element detection and dynamic action execution. This approach provides a flexible, intelligent method to interact with web pages programmatically, paving the way for developing web agents that can adapt to various scenarios with minimal hardcoding.

## Project Overview

The primary goal is to navigate a webpage, identify and interact with elements like buttons, and retrieve page information. This setup leverages AI models and browser automation frameworks to streamline and sophisticate the interaction, demonstrating potential for a fully autonomous web navigation agent.

### Key Technologies

- **Ollama**: Serves as an LLM (Large Language Model) that interprets web page content to generate and validate dynamic selectors, enhancing adaptability by reading and analyzing HTML content.
- **Lavague**: Acts as a high-level agent capable of executing web actions such as button clicks or navigation, based on step-by-step user instructions.
- **Playwright**: Handles direct browser automation, making it efficient to interact with elements through pre-defined commands and conditions, ensuring stability and control over web elements.

## Usage Examples and Insights

### 1. Intelligent Web Navigation with Lavague

**Lavague** enables controlled, prompt-based web navigation, particularly useful for confirming actions like button clicks and navigation success. This tool is beneficial when step-by-step confirmation is necessary, as it provides feedback on each action’s success, ensuring every interaction is aligned with the desired outcome.

- **Best for**: Scenarios requiring progressive validation and reporting on action success.
- **Ease of Use**: Moderate – requires setting up objectives and monitoring each action for feedback, which may add complexity in multi-step interactions.

### 2. Web Interaction with LangChain and Ollama

Combining **LangChain** with **Ollama** creates a dynamic, adaptable solution for web interaction, where the LLM helps generate element selectors (like XPath) based on content analysis. This tool’s strength is in its adaptability, as it identifies elements through natural language processing, enabling it to respond flexibly to page layout changes.

- **Best for**: Dynamic or content-driven element selection where adaptability is required.
- **Ease of Use**: Moderate to high – once set up, this combination allows for powerful interactions without extensive hardcoding, although it may involve some learning curve with prompt configurations.

### 3. Direct Interaction with Ollama for Custom Navigation

Directly prompting **Ollama** for HTML analysis and element selection allows precise control and adaptability without relying on an overarching framework. This approach is best for simpler, targeted navigation tasks with immediate feedback on interactions, particularly useful for capturing and validating page data.

- **Best for**: Single-action interactions with immediate validation (e.g., capturing HTML or screenshots post-interaction).
- **Ease of Use**: High – straightforward function calls allow for quick interaction without complex setup.

### 4. Combining Playwright and Ollama for Validation and Action

Integrating **Playwright** with **Ollama** offers robust control over browser actions with the added intelligence of conditional interaction. Playwright’s flexibility makes it easy to perform actions based on real-time feedback from the LLM, like verifying the presence of a button before clicking it.

- **Best for**: Scenarios where actions depend on real-time validation (e.g., confirming element presence).
- **Ease of Use**: High – Playwright’s structured commands and Ollama’s validation combine for a seamless setup with straightforward actions.

## Best Results and Ease of Use

Out of the tested approaches, the **Playwright + Ollama** combination yielded the best balance between effectiveness and ease of use. Playwright’s direct handling of browser tasks, enhanced by Ollama’s validation, allows for conditional actions with high reliability and adaptability. Lavague, while powerful, may be better suited for cases where task-by-task feedback is essential, though it can require more detailed setup.

## Next Steps: Towards a Web Agent using an LLM

To advance towards developing a fully autonomous web agent with an LLM, consider the following steps:

1. **Dynamic Goal Setting**: Implement dynamic prompts that allow the LLM to assess the page structure and set interaction goals based on the page’s content, enabling adaptability to new or changing layouts.

2. **Enhanced Element Detection**: Use the LLM to interpret page content and user context, making element identification more flexible. For example, the LLM could be trained to identify elements based on synonyms or contextual clues, enhancing robustness across different pages.

3. **Reinforcement Learning for Task Optimization**: Integrate reinforcement learning to allow the agent to learn from past interactions, refining its approach to navigation, element selection, and error handling over time.

4. **Error Recovery Mechanisms**: Implement error-handling strategies so the agent can retry, backtrack, or adapt its approach when encountering issues (e.g., elements not found, navigation timeouts).

5. **Data Storage and Interaction Logging**: For building an agent that improves over time, it’s helpful to log each action, interaction result, and error encountered. This data will serve to train the agent for future improvements, enabling it to recognize patterns and optimize interactions.

## Conclusion

The examples in this project illustrate how combining LLMs like Ollama with tools such as Lavague and Playwright can lead to an adaptive, flexible web interaction agent. Moving forward, incorporating dynamic learning and real-time decision-making will be key in creating an agent capable of navigating and interacting with websites autonomously, ultimately making web interactions as adaptable and intelligent as human users.

## Repositories

- https://github.com/fellowship/web-agent/tree/main/ollama
- https://github.com/fellowship/web-agent/tree/main/lavague
- https://github.com/fellowship/web-agent/tree/main/playwright

---

**Note**: Ensure all dependencies (`Playwright`, `Lavague`, `Ollama`) are installed to utilize these examples fully. Refer to each tool’s documentation for detailed setup instructions.
