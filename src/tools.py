import asyncio
import platform

from src.custom_types import AgentState


async def click(state: AgentState):
    # - Click [Numerical_label]
    page = state['page']
    # curr_url = page.url
    click_args = state['prediction']['args']
    if click_args is None or len(click_args) != 1:
        return f"Failed to click bounding box labeled as number {click_args}"
    bbox_id = click_args[0]
    bbox_id = int(bbox_id)
    try:
        bbox = state['bboxes'][bbox_id]
    except Exception:
        return f"Error: no bbox for : {bbox_id}"

    x, y = bbox['x'], bbox['y']
    await page.mouse.click(x,y)
    # TODO: In the paper, they automatically parse any downloaded PDFs
    # We could add something similar here as well and generally
    # improve response format.

    # # Verify if the button changed anything 
    # # TODO(Ben): doesn't handle button presses that changed the state
    # # without changing the url. Also doesn't handle loops
    # if curr_url == page.url:
    #     status = "[FAILED] "
    # else:
    #     status = ""
    status = ""

    return f"{status}Clicked {bbox_id}"

async def type_text(state: AgentState):
    page = state['page']
    # curr_url = page.url
    type_args = state['prediction']['args']
    if type_args is None or len(type_args) != 2:
        return f"[FAILED] Typed in element from bounding box labeled as number {type_args}"
    bbox_id = type_args[0]
    bbox_id = int(bbox_id)
    bbox = state['bboxes'][bbox_id]
    x, y = bbox['x'], bbox['y']
    text_content = str(type_args[1])
    await page.mouse.click(x, y)

    # check if MacOS
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(text_content)
    await page.keyboard.press("Enter")
    # # Verify if the button changed anything 
    # # TODO(Ben): doesn't handle button presses that changed the state
    # # without changing the url. Also doesn't handle loops
    # if curr_url == page.url:
    #     status = "[FAILED] "
    # else:
    #     status = ""
    status = ""
    return f"{status}Typed {text_content} into {bbox_id} and submitted"

async def scroll(state: AgentState):
    page = state['page']
    scroll_args = state['prediction']['args']
    if scroll_args is None or len(scroll_args) != 2:
        return "[FAILED] Scroll due to incorrect arguments as arguments were {scroll_args}"
    
    target, direction = scroll_args

    if target.upper() == 'WINDOW':
        # Not sure of the best value for this...
        scroll_amount = 500
        scroll_direction = (
            -scroll_amount if direction.lower() == 'up' else scroll_amount
        )
        await page.evaluate(f"window.scrollBy(0, {scroll_direction})")
    else:
        # scrolling within a specific element
        scroll_amount=200
        target_id = int(target)
        bbox = state['bboxes'][target_id]
        x, y = bbox['x'], bbox['y']
        scroll_direction = (
            -scroll_amount if direction.lower() == 'up' else scroll_amount
        )
        await page.mouse.move(x, y)
        await page.mouse.wheel(0, scroll_direction)
    return f"Scrolled {direction} in {'window' if target.upper == 'WINDOW' else 'element'}"

async def wait(state: AgentState):
    sleep_time = 5
    await asyncio.sleep(sleep_time)
    return f"Waited for {sleep_time}s"

async def go_back(state: AgentState):
    page = state['page']
    await page.go_back()
    return f"Navigated back to page to {page.url}"

async def to_google(state: AgentState):
    page = state['page']
    await page.goto("https://www.google.com/")
    return "Navigated to google.com"
