import asyncio
import platform


async def click(page, value, position):
    await page.mouse.click(*position)
    return f"Clicked at position {position}"


async def type_text(page, value, position):
    await page.mouse.click(*position)
    # check if MacOS
    select_all = "Meta+A" if platform.system() == "Darwin" else "Control+A"
    await page.keyboard.press(select_all)
    await page.keyboard.press("Backspace")
    await page.keyboard.type(value)

    return f"Typed {value} into {position}"


async def hover(page, value, position):
    """Move mouse to a location and then wait"""
    await page.mouse.move(*position)
    wait_text = await wait(page, value, position)
    return f"Hovered mouse at {position} and " + wait_text


async def enter(page, value, position):
    """Hits enter"""
    await page.keyboard.press("Enter")
    return f"Hit enter"


async def answer(page, value, position):
    return value


async def scroll(page, value, position):
    scroll_amount = 500  # Not sure the best amount for this ...
    if value.lower().strip("\"'") == "down":
        await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
    else:
        await page.evaluate(f"window.scrollBy(0, {-scroll_amount})")

    return f"Scrolled {value}"


async def select_text(page, value, position):
    """Select some text content, secretly it's just a drag and drop under the hood"""
    first_pos = position[0]
    second_pos = position[1]
    await page.mouse.move(*first_pos)
    await page.mouse.down()
    await page.mouse.move(*second_pos)
    await page.mouse.up()
    return f"Highlighted text from {position[0]} to {position[1]}"


async def copy(page, value, position):
    """Copy Highlighted Text"""
    print(
        "WARNING: have not implemented actual access to the clipboard, so only copied into observations"
    )
    return "Copied the text {value}"


async def wait(page, value, position):
    await asyncio.sleep(
        3
    )  # Should we hardcode this? should we test other values? maybe check if value is None
    return f"Waited for 3s"


async def go_back(page, value, position):
    await page.go_back()
    return f"Navigated back to page to {page.url}"


async def to_google(page, value, position):
    await page.goto("https://www.google.com/")
    return "Navigated to google.com"
