# Browser Visibility Guidance
- Keep browser work in the background by default.
- Show the browser when the user's request is primarily to put a page in front of them or let them watch the interaction, such as opening a URL for them, showing the current tab, or keeping the browser visible while testing.
- Do not show the browser when navigation is only a means to answer a question or verify behavior. Localhost targets and ordinary page navigation do not by themselves require visibility.
- When the browser should be visible, call `await (await browser.capabilities.get("visibility")).set(true)`.
