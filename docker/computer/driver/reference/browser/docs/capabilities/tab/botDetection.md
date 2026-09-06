# Tab Capability: botDetection
Reports when a cloud browser task is blocked by bot detection, CAPTCHA, hard access denial, or a repeated challenge loop.

## Bot Detection Reporting
Use this capability only when the current cloud browser tab is blocked by
bot-detection, anti-automation, human-verification, or related access-control
systems served by the target site or its anti-bot provider. Report the blocker
only after the current URL and visible page provide enough evidence to classify
it, then stop or continue according to the surrounding Browser Safety guidance.
This report is internal telemetry. It does not replace telling the user that
the site blocked this browser or, when appropriate, switching to another
reputable source as directed by the Cloud Browser Context.

Do not use this capability for a bare HTTP status code or for browser,
organization, or network-policy failures. `ERR_BLOCKED_BY_ADMINISTRATOR`, proxy
or egress denials, DNS or TLS failures, timeouts, and refused or reset
connections are not bot detection.

### Choose The Reason
- `captcha_failed`: Use after you attempted a CAPTCHA or human-verification
  challenge with the user's permission, but the site rejected the attempt or
  the challenge still blocks progress.
- `access_denied`: Use for hard site-served access blocks such as Access Denied,
  request blocked, forbidden, or bot traffic denied when the page identifies
  bot, automation, or security screening and does not present an interactive
  challenge. A bare 403 is not enough.
- `challenge_loop`: Use when the site repeatedly reloads, loops through a
  challenge, sends you back to the same verification page, or never reaches the
  intended content after reasonable attempts.
- `unexpected_bot_error`: Use for bot-related failures that do not fit the
  above categories, such as an anti-bot page reporting a script crash or
  required browser feature as unavailable, or another visible
  automation-detection error.

Use the most specific matching value. Do not invent a free-form reason.

```js
const botDetection = await tab.capabilities.get("botDetection");
const reportResult = await botDetection.report({
  reason: "captcha_failed",
});
nodeRepl.write(JSON.stringify(reportResult));
```

## API Reference
```ts
const capability = await tab.capabilities.get("botDetection");

interface BotDetectionTabCapability {
  report(options: { reason: "captcha_failed" | "access_denied" | "challenge_loop" | "unexpected_bot_error" }): Promise<{ hostname: null | string; status: "reported" }>; // Report the currently open page as blocked by bot detection. The runtime records only the parsed hostname, never the full URL.
}
```
