import { readFileSync } from "node:fs";

/** Load one exported assistant Turn without retaining unrelated session data. */
export function loadReplayTurn(path) {
  const payload = JSON.parse(readFileSync(path, "utf8"));
  if (!Array.isArray(payload)) throw new TypeError("replay turn must be a message array");
  const assistants = payload.filter((message) => record(message) && message.role === "assistant");
  if (assistants.length !== 1) throw new TypeError("replay turn must contain exactly one assistant message");
  const assistant = assistants[0];
  if (typeof assistant.content !== "string" || assistant.content.length === 0) {
    throw new TypeError("replay assistant content must be a non-empty string");
  }
  const rawStages = typeof assistant.tool_chain === "string"
    ? JSON.parse(assistant.tool_chain)
    : assistant.tool_chain;
  if (!Array.isArray(rawStages)) throw new TypeError("replay assistant tool_chain must be an array");
  return {
    content: assistant.content,
    stages: rawStages.map((stage, stageIndex) => loadStage(stage, stageIndex)),
  };
}

function loadStage(value, stageIndex) {
  if (!record(value)) throw new TypeError(`replay stage ${stageIndex + 1} must be an object`);
  const text = value.text ?? "";
  const reasoning = value.reasoning_content ?? "";
  const calls = value.calls ?? [];
  if (typeof text !== "string" || typeof reasoning !== "string" || !Array.isArray(calls)) {
    throw new TypeError(`replay stage ${stageIndex + 1} has invalid text, reasoning, or calls`);
  }
  return {
    text,
    reasoning,
    calls: calls.map((call, callIndex) => loadCall(call, stageIndex, callIndex)),
  };
}

function loadCall(value, stageIndex, callIndex) {
  const location = `replay stage ${stageIndex + 1} call ${callIndex + 1}`;
  if (!record(value)) throw new TypeError(`${location} must be an object`);
  for (const field of ["call_id", "name", "status", "result"]) {
    if (typeof value[field] !== "string" || (field !== "result" && value[field].length === 0)) {
      throw new TypeError(`${location} has invalid ${field}`);
    }
  }
  if (!record(value.arguments) || !record(value.final_arguments)) {
    throw new TypeError(`${location} has invalid arguments`);
  }
  return {
    callId: value.call_id,
    name: value.name,
    status: value.status,
    arguments: value.arguments,
    finalArguments: value.final_arguments,
    result: value.result,
  };
}

function record(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}
