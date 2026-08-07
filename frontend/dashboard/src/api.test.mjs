import assert from "node:assert/strict";
import test from "node:test";

import {
  ApiError,
  api,
  interactionDeleteRequirement,
} from "./api.ts";

test("api preserves structured interaction delete requirements", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({
    detail: {
      code: "interaction_delete_required",
      message_id: "u2",
      control_turn_id: "interaction-1",
    },
  }), {
    status: 409,
    headers: { "Content-Type": "application/json" },
  });

  try {
    await assert.rejects(
      api("/api/dashboard/messages/batch-delete"),
      (error) => {
        assert.ok(error instanceof ApiError);
        assert.equal(error.status, 409);
        assert.deepEqual(interactionDeleteRequirement(error), {
          code: "interaction_delete_required",
          message_id: "u2",
          control_turn_id: "interaction-1",
        });
        return true;
      },
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("interaction delete requirement rejects unrelated failures", () => {
  assert.equal(
    interactionDeleteRequirement(new ApiError(500, "boom", "boom")),
    null,
  );
});
