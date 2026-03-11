const questionInput = document.getElementById("question");
const fileInput = document.getElementById("file");

// ------------------------
// Helper: Add chat bubble
// ------------------------
function add(role, text) {
  const chat = document.getElementById("chat");
  const div = document.createElement("div");

  div.classList.add("chat-bubble", role);
  div.innerText = text;

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;

  return div; // return node for later update
}

function toTitleCase(key) {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatMetricValue(value, indent = "  ") {
  if (Array.isArray(value)) {
    if (value.length === 0) return `${indent}(empty)`;
    let out = "";
    value.forEach((item, idx) => {
      if (typeof item === "string") {
        out += `${indent}[${idx + 1}] ${item}\n`;
      } else {
        out += `${indent}[${idx + 1}] ${JSON.stringify(item)}\n`;
      }
    });
    return out.trimEnd();
  }

  if (value && typeof value === "object") {
    return `${indent}${JSON.stringify(value, null, 2).replace(/\n/g, `\n${indent}`)}`;
  }

  if (typeof value === "number") {
    return `${indent}${Number.isInteger(value) ? value : value.toFixed(4)}`;
  }

  return `${indent}${String(value)}`;
}

// ------------------------
// INDEX DOCUMENT
// ------------------------
async function indexDoc() {
  const file = fileInput.files[0];
  if (!file) {
    alert("Please choose a file first");
    return;
  }

  const form = new FormData();
  form.append("file", file);

  add("bot", "Uploading document...");

  try {
    const res = await fetch("/index", {
      method: "POST",
      body: form,
    });

    if (!res.ok) {
      let message = `Upload failed (${res.status})`;
      try {
        const err = await res.json();
        if (err && err.detail) message = err.detail;
      } catch (_) {}
      throw new Error(message);
    }

    add("bot", "Document uploaded successfully");
  } catch (err) {
    add("bot", `Upload failed: ${err.message}`);
  }
}

// ------------------------
// ASK QUESTION
// ------------------------
async function ask() {
  const q = questionInput.value.trim();
  if (!q) return;

  add("user", q);
  questionInput.value = "";

  // Show thinking indicator
  const thinkingBubble = document.createElement("div");
  thinkingBubble.className = "chat-bubble bot";
  thinkingBubble.innerText = "Thinking...";
  document.getElementById("chat").appendChild(thinkingBubble);

  let data;
  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Request failed (${res.status}): ${errText}`);
    }

    data = await res.json();
  } catch (err) {
    thinkingBubble.remove();
    add("bot", `Request failed: ${err.message}`);
    return;
  }

  // Remove thinking
  thinkingBubble.remove();

  // 1. Show refinement information (may be asking for confirmation)
  const awaitingConfirmation = data.confirmation_status === "awaiting_confirmation";
  if (awaitingConfirmation) {
    let refText = "";

    if (data.clarifying_questions && data.clarifying_questions.length > 0) {
      refText += "Clarifying questions:\n";
      data.clarifying_questions.forEach((cq, idx) => {
        refText += `\n${idx + 1}. ${cq}`;
      });
    } else if (data.refined_query) {
      refText += `Do you mean: "${data.refined_query}"?`;
    }

    if (data.assumptions && data.assumptions.length > 0) {
      refText += "\n\nAssumptions:\n";
      data.assumptions.forEach((a, idx) => {
        refText += `\n${idx + 1}. ${a}`;
      });
    }

    if (refText.trim().length > 0) {
      add("bot", refText);
    }
  }

  // 2. Answer (may be empty when we're only asking for clarification)
  if (data.answer && data.answer.trim().length > 0) {
    add("bot", data.answer);
  }

  // 2b. Reasoning summary from Answer Synthesizer Agent
  if (data.reasoning_summary) {
    add("evaluation", `Reasoning summary:\n${data.reasoning_summary}`);
  }

  // 2c. Retry visibility (useful for workflow debugging/demo)
  if (data.retry_count && Number(data.retry_count) > 0) {
    let retryText = `Retry info\nAttempts: ${data.retry_count}`;
    if (data.retry_target) {
      retryText += `\nStage: ${data.retry_target}`;
    }
    if (data.confirmation_status === "failed_after_retries" && data.last_error) {
      retryText += `\nLast error: ${data.last_error}`;
    }
    add("evaluation", retryText);
  }

  // 3. Evaluation (RAGAS only)
  if (data.evaluation && data.answer && data.answer.trim().length > 0) {
    let ragasText = "RAGAS Evaluation";
    const metrics = data.evaluation.ragas_metrics || {};
    const userInput = metrics.user_input;
    const retrievedContexts = metrics.retrieved_contexts;
    const responseText = metrics.response;

    const faithfulnessVal =
      metrics.faithfulness !== undefined && metrics.faithfulness !== null
        ? metrics.faithfulness
        : data.evaluation.faithfulness_score;

    const answerRelevancyVal =
      metrics.answer_relevancy !== undefined && metrics.answer_relevancy !== null
        ? metrics.answer_relevancy
        : data.evaluation.answer_relevancy_score;

    if (userInput !== undefined) {
      ragasText += `\n- User Input:\n${String(userInput)}`;
    }

    if (retrievedContexts !== undefined) {
      ragasText += "\n\n- Retrieved Contexts:";
      if (Array.isArray(retrievedContexts) && retrievedContexts.length > 0) {
        retrievedContexts.forEach((ctx, idx) => {
          ragasText += `\n[${idx + 1}] ${String(ctx)}`;
        });
      } else {
        ragasText += `\n${String(retrievedContexts)}`;
      }
    }

    if (responseText !== undefined) {
      ragasText += `\n\n- Response:\n${String(responseText)}`;
    }

    if (faithfulnessVal !== undefined && faithfulnessVal !== null) {
      const f = Number(faithfulnessVal);
      ragasText += `\n\n- Faithfulness:\n${
        Number.isFinite(f) ? f.toFixed(4) : String(faithfulnessVal)
      }`;
    }

    if (answerRelevancyVal !== undefined && answerRelevancyVal !== null) {
      const ar = Number(answerRelevancyVal);
      ragasText += `\n\n- Answer Relevancy:\n${
        Number.isFinite(ar) ? ar.toFixed(4) : String(answerRelevancyVal)
      }`;
    }

    add("evaluation", ragasText);
  }

  // 4. Sources
  if (data.sources && data.sources.length > 0) {
    let sourceText = "Sources\n";

    data.sources.forEach((src, idx) => {
      sourceText += `\n${idx + 1}. ${src.source}`;
      if (src.page !== null && src.page !== undefined) {
        sourceText += ` (page ${src.page})`;
      }
      sourceText += `\n"${src.snippet}"\n`;
    });

    add("sources", sourceText);
  }

  // 5. Retrieved context with scores (from Retrieval Agent)
  if (data.retrieved_context && data.retrieved_context.length > 0) {
    let ctxText = "Retrieved context (top chunks):\n";
    data.retrieved_context.forEach((c, idx) => {
      ctxText += `\n${idx + 1}. chunk_id: ${c.chunk_id || "n/a"}, page: ${
        c.page ?? "n/a"
      }, score: ${c.score?.toFixed ? c.score.toFixed(4) : c.score}\n`;
      ctxText += `"${(c.text || "").slice(0, 220)}"...\n`;
    });
    add("sources", ctxText);
  }

  // 6. Citations per key claim (from Answer Synthesizer Agent)
  if (data.citations && data.citations.length > 0) {
    let citText = "Citations (claim → chunk/page):\n";
    data.citations.forEach((c, idx) => {
      citText += `\n${idx + 1}. "${c.claim}" → chunk_id: ${
        c.chunk_id || "n/a"
      }, page: ${c.page ?? "n/a"}`;
    });
    add("sources", citText);
  }
}


// ENTER key support
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask();
});
