const TOKENIZER_VERSION = "drle-regex-en-v1";
const MAX_BODY_BYTES = 16 * 1024;
const MAX_CONTEXT_CHARS = 4000;
const MAX_GENERATION_TOKENS = 64;
const MODEL_PATH = "/model.json";
let cachedModelPromise;

function encode(text) {
  if (typeof text !== "string") throw new TypeError("text must be a string");
  const matches = text.match(/[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*|[^\w\s]/gu) || [];
  return matches.map((token) => token.toLowerCase().replaceAll("’", "'"));
}

function decode(tokens) {
  let out = "";
  const noSpaceBefore = new Set(".,!?;:%)]}»”’");
  const noSpaceAfter = new Set("([{«“");
  for (const token of tokens) {
    if (!out) out = token;
    else if (noSpaceBefore.has(token) || noSpaceAfter.has(out.at(-1))) out += token;
    else out += ` ${token}`;
  }
  return out;
}

async function loadModel(env) {
  if (!cachedModelPromise) {
    cachedModelPromise = (async () => {
      if (!env.ASSETS || typeof env.ASSETS.fetch !== "function") throw new Error("ASSETS binding is not configured");
      const response = await env.ASSETS.fetch(new Request(`https://drle-assets.local${MODEL_PATH}`));
      if (!response.ok) throw new Error(`Model asset unavailable (${response.status})`);
      const model = await response.json();
      if (model.format_version !== 1 || model.tokenizer !== TOKENIZER_VERSION || !Array.isArray(model.transitions)) {
        throw new Error("Unsupported or invalid model artifact");
      }
      const tables = Array.from({ length: model.max_order + 1 }, () => new Map());
      for (const row of model.transitions) {
        if (!Array.isArray(row) || row.length !== 4) throw new Error("Malformed transition record");
        const [order, context, token, count] = row;
        if (!Number.isInteger(order) || order < 1 || order > model.max_order || !Array.isArray(context) || context.length !== order || typeof token !== "string" || !Number.isSafeInteger(count) || count < 1) throw new Error("Invalid transition record");
        const key = JSON.stringify(context);
        let candidates = tables[order].get(key);
        if (!candidates) tables[order].set(key, candidates = []);
        candidates.push([token, count]);
      }
      return { ...model, tables };
    })().catch((error) => { cachedModelPromise = undefined; throw error; });
  }
  return cachedModelPromise;
}

function predict(model, context, topK = 8, minCount = 1, backoff = true) {
  const tokens = encode(context);
  const gathered = new Map();
  const orders = [];
  const upper = Math.min(model.max_order, tokens.length || 1);
  for (let order = upper; order >= 1; order--) {
    const history = tokens.length ? tokens.slice(-order) : Array(order).fill("<s>");
    const found = model.tables[order].get(JSON.stringify(history));
    if (!found) continue;
    orders.push(order);
    for (const [token, count] of found) gathered.set(token, (gathered.get(token) || 0) + count * order * order);
    if (!backoff) break;
  }
  const candidates = [...gathered.entries()]
    .filter(([token, score]) => score >= minCount && token !== "<s>")
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .slice(0, topK);
  return { token: candidates[0]?.[0] ?? null, candidates, orders, status: candidates.length ? "OK" : "NO_SUPPORTED_CONTINUATION" };
}

function json(data, status = 200, origin = "*") {
  return new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", "access-control-allow-origin": origin, "access-control-allow-methods": "GET,POST,OPTIONS", "access-control-allow-headers": "content-type" } });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const origin = request.headers.get("origin") || "*";
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: { "access-control-allow-origin": origin, "access-control-allow-methods": "GET,POST,OPTIONS", "access-control-allow-headers": "content-type", "access-control-max-age": "86400" } });
    if (url.pathname === "/health" && request.method === "GET") return json({ status: "ok", service: "DRLE", version: "0.1.0" }, 200, origin);
    if (url.pathname !== "/predict" && url.pathname !== "/generate") return json({ error: "Not found" }, 404, origin);
    if (request.method !== "POST") return json({ error: "Method not allowed" }, 405, origin);
    const declaredLength = Number(request.headers.get("content-length") || 0);
    if (declaredLength > MAX_BODY_BYTES) return json({ error: "Request body too large" }, 413, origin);
    let body;
    try {
      const bytes = await request.arrayBuffer();
      if (bytes.byteLength > MAX_BODY_BYTES) return json({ error: "Request body too large" }, 413, origin);
      body = JSON.parse(new TextDecoder().decode(bytes));
    } catch { return json({ error: "Request body must be valid JSON" }, 400, origin); }
    if (!body || typeof body.context !== "string" && typeof body.prompt !== "string") return json({ error: "Provide a context or prompt string" }, 400, origin);
    const input = body.context ?? body.prompt;
    if (input.length > MAX_CONTEXT_CHARS) return json({ error: `Input exceeds ${MAX_CONTEXT_CHARS} characters` }, 413, origin);
    try {
      const model = await loadModel(env);
      if (url.pathname === "/predict") {
        const topK = Number(body.top_k ?? 8), minCount = Number(body.min_count ?? 1);
        if (!Number.isInteger(topK) || topK < 1 || topK > 20 || !Number.isSafeInteger(minCount) || minCount < 1) return json({ error: "top_k must be 1..20 and min_count must be a positive integer" }, 400, origin);
        return json(predict(model, input, topK, minCount, body.backoff !== false), 200, origin);
      }
      const maxTokens = Number(body.max_tokens ?? 24);
      if (!Number.isInteger(maxTokens) || maxTokens < 0 || maxTokens > MAX_GENERATION_TOKENS) return json({ error: `max_tokens must be 0..${MAX_GENERATION_TOKENS}` }, 400, origin);
      const output = encode(input);
      for (let i = 0; i < maxTokens; i++) {
        const result = predict(model, decode(output), 1);
        if (!result.token || result.token === "</s>") break;
        output.push(result.token);
      }
      return json({ text: decode(output), generated_tokens: output.length - encode(input).length, status: "OK" }, 200, origin);
    } catch (error) {
      return json({ error: "Model inference unavailable", detail: error instanceof Error ? error.message : "Unknown error" }, 500, origin);
    }
  }
};

export { encode, decode, predict, loadModel };
