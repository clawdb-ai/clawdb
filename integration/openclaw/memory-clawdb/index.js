import crypto from "node:crypto";

function normalizeBaseUrl(baseUrl) {
  return String(baseUrl || "").replace(/\/+$/, "");
}

function parseProviderOrder(rawOrder, envOrder) {
  const direct = Array.isArray(rawOrder) ? rawOrder : null;
  if (direct && direct.length > 0) {
    return direct.map((item) => String(item).trim().toLowerCase()).filter(Boolean);
  }
  const envParsed = String(envOrder || "")
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
  if (envParsed.length > 0) {
    return envParsed;
  }
  return ["openai", "voyage", "mistral"];
}

function defaultEmbeddingModel(provider, override) {
  if (override && String(override).trim()) {
    return String(override).trim();
  }
  if (provider === "voyage") {
    return "voyage-3.5-lite";
  }
  if (provider === "mistral") {
    return "mistral-embed";
  }
  return "text-embedding-3-small";
}

async function resolveEmbeddingAuth({ api, providerOrder, modelOverride, baseUrlOverride }) {
  for (const provider of providerOrder) {
    try {
      const resolved = await api.runtime.modelAuth.resolveApiKeyForProvider({
        provider,
        cfg: api.config,
      });
      if (resolved?.apiKey) {
        return {
          provider,
          apiKey: resolved.apiKey,
          model: defaultEmbeddingModel(provider, modelOverride),
          baseUrl: baseUrlOverride ? normalizeBaseUrl(baseUrlOverride) : undefined,
          authSource: resolved.source || "openclaw.modelAuth",
        };
      }
    } catch {
      // Keep trying next provider.
    }
  }
  return null;
}

async function callBackend({
  baseUrl,
  apiKey,
  path,
  payload,
  timeoutMs,
  embeddingAuth,
  forwardEmbeddingHeaders = true,
}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const bodyText = JSON.stringify(payload || {});
  const headers = { "content-type": "application/json" };
  const authKey = apiKey || embeddingAuth?.apiKey || "";
  if (authKey) {
    headers.authorization = `Bearer ${authKey}`;
  }
  if (embeddingAuth && forwardEmbeddingHeaders) {
    headers["x-clawdb-embedding-provider"] = embeddingAuth.provider;
    headers["x-clawdb-embedding-key"] = embeddingAuth.apiKey;
    headers["x-clawdb-embedding-model"] = embeddingAuth.model;
    if (embeddingAuth.baseUrl) {
      headers["x-clawdb-embedding-base-url"] = embeddingAuth.baseUrl;
    }
    if (embeddingAuth.authSource) {
      headers["x-clawdb-embedding-auth-source"] = embeddingAuth.authSource;
    }
  }
  const signingKey = authKey;
  if (signingKey) {
    const ts = Math.floor(Date.now() / 1000);
    const signature = crypto
      .createHmac("sha256", signingKey)
      .update(`${path}\n${ts}`)
      .digest("hex");
    headers["x-openclaw-signature"] = signature;
    headers["x-openclaw-signature-ts"] = String(ts);
  }
  try {
    const response = await fetch(`${baseUrl}${path}`, {
      method: "POST",
      headers,
      body: bodyText,
      signal: controller.signal,
    });
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};
    if (!response.ok) {
      throw new Error(`clawdb request failed (${response.status}): ${JSON.stringify(data)}`);
    }
    return data;
  } finally {
    clearTimeout(timer);
  }
}

const plugin = {
  id: "memory-clawdb",
  name: "Memory (clawdb)",
  description: "Route OpenClaw memory tools and CLI through clawdb backend",
  kind: "memory",
  configSchema: {
    type: "object",
    additionalProperties: false,
    properties: {
      baseUrl: { type: "string" },
      apiKey: { type: "string" },
      requestTimeoutMs: { type: "number", minimum: 100, maximum: 60000 },
      embeddingProvider: { type: "string" },
      embeddingModel: { type: "string" },
      embeddingBaseUrl: { type: "string" },
      embeddingProviders: {
        type: "array",
        items: { type: "string" },
      },
    },
  },

  register(api) {
    const raw = api.pluginConfig || {};
    const baseUrl = normalizeBaseUrl(raw.baseUrl || process.env.CLAWDB_BASE_URL || "http://127.0.0.1:8080");
    const apiKey = raw.apiKey || process.env.CLAWDB_API_KEY || "";
    const timeoutMs = Number(raw.requestTimeoutMs || process.env.CLAWDB_REQUEST_TIMEOUT_MS || 10000);
    const providerOrder = raw.embeddingProvider
      ? [String(raw.embeddingProvider).trim().toLowerCase()]
      : parseProviderOrder(raw.embeddingProviders, process.env.CLAWDB_EMBEDDING_PROVIDERS);
    const embeddingModel = raw.embeddingModel || process.env.CLAWDB_EMBEDDING_MODEL || "";
    const embeddingBaseUrl = raw.embeddingBaseUrl || process.env.CLAWDB_EMBEDDING_BASE_URL || "";

    api.registerTool(
      {
        name: "memory_search",
        label: "Memory Search",
        description: "Search memory backed by clawdb.",
        parameters: {
          type: "object",
          additionalProperties: false,
          required: ["query"],
          properties: {
            query: { type: "string", description: "Search query" },
            max_results: { type: "number", description: "Max results" },
            min_score: { type: "number", description: "Minimum score" },
            session_key: { type: "string", description: "Optional session key" },
            tenant_id: { type: "string", description: "Optional tenant scope (default)" },
          },
        },
        async execute(_toolCallId, params) {
          const embeddingAuth = await resolveEmbeddingAuth({
            api,
            providerOrder,
            modelOverride: embeddingModel,
            baseUrlOverride: embeddingBaseUrl,
          });
          const payload = {
            query: params.query,
            tenantId: params.tenant_id ?? "default",
            maxResults: params.max_results ?? 6,
            minScore: params.min_score ?? 0,
            sessionKey: params.session_key
          };
          const results = await callBackend({
            baseUrl,
            apiKey,
            path: "/v1/openclaw/memory/search",
            payload,
            timeoutMs,
            embeddingAuth,
          });
          if (!Array.isArray(results) || results.length === 0) {
            return {
              content: [{ type: "text", text: "No memory results." }],
              details: { count: 0, source: "clawdb" }
            };
          }
          const text = results
            .map((item, idx) => `${idx + 1}. ${item.path}:${item.startLine} score=${item.score} ${item.snippet}`)
            .join("\n");
          return {
            content: [{ type: "text", text: text.slice(0, 6000) }],
            details: { count: results.length, source: "clawdb", results }
          };
        }
      },
      { names: ["memory_search"] }
    );

    api.registerTool(
      {
        name: "memory_get",
        label: "Memory Get",
        description: "Read memory content backed by clawdb.",
        parameters: {
          type: "object",
          additionalProperties: false,
          required: ["rel_path"],
          properties: {
            rel_path: { type: "string", description: "Relative memory path" },
            from: { type: "number", description: "Start line" },
            lines: { type: "number", description: "Line count" },
          },
        },
        async execute(_toolCallId, params) {
          const embeddingAuth = await resolveEmbeddingAuth({
            api,
            providerOrder,
            modelOverride: embeddingModel,
            baseUrlOverride: embeddingBaseUrl,
          });
          const payload = {
            relPath: params.rel_path,
            from: params.from ?? 1,
            lines: params.lines ?? 200,
          };
          const data = await callBackend({
            baseUrl,
            apiKey,
            path: "/v1/openclaw/memory/get",
            payload,
            timeoutMs,
            embeddingAuth,
            forwardEmbeddingHeaders: false,
          });
          return {
            content: [{ type: "text", text: data.text || "" }],
            details: { source: "clawdb", path: data.path || payload.relPath }
          };
        }
      },
      { names: ["memory_get"] }
    );

    api.registerCli(({ program }) => {
      const clawdbMemory = program
        .command("clawdb-memory")
        .description("Memory commands routed to clawdb");

      clawdbMemory
        .command("status")
        .description("Show clawdb health and cache-hit report")
        .action(async () => {
          const health = await fetch(`${baseUrl}/v1/memory/health`).then((r) => r.json());
          const cacheHit = await fetch(`${baseUrl}/v1/memory/metrics/cache-hit`).then((r) => r.json());
          console.log(JSON.stringify({ health, cacheHit }, null, 2));
        });

      clawdbMemory
        .command("search")
        .argument("<query>", "search query")
        .option("--max-results <n>", "max results", "6")
        .option("--min-score <n>", "min score", "0")
        .option("--session-key <id>", "session key")
        .option("--tenant-id <id>", "tenant scope", "default")
        .action(async (query, opts) => {
          const embeddingAuth = await resolveEmbeddingAuth({
            api,
            providerOrder,
            modelOverride: embeddingModel,
            baseUrlOverride: embeddingBaseUrl,
          });
          const payload = {
            query,
            tenantId: opts.tenantId ?? "default",
            maxResults: Number(opts.maxResults),
            minScore: Number(opts.minScore),
            sessionKey: opts.sessionKey,
          };
          const results = await callBackend({
            baseUrl,
            apiKey,
            path: "/v1/openclaw/memory/search",
            payload,
            timeoutMs,
            embeddingAuth,
          });
          console.log(JSON.stringify(results, null, 2));
        });

      clawdbMemory
        .command("get")
        .argument("<relPath>", "relative memory path")
        .option("--from <n>", "from line", "1")
        .option("--lines <n>", "line count", "200")
        .action(async (relPath, opts) => {
          const embeddingAuth = await resolveEmbeddingAuth({
            api,
            providerOrder,
            modelOverride: embeddingModel,
            baseUrlOverride: embeddingBaseUrl,
          });
          const data = await callBackend({
            baseUrl,
            apiKey,
            path: "/v1/openclaw/memory/get",
            payload: {
              relPath,
              from: Number(opts.from),
              lines: Number(opts.lines),
            },
            timeoutMs,
            embeddingAuth,
            forwardEmbeddingHeaders: false,
          });
          console.log(data.text || "");
        });
    }, { commands: ["clawdb-memory"] });

    api.logger.info(`memory-clawdb registered: ${baseUrl}`);
  },
};

export default plugin;
