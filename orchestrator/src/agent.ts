// The agentic orchestrator (Vercel AI SDK v6 ToolLoopAgent). It drives the Rust festival-factory
// pipeline + Python stages as TOOLS, making coverage-driven decisions — it does not reimplement any
// pipeline logic. Faithfulness is enforced inside the Rust engine; the agent's job is sequencing and
// "is coverage good enough yet?".
import { ToolLoopAgent, stepCountIs } from 'ai';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { PIPELINE_TOOLS } from './tools';

// Reuse the OpenRouter key the Rust engine already uses (env, or export it from secret-tool first).
const openrouter = createOpenAICompatible({
  name: 'openrouter',
  baseURL: 'https://openrouter.ai/api/v1',
  apiKey: process.env.OPENROUTER_API_KEY ?? '',
});

const INSTRUCTIONS = `You orchestrate a festival continued-pretraining DATA pipeline. Your goal: take a
festival edition from its verified triples to a train-ready tokenized corpus, with RELATIONAL COVERAGE
as dense and multi-directional as possible. You drive stages via tools; you never invent facts —
faithfulness is enforced inside the hydration engine.

Run this loop, economically (don't repeat a stage that already succeeded):
1. coveragePlan — see which relations are buildable now vs blocked on missing data. Report blocked ones
   but do NOT try to generate them (no data = fabrication).
2. buildSchedule — only if the lineup triples look stale/missing.
3. hydrate with a small --limit first (smoke) to confirm it runs and lines are faithful, then a full run.
4. hydratedStats — check the distinct-ratio (dedup health; want > 0.8) and per-relation counts. If a
   supported relation is far below the others (thin coverage), hydrate again with a higher base-n —
   thin relations do not install (this is the whole point). Stop topping up when relations are balanced.
5. tokenize — produce the trainable shard. This is the finish line for data; training is run separately.

Be concise in your final summary: what you ran, the coverage tally, distinct-ratio, final token count,
and any blocked relations with the data file that would unblock them. If a tool returns ok:false, report
the error and stop rather than guessing.`;

export function buildAgent(model = 'anthropic/claude-sonnet-4.5') {
  return new ToolLoopAgent({
    model: openrouter(model),
    instructions: INSTRUCTIONS,
    tools: PIPELINE_TOOLS,
    stopWhen: stepCountIs(24),
  });
}
