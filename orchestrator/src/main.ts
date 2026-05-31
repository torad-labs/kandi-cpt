// CLI entry for the agentic orchestrator.
//   OPENROUTER_API_KEY=... npx tsx src/main.ts --festival edc --edition las_vegas/2026
// (export the key first: export OPENROUTER_API_KEY="$(secret-tool lookup service openrouter key_type api_key)")
import { buildAgent } from './agent';

function arg(name: string, def?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 ? process.argv[i + 1] : def;
}

const festival = arg('festival', 'edc')!;
const edition = arg('edition', 'las_vegas/2026')!;
const model = arg('model', 'anthropic/claude-sonnet-4.5')!;
const goal =
  arg('goal') ??
  `Bring festival="${festival}" edition="${edition}" to a train-ready tokenized corpus with strong, ` +
    `balanced relational coverage. Start with coveragePlan, then proceed per your instructions.`;

if (!process.env.OPENROUTER_API_KEY) {
  console.error(
    'Set OPENROUTER_API_KEY first:\n' +
      '  export OPENROUTER_API_KEY="$(secret-tool lookup service openrouter key_type api_key)"',
  );
  process.exit(1);
}

const agent = buildAgent(model);
console.error(`▶ orchestrating ${festival}/${edition} with ${model}\n`);

const res = await agent.generate({ prompt: goal });

// Surface the tool trajectory (which stages the agent decided to run, in order).
const steps = (res as { steps?: Array<{ toolCalls?: Array<{ toolName: string; input?: unknown }> }> }).steps ?? [];
for (const step of steps) {
  for (const tc of step.toolCalls ?? []) {
    console.error(`  · ${tc.toolName}(${JSON.stringify(tc.input ?? {})})`);
  }
}
console.log('\n=== ORCHESTRATOR SUMMARY ===\n' + res.text);
