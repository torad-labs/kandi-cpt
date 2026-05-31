// Pipeline-stage tools for the orchestrator agent. Each shells out to the Rust `festival-factory`
// binary or a Python stage and returns STRUCTURED results (counts, coverage, ok/err) the agent can
// reason over. The agent never reimplements pipeline logic — it decides ordering + when coverage is
// good enough; the stages do the work (Rust compute + Python curate/tokenize).
import { tool } from 'ai';
import { z } from 'zod';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { readFileSync, existsSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve, join } from 'node:path';

const execFileP = promisify(execFile);
const HERE = dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = process.env.KANDI_ROOT ?? resolve(HERE, '../..');
const RUST_BIN = join(REPO_ROOT, 'festival-factory/target/debug/festival-factory');
const PY = join(REPO_ROOT, '.venv/bin/python');

const Target = { festival: z.string().describe('festival slug, e.g. "edc"'),
                 edition: z.string().describe('edition, e.g. "las_vegas/2026"') };

async function run(cmd: string, args: string[], opts: { cwd?: string; env?: Record<string, string> } = {}) {
  try {
    const { stdout, stderr } = await execFileP(cmd, args, {
      cwd: opts.cwd ?? REPO_ROOT,
      env: { ...process.env, PYTHONPATH: 'scripts', ...(opts.env ?? {}) },
      maxBuffer: 32 * 1024 * 1024,
      timeout: 30 * 60_000,
    });
    return { ok: true as const, stdout, stderr };
  } catch (e: any) {
    return { ok: false as const, stdout: e.stdout ?? '', stderr: e.stderr ?? String(e) };
  }
}

export const coveragePlan = tool({
  description: 'Load the festival edition and print the relation coverage plan (which relations are buildable now vs blocked on missing data). No generation. Fast. Run this first.',
  inputSchema: z.object(Target),
  execute: async ({ festival, edition }) => {
    const r = await run(RUST_BIN, ['plan', '--root', REPO_ROOT, '--festival', festival, '--edition', edition]);
    return { ok: r.ok, plan: r.stdout.trim() || r.stderr.trim() };
  },
});

export const buildSchedule = tool({
  description: 'Deterministically (re)build the verified lineup triples (15_structured) from the official source — the grounding for hydration. Run before hydrate if triples are stale.',
  inputSchema: z.object(Target),
  execute: async ({ festival: _f, edition }) => {
    const r = await run(PY, ['scripts/build_schedule.py', '--editions', edition]);
    return { ok: r.ok, summary: r.stdout.split('\n').slice(0, 4).join('\n') || r.stderr.slice(0, 800) };
  },
});

export const hydrate = tool({
  description: 'Run the Rust hydration engine: grounded, faithfulness-gated relational generation → 25_hydrated/relations.jsonl. Returns the per-relation×direction coverage tally. Use --limit for a smoke run first.',
  inputSchema: z.object({ ...Target,
    baseN: z.number().int().min(1).max(60).default(12).describe('base lines per cell'),
    limit: z.number().int().positive().optional().describe('cap cells (smoke test)'),
    concurrency: z.number().int().min(1).max(32).default(8) }),
  execute: async ({ festival, edition, baseN, limit, concurrency }) => {
    const args = ['hydrate', '--root', REPO_ROOT, '--festival', festival, '--edition', edition,
                  '--base-n', String(baseN), '--concurrency', String(concurrency)];
    if (limit) args.push('--limit', String(limit));
    const r = await run(RUST_BIN, args);
    return { ok: r.ok, report: r.stdout.trim() || r.stderr.trim().slice(0, 1000) };
  },
});

export const hydratedStats = tool({
  description: 'Read 25_hydrated/relations.jsonl and report per-relation line counts, distinct-ratio (dedup health), and split breakdown. Use to judge whether coverage meets the floor before tokenizing.',
  inputSchema: z.object(Target),
  execute: async ({ festival, edition }) => {
    const path = join(REPO_ROOT, 'data', festival, edition, '25_hydrated/relations.jsonl');
    if (!existsSync(path)) return { ok: false, error: 'no relations.jsonl yet — run hydrate first' };
    const lines = readFileSync(path, 'utf8').split('\n').filter(Boolean);
    const byRel: Record<string, number> = {};
    const bySplit: Record<string, number> = {};
    const seen = new Set<string>();
    let dups = 0;
    for (const l of lines) {
      try {
        const o = JSON.parse(l);
        byRel[o.relation_type] = (byRel[o.relation_type] ?? 0) + 1;
        bySplit[o.split] = (bySplit[o.split] ?? 0) + 1;
        const k = String(o.text).toLowerCase().replace(/\s+/g, ' ').trim();
        if (seen.has(k)) dups++; else seen.add(k);
      } catch { /* skip */ }
    }
    const total = lines.length;
    return { ok: true, total, distinctRatio: total ? +(1 - dups / total).toFixed(3) : 0, byRelation: byRel, bySplit };
  },
});

export const tokenize = tool({
  description: 'Tokenize the corpus (clean + synthetic + hydrated) into the trainable Arrow shard via the Rust festival-factory (tokenizers + arrow IPC), emitting input_ids + lever metadata (entity_mask/src/fact_ids). Final stage → train-ready. Returns token + chunk + mask counts. (Python scripts/tokenize_corpus.py is the legacy fallback.)',
  inputSchema: z.object({ ...Target, seqLen: z.number().int().default(512) }),
  execute: async ({ festival, edition, seqLen }) => {
    const out = join(REPO_ROOT, 'data/_build/30_tokenized/corpus_v1.arrow');
    rmSync(out, { recursive: true, force: true }); // clear any prior dir (Python save_to_disk) or file
    const r = await run(RUST_BIN, ['tokenize', '--root', REPO_ROOT, '--festival', festival,
                                   '--edition', edition, '--seq-len', String(seqLen), '--out', out]);
    const stats = r.stderr.trim().split('\n').filter(Boolean).slice(-3).join(' ');
    return { ok: r.ok, summary: stats || r.stdout.slice(0, 600) };
  },
});

export const PIPELINE_TOOLS = { coveragePlan, buildSchedule, hydrate, hydratedStats, tokenize };
