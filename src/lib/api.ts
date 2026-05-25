export type InferenceResult = {
  id: number;
  filename: string;
  model_name: string;
  centiloid: number;
  raw_output: number;
  label: string | null;
  fold: number | null;
  created_at: string;
};

export type ResultsPage = {
  count: number;
  limit: number;
  offset: number;
  has_more: boolean;
  results: InferenceResult[];
};

export async function getModels(apiBase: string): Promise<string[]> {
  const res = await fetch(`${apiBase}/models`);
  if (!res.ok) throw new Error(`GET /models failed (${res.status})`);
  const data: { models: string[] } = await res.json();
  return data.models;
}

export async function runInference(
  apiBase: string,
  file: File,
  modelName: string,
  label?: string,
): Promise<InferenceResult> {
  const form = new FormData();
  form.append("file", file);
  form.append("model_name", modelName);
  if (label) form.append("label", label);

  const res = await fetch(`${apiBase}/inference`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail ?? res.statusText;
    throw new Error(`Inference failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function getResults(apiBase: string): Promise<InferenceResult[]> {
  const all: InferenceResult[] = [];
  const limit = 1000;
  let offset = 0;

  for (;;) {
    const page = await getResultsPage(apiBase, { limit, offset });
    all.push(...page.results);
    if (!page.has_more) return all;
    offset += page.results.length;
    if (page.results.length === 0) return all;
  }
}

export async function getResultsPage(
  apiBase: string,
  options: { fold?: number | null; limit?: number; offset?: number } = {},
): Promise<ResultsPage> {
  const params = new URLSearchParams();
  params.set("limit", String(options.limit ?? 250));
  params.set("offset", String(options.offset ?? 0));
  if (options.fold !== undefined && options.fold !== null) {
    params.set("fold", String(options.fold));
  }

  const res = await fetch(`${apiBase}/results?${params.toString()}`);
  if (!res.ok) throw new Error(`GET /results failed (${res.status})`);
  return res.json();
}

export async function getFolds(apiBase: string): Promise<number[]> {
  const res = await fetch(`${apiBase}/results/folds`);
  if (!res.ok) throw new Error(`GET /results/folds failed (${res.status})`);
  const data: { folds: number[] } = await res.json();
  return data.folds;
}

export async function patchResultFolds(
  apiBase: string,
  updates: Array<
    | { id: number; fold: number | null }
    | { filename: string; model_name: string; fold: number | null }
  >,
): Promise<{ updated: number; missing: unknown[] }> {
  const res = await fetch(`${apiBase}/results/fold`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ updates }),
  });
  if (!res.ok) throw new Error(`PATCH /results/fold failed (${res.status})`);
  return res.json();
}
