export type InferenceResult = {
  id: number;
  filename: string;
  model_name: string;
  centiloid: number;
  raw_output: number;
  label: string | null;
  created_at: string;
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
  const res = await fetch(`${apiBase}/results`);
  if (!res.ok) throw new Error(`GET /results failed (${res.status})`);
  const data: { count: number; results: InferenceResult[] } = await res.json();
  return data.results;
}
