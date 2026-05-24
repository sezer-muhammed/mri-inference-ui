"use client";

import dynamic from "next/dynamic";
import {
  AlertCircle,
  Brain,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  FileUp,
  Loader2,
  RefreshCw,
  Settings2,
  Upload,
  X,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/cn";
import { type InferenceResult, getModels, getResults, runInference } from "@/lib/api";
import { DataTable, type TableColumn } from "@/components/ui/data-table";

const NiiViewer = dynamic(
  () => import("@/components/nii-viewer").then((m) => ({ default: m.NiiViewer })),
  {
    ssr: false,
    loading: () => (
      <div className="flex h-full items-center justify-center rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)]">
        <div className="flex items-center gap-2 font-mono text-[12px] text-[var(--ds-gray-700)]">
          <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-[var(--ds-blue-700)] border-t-transparent" />
          Initialising viewer…
        </div>
      </div>
    ),
  },
);

/* ─── types ───────────────────────────────────────────────── */
type RunState =
  | { status: "idle" }
  | { status: "running" }
  | { status: "done"; result: InferenceResult }
  | { status: "error"; message: string };

/* ─── centiloid helpers ───────────────────────────────────── */
const CENTILOID_THRESHOLDS = [
  { max: 0, label: "Very low", color: "var(--ds-gray-700)" },
  { max: 25, label: "Low", color: "var(--ds-green-700)" },
  { max: 50, label: "Borderline", color: "var(--ds-amber-700)" },
  { max: 100, label: "Positive", color: "var(--ds-red-700)" },
  { max: Infinity, label: "High positive", color: "var(--ds-pink-700, #e5195e)" },
] as const;

function centiloidMeta(v: number) {
  return CENTILOID_THRESHOLDS.find((t) => v < t.max) ?? CENTILOID_THRESHOLDS[3];
}

function CentiloidGauge({ value }: { value: number }) {
  const meta = centiloidMeta(value);
  // display range: -20 → 150, clamped
  const min = -20;
  const max = 150;
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));

  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline gap-2">
        <span className="text-[24px] font-semibold leading-none tabular-nums" style={{ color: meta.color }}>
          {value.toFixed(1)}
        </span>
        <span className="font-mono text-[11px] uppercase" style={{ color: meta.color }}>
          {meta.label}
        </span>
      </div>
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-[var(--ds-gray-200)]">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: meta.color }}
        />
        {/* Threshold ticks */}
        {[0, 25, 50, 100].map((t) => {
          const tp = Math.max(0, Math.min(100, ((t - min) / (max - min)) * 100));
          return (
            <div
              key={t}
              className="absolute top-0 h-full w-px bg-[var(--ds-background-100)]/60"
              style={{ left: `${tp}%` }}
            />
          );
        })}
      </div>
      <div className="flex justify-between font-mono text-[10px] text-[var(--ds-gray-600)]">
        <span>0</span>
        <span>25</span>
        <span>50</span>
        <span>100</span>
        <span>150</span>
      </div>
    </div>
  );
}

function CentiloidCell({ value }: { value: number }) {
  const meta = centiloidMeta(value);
  const pct = Math.max(0, Math.min(100, ((value + 20) / 170) * 100));
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-[var(--ds-gray-200)]">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: meta.color }} />
      </div>
      <span className="font-mono text-[12px] font-semibold tabular-nums" style={{ color: meta.color }}>
        {value.toFixed(1)}
      </span>
    </div>
  );
}

/* ─── misc helpers ────────────────────────────────────────── */
function fmtDate(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function StatusDot({ color, pulse }: { color: string; pulse?: boolean }) {
  return (
    <span
      className={cn("inline-block h-2 w-2 shrink-0 rounded-full border border-[var(--ds-gray-alpha-500)]", pulse && "animate-pulse")}
      style={{ background: color }}
    />
  );
}

function deltaColor(delta: number) {
  if (delta < 10) return "var(--ds-green-700)";
  if (delta < 25) return "var(--ds-amber-700)";
  return "var(--ds-red-700)";
}

/* ─── page ────────────────────────────────────────────────── */
export default function Page() {
  const defaultApi = process.env.NEXT_PUBLIC_API_BASE ?? "https://sezer-muhammed-mri-inference-api.hf.space";
  const [apiBase, setApiBase] = useState(defaultApi);
  const [apiBaseInput, setApiBaseInput] = useState(defaultApi);
  const [showSettings, setShowSettings] = useState(false);

  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [models, setModels] = useState<string[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [modelsWarmingUp, setModelsWarmingUp] = useState(false);
  const [modelName, setModelName] = useState("");

  const [label, setLabel] = useState("");
  const [runState, setRunState] = useState<RunState>({ status: "idle" });

  const [results, setResults] = useState<InferenceResult[]>([]);
  const [resultsLoading, setResultsLoading] = useState(false);
  const [resultsError, setResultsError] = useState<string | null>(null);

  /* Load models + results whenever apiBase changes */
  const loadModels = useCallback(async () => {
    setModelsLoading(true);
    setModelsWarmingUp(false);
    // Retry up to 8 times (covers ~60s HF Space cold-start)
    for (let attempt = 0; attempt < 8; attempt++) {
      try {
        const list = await getModels(apiBase);
        setModels(list);
        if (list.length > 0 && !modelName) setModelName(list[0]);
        setModelsWarmingUp(false);
        setModelsLoading(false);
        return;
      } catch {
        if (attempt < 7) {
          setModelsWarmingUp(true);
          await new Promise((r) => setTimeout(r, 8000));
        }
      }
    }
    setModels([]);
    setModelsWarmingUp(false);
    setModelsLoading(false);
  }, [apiBase, modelName]);

  const loadResults = useCallback(async () => {
    setResultsLoading(true);
    setResultsError(null);
    try {
      setResults(await getResults(apiBase));
    } catch (err: any) {
      setResultsError(err.message);
    } finally {
      setResultsLoading(false);
    }
  }, [apiBase]);

  useEffect(() => {
    loadModels();
    loadResults();
  }, [loadModels, loadResults]);

  /* File handling */
  const handleFile = useCallback((f: File) => {
    if (!/\.(nii|nii\.gz|tar)$/i.test(f.name)) {
      alert("Please select a .nii, .nii.gz, or .tar file.");
      return;
    }
    setFile(f);
    setRunState({ status: "idle" });
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  /* Inference */
  const handleRun = useCallback(async () => {
    if (!file || !modelName.trim() || runState.status === "running") return;
    setRunState({ status: "running" });
    try {
      const result = await runInference(apiBase, file, modelName.trim(), label.trim() || undefined);
      setRunState({ status: "done", result });
      await loadResults();
    } catch (err: any) {
      setRunState({ status: "error", message: err.message });
    }
  }, [file, modelName, label, runState.status, apiBase, loadResults]);

  const canRun = !!file && !!modelName.trim() && runState.status !== "running";
  const isRunning = runState.status === "running";

  /* ─── render ────────────────────────────────────────────── */
  return (
    <div className="min-h-screen text-[var(--ds-gray-1000)]">

      {/* Header */}
      <header className="depth-surface sticky top-3 z-20 mx-auto grid min-h-14 w-full max-w-[1440px] grid-cols-[1fr_auto] items-center gap-3 rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="grid h-9 w-9 shrink-0 place-items-center rounded-[7px] border border-[var(--ds-gray-1000)] bg-[var(--ds-gray-1000)]">
            <Brain aria-hidden className="h-4 w-4 text-[var(--ds-background-100)]" />
          </div>
          <div className="min-w-0">
            <p className="truncate text-[14px] font-semibold leading-5">MRI Inference UI</p>
            <p className="truncate font-mono text-[11px] text-[var(--ds-gray-700)]">
              Centiloid regression · 3D NIfTI
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="hidden items-center gap-2 md:flex">
            <StatusDot
              color={resultsError ? "var(--ds-red-700)" : "var(--ds-green-700)"}
              pulse={resultsLoading || modelsLoading}
            />
            <span className="font-mono text-[12px] text-[var(--ds-gray-700)]">{apiBase}</span>
          </div>
          <button
            onClick={() => { setApiBaseInput(apiBase); setShowSettings((v) => !v); }}
            className="inline-flex h-8 items-center gap-1.5 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2.5 text-[12px] font-medium text-[var(--ds-gray-900)] transition hover:bg-[var(--ds-gray-100)]"
          >
            <Settings2 aria-hidden className="h-3.5 w-3.5" />
            API
            {showSettings ? <ChevronUp aria-hidden className="h-3 w-3" /> : <ChevronDown aria-hidden className="h-3 w-3" />}
          </button>
        </div>
      </header>

      {/* API settings */}
      {showSettings && (
        <div className="mx-auto mt-2 max-w-[1440px]">
          <div className="rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] p-3">
            <p className="mb-2 font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">API base URL</p>
            <div className="flex gap-2">
              <input
                className="h-9 flex-1 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-3 font-mono text-[13px] outline-none focus:border-[var(--ds-blue-700)]"
                value={apiBaseInput}
                onChange={(e) => setApiBaseInput(e.target.value)}
                placeholder="http://localhost:7860"
              />
              <button
                onClick={() => {
                  setApiBase(apiBaseInput.replace(/\/$/, ""));
                  setShowSettings(false);
                }}
                className="inline-flex h-9 items-center rounded-[7px] border border-[var(--ds-gray-1000)] bg-[var(--ds-gray-1000)] px-3 text-[13px] font-medium text-[var(--ds-background-100)] transition hover:bg-black"
              >
                Save &amp; reconnect
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main */}
      <main className="mx-auto max-w-[1440px] space-y-4 py-4">
        <div className="grid gap-4 xl:grid-cols-[380px_1fr]">

          {/* ── Left panel ── */}
          <div className="space-y-3">

            {/* Upload + controls */}
            <div className="rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)]">
              <div className="border-b border-[var(--ds-gray-alpha-300)] px-4 py-3">
                <p className="font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">01 / Input</p>
                <h2 className="mt-1 text-[15px] font-semibold">Upload MRI scan</h2>
              </div>

              <div className="space-y-4 p-4">
                {/* Drop zone */}
                <div
                  onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={cn(
                    "relative flex min-h-[120px] cursor-pointer flex-col items-center justify-center gap-2 rounded-[8px] border-2 border-dashed p-4 text-center transition",
                    isDragging
                      ? "border-[var(--ds-blue-700)] bg-[var(--ds-blue-100)]"
                      : file
                      ? "border-[var(--ds-green-700)] bg-[var(--ds-green-100)]"
                      : "border-[var(--ds-gray-300)] bg-[var(--ds-background-200)] hover:border-[var(--ds-gray-500)] hover:bg-[var(--ds-gray-100)]",
                  )}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".nii,.nii.gz,.tar,.tar.gz"
                    className="hidden"
                    onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
                  />
                  {file ? (
                    <>
                      <CheckCircle2 className="h-7 w-7 text-[var(--ds-green-700)]" />
                      <div>
                        <p className="text-[13px] font-semibold text-[var(--ds-green-900)]">{file.name}</p>
                        <p className="mt-0.5 font-mono text-[11px] text-[var(--ds-green-700)]">
                          {(file.size / 1024 / 1024).toFixed(1)} MB
                        </p>
                      </div>
                      <button
                        onClick={(e) => { e.stopPropagation(); setFile(null); setRunState({ status: "idle" }); }}
                        className="absolute right-2 top-2 grid h-6 w-6 place-items-center rounded-full border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] transition hover:bg-[var(--ds-gray-100)]"
                      >
                        <X className="h-3.5 w-3.5 text-[var(--ds-gray-800)]" />
                      </button>
                    </>
                  ) : (
                    <>
                      <Upload className="h-7 w-7 text-[var(--ds-gray-500)]" />
                      <div>
                        <p className="text-[13px] font-medium text-[var(--ds-gray-900)]">
                          Drop .nii / .nii.gz / .tar here
                        </p>
                        <p className="mt-0.5 font-mono text-[11px] text-[var(--ds-gray-600)]">or click to browse</p>
                      </div>
                    </>
                  )}
                </div>

                {/* Model selector */}
                <div>
                  <label className="block font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">
                    Model <span className="text-[var(--ds-red-700)]">*</span>
                  </label>
                  {modelsLoading ? (
                    <div className="mt-1 flex h-9 items-center gap-2 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-3">
                      <div className="h-3 w-3 animate-spin rounded-full border-2 border-[var(--ds-blue-700)] border-t-transparent" />
                      <span className="font-mono text-[12px] text-[var(--ds-gray-600)]">
                        {modelsWarmingUp ? "Warming up API…" : "Loading models…"}
                      </span>
                    </div>
                  ) : models.length > 0 ? (
                    <select
                      className="mt-1 block h-9 w-full rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-3 text-[13px] font-medium outline-none transition focus:border-[var(--ds-blue-700)]"
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                    >
                      {models.map((m) => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      className="mt-1 block h-9 w-full rounded-[7px] border border-[var(--ds-amber-400)] bg-[var(--ds-amber-100)] px-3 font-mono text-[12px] outline-none transition focus:border-[var(--ds-blue-700)] focus:bg-[var(--ds-background-200)]"
                      placeholder="No models found — type name manually"
                      value={modelName}
                      onChange={(e) => setModelName(e.target.value)}
                    />
                  )}
                  {models.length === 0 && !modelsLoading && (
                    <p className="mt-1 font-mono text-[11px] text-[var(--ds-amber-700)]">
                      API unreachable after retries. Check the URL or try again.
                    </p>
                  )}
                </div>

                {/* Ground truth label */}
                <div>
                  <label className="block font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">
                    Ground truth centiloid{" "}
                    <span className="normal-case text-[var(--ds-gray-500)]">(optional)</span>
                  </label>
                  <input
                    className="mt-1 block h-9 w-full rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-3 text-[13px] outline-none transition focus:border-[var(--ds-blue-700)]"
                    placeholder="e.g. 38.5"
                    type="number"
                    step="0.1"
                    value={label}
                    onChange={(e) => setLabel(e.target.value)}
                  />
                </div>

                {/* Run button */}
                <button
                  disabled={!canRun}
                  onClick={handleRun}
                  className={cn(
                    "inline-flex h-10 w-full items-center justify-center gap-2 rounded-[7px] border text-[13px] font-semibold transition",
                    canRun
                      ? "border-[var(--ds-gray-1000)] bg-[var(--ds-gray-1000)] text-[var(--ds-background-100)] hover:border-black hover:bg-black"
                      : "cursor-not-allowed border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] text-[var(--ds-gray-500)]",
                  )}
                >
                  {isRunning ? (
                    <><Loader2 className="h-4 w-4 animate-spin" />Running inference…</>
                  ) : (
                    <><FileUp className="h-4 w-4" />Run inference</>
                  )}
                </button>
              </div>
            </div>

            {/* Last result */}
            {runState.status === "done" && (() => {
              const r = runState.result;
              const labelNum = label ? parseFloat(label) : null;
              const delta = labelNum !== null ? Math.abs(r.centiloid - labelNum) : null;
              return (
                <div className="rounded-[8px] border border-[var(--ds-green-400)] bg-[var(--ds-green-100)]">
                  <div className="flex items-center gap-2 border-b border-[var(--ds-green-400)] px-4 py-3">
                    <CheckCircle2 className="h-4 w-4 text-[var(--ds-green-700)]" />
                    <p className="text-[13px] font-semibold text-[var(--ds-green-900)]">Inference complete</p>
                  </div>
                  <div className="p-4">
                    <CentiloidGauge value={r.centiloid} />
                  </div>
                  <dl className="divide-y divide-[var(--ds-green-400)] text-[13px]">
                    <div className="flex items-center justify-between gap-3 px-4 py-2">
                      <dt className="font-mono text-[11px] uppercase text-[var(--ds-green-700)]">Raw output</dt>
                      <dd className="font-mono font-medium tabular-nums text-[var(--ds-green-900)]">
                        {r.raw_output.toFixed(6)}
                      </dd>
                    </div>
                    {delta !== null && (
                      <div className="flex items-center justify-between gap-3 px-4 py-2">
                        <dt className="font-mono text-[11px] uppercase text-[var(--ds-green-700)]">Δ vs label</dt>
                        <dd className="font-mono font-semibold tabular-nums" style={{ color: deltaColor(delta) }}>
                          ±{delta.toFixed(1)} CL
                        </dd>
                      </div>
                    )}
                    {[
                      ["Model", r.model_name],
                      ["File", r.filename],
                      ...(r.label ? [["Label", `${r.label} CL`]] : []),
                    ].map(([k, v]) => (
                      <div key={k} className="flex items-center justify-between gap-3 px-4 py-2">
                        <dt className="font-mono text-[11px] uppercase text-[var(--ds-green-700)]">{k}</dt>
                        <dd className="min-w-0 truncate text-right font-medium text-[var(--ds-green-900)]">{v}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              );
            })()}

            {/* Error */}
            {runState.status === "error" && (
              <div className="rounded-[8px] border border-[var(--ds-red-400)] bg-[var(--ds-red-100)] p-4">
                <div className="flex items-start gap-2">
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-[var(--ds-red-700)]" />
                  <div>
                    <p className="text-[13px] font-semibold text-[var(--ds-red-900)]">Inference failed</p>
                    <p className="mt-1 font-mono text-[11px] leading-5 text-[var(--ds-red-700)]">
                      {runState.message}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ── NII Viewer ── */}
          <div className="min-h-[540px]">
            <NiiViewer file={file} />
          </div>
        </div>

        {/* ── Results table ── */}
        <div className="rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)]">
          <div className="flex items-center justify-between border-b border-[var(--ds-gray-alpha-300)] px-4 py-3">
            <div>
              <p className="font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">02 / History</p>
              <h2 className="mt-0.5 text-[15px] font-semibold">Inference results</h2>
            </div>
            <div className="flex items-center gap-2">
              {results.length > 0 && (
                <span className="inline-flex h-6 items-center rounded-[5px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-2 font-mono text-[11px] text-[var(--ds-gray-800)]">
                  {results.length} records
                </span>
              )}
              <button
                onClick={loadResults}
                disabled={resultsLoading}
                className="inline-flex h-8 items-center gap-1.5 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2.5 text-[12px] font-medium text-[var(--ds-gray-900)] transition hover:bg-[var(--ds-gray-100)] disabled:opacity-50"
              >
                <RefreshCw className={cn("h-3.5 w-3.5", resultsLoading && "animate-spin")} />
                Refresh
              </button>
            </div>
          </div>

          {resultsError ? (
            <div className="flex items-center gap-2 px-4 py-6 text-[13px] text-[var(--ds-red-700)]">
              <AlertCircle className="h-4 w-4 shrink-0" />
              Cannot reach API: {resultsError}
            </div>
          ) : results.length === 0 && !resultsLoading ? (
            <div className="flex flex-col items-center justify-center gap-2 py-10">
              <p className="text-[13px] text-[var(--ds-gray-700)]">No results yet</p>
              <p className="font-mono text-[11px] text-[var(--ds-gray-600)]">Run your first inference above</p>
            </div>
          ) : (() => {
            type ResultRow = Omit<InferenceResult, "id"> & { id: string };
            const rows: ResultRow[] = results.map((r) => ({ ...r, id: String(r.id) }));
            const columns: TableColumn<ResultRow>[] = [
              {
                key: "id",
                header: "ID",
                render: (r) => <span className="font-mono text-[11px] text-[var(--ds-gray-700)]">#{r.id}</span>,
              },
              {
                key: "filename",
                header: "File",
                render: (r) => (
                  <span className="block max-w-[160px] truncate font-medium" title={r.filename}>
                    {r.filename}
                  </span>
                ),
              },
              {
                key: "model_name",
                header: "Model",
                render: (r) => (
                  <span className="inline-flex h-6 items-center rounded-[5px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-2 font-mono text-[11px]">
                    {r.model_name}
                  </span>
                ),
              },
              {
                key: "centiloid",
                header: "Centiloid",
                render: (r) => <CentiloidCell value={r.centiloid} />,
              },
              {
                key: "raw_output",
                header: "Raw output",
                align: "right",
                render: (r) => (
                  <span className="font-mono text-[11px] text-[var(--ds-gray-700)]">
                    {r.raw_output.toFixed(4)}
                  </span>
                ),
              },
              {
                key: "delta",
                header: "Δ vs label",
                align: "right",
                render: (r) => {
                  const labelNum = r.label ? parseFloat(r.label) : null;
                  const delta = labelNum !== null ? Math.abs(r.centiloid - labelNum) : null;
                  return delta !== null ? (
                    <span className="font-mono text-[12px] font-semibold tabular-nums" style={{ color: deltaColor(delta) }}>
                      ±{delta.toFixed(1)}
                    </span>
                  ) : (
                    <span className="text-[var(--ds-gray-500)]">—</span>
                  );
                },
              },
              {
                key: "label",
                header: "Label",
                render: (r) =>
                  r.label ? (
                    <span className="font-mono text-[12px]">{r.label} CL</span>
                  ) : (
                    <span className="text-[var(--ds-gray-500)]">—</span>
                  ),
              },
              {
                key: "created_at",
                header: "Date",
                render: (r) => (
                  <span className="whitespace-nowrap font-mono text-[11px] text-[var(--ds-gray-700)]">
                    {fmtDate(r.created_at)}
                  </span>
                ),
              },
            ];
            return <DataTable rows={rows} columns={columns} />;
          })()}
        </div>
      </main>
    </div>
  );
}
