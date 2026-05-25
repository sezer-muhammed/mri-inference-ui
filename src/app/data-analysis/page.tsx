"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Activity,
  AlertCircle,
  BarChart2,
  Brain,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Settings2,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { cn } from "@/lib/cn";
import { type InferenceResult, getModels, getResults } from "@/lib/api";
import { DataTable, type TableColumn } from "@/components/ui/data-table";

/* ─── clip ──────────────────────────────────────────────────────────────────
   Both model output and ground-truth labels are clamped to this range for
   all analysis. Values outside [-20, 130] are physiologically implausible
   and distort metrics / charts.
────────────────────────────────────────────────────────────────────────────── */
const CL_MIN = -20;
const CL_MAX = 130;
const clip = (v: number) => Math.max(CL_MIN, Math.min(CL_MAX, v));

/* ─── constants ─────────────────────────────────────────────────────────── */
const DEFAULT_THRESHOLD = 18;
const DEFAULT_BINS = 24;
const MAX_SCATTER = 3000;

/* ─── helpers ────────────────────────────────────────────────────────────── */
function parsedLabel(r: { label: string | null }): number | null {
  if (r.label === null) return null;
  const n = parseFloat(r.label);
  return isNaN(n) ? null : clip(n);
}

function fmtDate(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function buildHistogram(values: number[], bins: number) {
  if (values.length === 0) return [];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const step = (max - min) / bins || 1;
  const result = Array.from({ length: bins }, (_, i) => ({
    bin: (min + i * step).toFixed(0),
    count: 0,
  }));
  for (const v of values) {
    const i = Math.min(Math.floor((v - min) / step), bins - 1);
    result[i].count++;
  }
  return result;
}

function buildDualHistogram(actuals: number[], preds: number[], bins: number) {
  if (actuals.length === 0 && preds.length === 0) return [];
  const all = [...actuals, ...preds];
  const min = Math.min(...all);
  const max = Math.max(...all);
  const step = (max - min) / bins || 1;
  const result = Array.from({ length: bins }, (_, i) => ({
    bin: (min + i * step).toFixed(0),
    actual: 0,
    pred: 0,
  }));
  for (const v of actuals) {
    const i = Math.min(Math.floor((v - min) / step), bins - 1);
    result[i].actual++;
  }
  for (const v of preds) {
    const i = Math.min(Math.floor((v - min) / step), bins - 1);
    result[i].pred++;
  }
  return result;
}

function computeMetrics(
  pairs: { pred: number; actual: number }[],
  threshold: number,
) {
  const n = pairs.length;
  if (n === 0) return null;

  let sumAE = 0, sumSE = 0, sumActual = 0, sumPred = 0;
  for (const { pred, actual } of pairs) {
    sumAE += Math.abs(pred - actual);
    sumSE += (pred - actual) ** 2;
    sumActual += actual;
    sumPred += pred;
  }
  const mae = sumAE / n;
  const mse = sumSE / n;
  const rmse = Math.sqrt(mse);
  const meanActual = sumActual / n;
  const meanPred = sumPred / n;

  let ssTot = 0, cov = 0, varActual = 0, varPred = 0;
  for (const { pred, actual } of pairs) {
    ssTot += (actual - meanActual) ** 2;
    cov += (pred - meanPred) * (actual - meanActual);
    varActual += (actual - meanActual) ** 2;
    varPred += (pred - meanPred) ** 2;
  }
  const r2 = ssTot > 0 ? 1 - sumSE / ssTot : 0;
  const pearson =
    varActual > 0 && varPred > 0 ? cov / Math.sqrt(varActual * varPred) : 0;

  let tp = 0, fp = 0, tn = 0, fn = 0;
  for (const { pred, actual } of pairs) {
    const pp = pred > threshold;
    const ap = actual > threshold;
    if (pp && ap) tp++;
    else if (pp && !ap) fp++;
    else if (!pp && ap) fn++;
    else tn++;
  }
  const accuracy = (tp + tn) / n;
  const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
  const sensitivity = tp + fn > 0 ? tp / (tp + fn) : 0;
  const specificity = tn + fp > 0 ? tn / (tn + fp) : 0;
  const f1 =
    precision + sensitivity > 0
      ? (2 * precision * sensitivity) / (precision + sensitivity)
      : 0;

  return {
    mae, mse, rmse, r2, pearson,
    accuracy, precision, sensitivity, specificity, f1,
    tp, fp, tn, fn, n,
  };
}

/* ─── sub-components ─────────────────────────────────────────────────────── */
function StatusDot({ color, pulse }: { color: string; pulse?: boolean }) {
  return (
    <span
      className={cn(
        "inline-block h-2 w-2 shrink-0 rounded-full border border-[var(--ds-gray-alpha-500)]",
        pulse && "animate-pulse",
      )}
      style={{ background: color }}
    />
  );
}

function MetricCard({
  eyebrow, value, sub, accent = "var(--ds-gray-1000)",
}: {
  eyebrow: string; value: string; sub?: string; accent?: string;
}) {
  return (
    <div className="depth-surface rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] p-4">
      <p className="font-mono text-[11px] uppercase tracking-wide text-[var(--ds-gray-700)]">
        {eyebrow}
      </p>
      <p
        className="mt-2 text-[26px] font-semibold tabular-nums leading-none"
        style={{ color: accent }}
      >
        {value}
      </p>
      {sub && (
        <p className="mt-1.5 font-mono text-[11px] text-[var(--ds-gray-600)]">{sub}</p>
      )}
    </div>
  );
}

function ConfusionMatrix({
  tp, fp, tn, fn, threshold,
}: {
  tp: number; fp: number; tn: number; fn: number; threshold: number;
}) {
  const total = tp + fp + tn + fn;
  const pct = (n: number) =>
    total > 0 ? `${((n / total) * 100).toFixed(1)}%` : "—";

  const cells = [
    { label: "TN", val: tn, pct: pct(tn), correct: true },
    { label: "FP", val: fp, pct: pct(fp), correct: false },
    { label: "FN", val: fn, pct: pct(fn), correct: false },
    { label: "TP", val: tp, pct: pct(tp), correct: true },
  ];

  return (
    <div className="flex flex-col gap-2">
      <div className="grid grid-cols-[88px_1fr_1fr] gap-2 text-center">
        <div />
        <div className="font-mono text-[10px] uppercase leading-tight text-[var(--ds-gray-600)]">
          Pred NEG<br />
          <span className="text-[var(--ds-gray-500)]">(≤{threshold} CL)</span>
        </div>
        <div className="font-mono text-[10px] uppercase leading-tight text-[var(--ds-gray-600)]">
          Pred POS<br />
          <span className="text-[var(--ds-gray-500)]">(&gt;{threshold} CL)</span>
        </div>
      </div>

      {(["NEG", "POS"] as const).map((row, ri) => (
        <div key={row} className="grid grid-cols-[88px_1fr_1fr] items-center gap-2">
          <div className="pr-2 text-right font-mono text-[10px] uppercase text-[var(--ds-gray-600)]">
            Actual {row}
          </div>
          {[cells[ri * 2], cells[ri * 2 + 1]].map((c) => (
            <div
              key={c.label}
              className="rounded-[6px] border p-3 text-center"
              style={{
                background: c.correct ? "var(--ds-green-100)" : "var(--ds-red-100)",
                borderColor: c.correct ? "var(--ds-green-400)" : "var(--ds-red-400)",
              }}
            >
              <p
                className="mb-1 font-mono text-[10px] uppercase"
                style={{ color: c.correct ? "var(--ds-green-700)" : "var(--ds-red-700)" }}
              >
                {c.label}
              </p>
              <p
                className="text-[22px] font-semibold tabular-nums leading-none"
                style={{ color: c.correct ? "var(--ds-green-900)" : "var(--ds-red-900)" }}
              >
                {c.val.toLocaleString()}
              </p>
              <p
                className="mt-1 font-mono text-[10px]"
                style={{ color: c.correct ? "var(--ds-green-700)" : "var(--ds-red-700)" }}
              >
                {c.pct}
              </p>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

/* ─── chart tooltip ─────────────────────────────────────────────────────── */
function ChartTip({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-[6px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] p-2 text-[11px] shadow-sm">
      {children}
    </div>
  );
}

/* ─── page ────────────────────────────────────────────────────────────────── */
export default function DataAnalysisPage() {
  const defaultApi =
    process.env.NEXT_PUBLIC_API_BASE ??
    "https://sezer-muhammed-mri-inference-api.hf.space";
  const [apiBase, setApiBase] = useState(defaultApi);
  const [apiBaseInput, setApiBaseInput] = useState(defaultApi);
  const [showSettings, setShowSettings] = useState(false);

  const [results, setResults] = useState<InferenceResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState("__all__");
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [binCount, setBinCount] = useState(DEFAULT_BINS);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [res, mods] = await Promise.all([
        getResults(apiBase),
        getModels(apiBase).catch(() => [] as string[]),
      ]);
      setResults(res);
      const fromResults = [...new Set(res.map((r) => r.model_name))];
      setModels([...new Set([...mods, ...fromResults])].sort());
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => { loadData(); }, [loadData]);

  /* ─── derived ──────────────────────────────────────────────────────────── */
  const filtered = useMemo(
    () =>
      selectedModel === "__all__"
        ? results
        : results.filter((r) => r.model_name === selectedModel),
    [results, selectedModel],
  );

  const labeled = useMemo(() => {
    const pairs: { pred: number; actual: number; filename: string; model: string }[] = [];
    for (const r of filtered) {
      const actual = parsedLabel(r);
      if (actual !== null)
        pairs.push({ pred: clip(r.centiloid), actual, filename: r.filename, model: r.model_name });
    }
    return pairs;
  }, [filtered]);

  const metrics = useMemo(
    () => computeMetrics(labeled, threshold),
    [labeled, threshold],
  );

  const scatterData = useMemo(() => {
    const data = labeled.map((p) => ({ x: p.actual, y: p.pred, name: p.filename }));
    if (data.length <= MAX_SCATTER) return { data, sampled: false };
    const shuffled = [...data].sort(() => Math.random() - 0.5).slice(0, MAX_SCATTER);
    return { data: shuffled, sampled: true };
  }, [labeled]);

  const scatterDomain: [number, number] = [CL_MIN, CL_MAX];
  const refLineData = [
    { x: CL_MIN, y: CL_MIN },
    { x: CL_MAX, y: CL_MAX },
  ];

  const residuals = useMemo(
    () => labeled.map((p) => p.pred - p.actual),
    [labeled],
  );
  const residualBins = useMemo(
    () => buildHistogram(residuals, binCount),
    [residuals, binCount],
  );
  const distBins = useMemo(
    () =>
      buildDualHistogram(
        labeled.map((p) => p.actual),
        labeled.map((p) => p.pred),
        binCount,
      ),
    [labeled, binCount],
  );

  /* ─── table ─────────────────────────────────────────────────────────────── */
  type Row = Omit<InferenceResult, "id"> & { id: string };
  const tableRows = useMemo<Row[]>(
    () => filtered.map((r) => ({ ...r, id: String(r.id) })),
    [filtered],
  );

  const tableColumns = useMemo((): TableColumn<Row>[] => [
    {
      key: "filename", header: "File",
      render: (r) => (
        <span className="block max-w-[200px] truncate font-medium" title={r.filename}>
          {r.filename}
        </span>
      ),
    },
    {
      key: "model_name", header: "Model",
      render: (r) => (
        <span className="inline-flex h-6 items-center rounded-[5px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-2 font-mono text-[11px]">
          {r.model_name}
        </span>
      ),
    },
    {
      key: "centiloid", header: "Predicted", align: "right",
      render: (r) => (
        <span className="font-mono text-[12px] font-semibold tabular-nums">
          {clip(r.centiloid).toFixed(1)}
        </span>
      ),
    },
    {
      key: "label", header: "Actual (clipped)", align: "right",
      render: (r) => {
        const v = parsedLabel(r);
        return v !== null
          ? <span className="font-mono text-[12px] tabular-nums">{v.toFixed(1)}</span>
          : <span className="text-[var(--ds-gray-500)]">—</span>;
      },
    },
    {
      key: "delta", header: "Δ Error", align: "right",
      render: (r) => {
        const actual = parsedLabel(r);
        if (actual === null) return <span className="text-[var(--ds-gray-500)]">—</span>;
        const delta = Math.abs(clip(r.centiloid) - actual);
        const color =
          delta < 10
            ? "var(--ds-green-700)"
            : delta < 25
            ? "var(--ds-amber-700)"
            : "var(--ds-red-700)";
        return (
          <span className="font-mono text-[12px] font-semibold tabular-nums" style={{ color }}>
            ±{delta.toFixed(1)}
          </span>
        );
      },
    },
    {
      key: "created_at", header: "Date",
      render: (r) => (
        <span className="whitespace-nowrap font-mono text-[11px] text-[var(--ds-gray-700)]">
          {fmtDate(r.created_at)}
        </span>
      ),
    },
  ], []);

  /* ─── reusable section card wrapper ─────────────────────────────────────── */
  function SectionCard({
    eyebrow, title, sub, children, action,
  }: {
    eyebrow: string; title: string; sub?: string;
    children: React.ReactNode; action?: React.ReactNode;
  }) {
    return (
      <div className="rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)]">
        <div className="flex items-start justify-between border-b border-[var(--ds-gray-alpha-300)] px-4 py-3">
          <div>
            <p className="font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">{eyebrow}</p>
            <h2 className="mt-0.5 text-[15px] font-semibold">{title}</h2>
            {sub && <p className="mt-0.5 font-mono text-[11px] text-[var(--ds-amber-700)]">{sub}</p>}
          </div>
          {action}
        </div>
        {children}
      </div>
    );
  }

  /* ─── render ─────────────────────────────────────────────────────────────── */
  return (
    <div className="min-h-screen text-[var(--ds-gray-1000)]">

      {/* Header */}
      <header className="depth-surface sticky top-3 z-20 mx-auto grid min-h-14 w-full max-w-[1440px] grid-cols-[1fr_auto] items-center gap-3 rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="grid h-9 w-9 shrink-0 place-items-center rounded-[7px] border border-[var(--ds-gray-1000)] bg-[var(--ds-gray-1000)]">
            <BarChart2 aria-hidden className="h-4 w-4 text-[var(--ds-background-100)]" />
          </div>
          <div className="min-w-0">
            <p className="truncate text-[14px] font-semibold leading-5">Data Analysis</p>
            <p className="truncate font-mono text-[11px] text-[var(--ds-gray-700)]">
              Centiloid regression · model evaluation · clipped [−20, 130]
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Link
            href="/"
            className="inline-flex h-8 items-center gap-1.5 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2.5 text-[12px] font-medium text-[var(--ds-gray-900)] transition hover:bg-[var(--ds-gray-100)]"
          >
            <Brain aria-hidden className="h-3.5 w-3.5" />
            Inference
          </Link>
          <div className="hidden items-center gap-2 md:flex">
            <StatusDot
              color={
                error
                  ? "var(--ds-red-700)"
                  : loading
                  ? "var(--ds-amber-700)"
                  : "var(--ds-green-700)"
              }
              pulse={loading}
            />
            <span className="font-mono text-[12px] text-[var(--ds-gray-700)]">{apiBase}</span>
          </div>
          <button
            onClick={() => { setApiBaseInput(apiBase); setShowSettings((v) => !v); }}
            className="inline-flex h-8 items-center gap-1.5 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2.5 text-[12px] font-medium text-[var(--ds-gray-900)] transition hover:bg-[var(--ds-gray-100)]"
          >
            <Settings2 aria-hidden className="h-3.5 w-3.5" />
            API
            {showSettings
              ? <ChevronUp aria-hidden className="h-3 w-3" />
              : <ChevronDown aria-hidden className="h-3 w-3" />}
          </button>
        </div>
      </header>

      {/* API settings panel */}
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
                onClick={() => { setApiBase(apiBaseInput.replace(/\/$/, "")); setShowSettings(false); }}
                className="inline-flex h-9 items-center rounded-[7px] border border-[var(--ds-gray-1000)] bg-[var(--ds-gray-1000)] px-3 text-[13px] font-medium text-[var(--ds-background-100)] transition hover:bg-black"
              >
                Save &amp; reconnect
              </button>
            </div>
          </div>
        </div>
      )}

      <main className="mx-auto max-w-[1440px] space-y-4 py-4">

        {/* ── 01 Controls ─────────────────────────────────────────────────── */}
        <SectionCard
          eyebrow="01 / Controls"
          title="Model & parameters"
          action={
            <button
              onClick={loadData}
              disabled={loading}
              className="inline-flex h-8 items-center gap-1.5 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2.5 text-[12px] font-medium text-[var(--ds-gray-900)] transition hover:bg-[var(--ds-gray-100)] disabled:opacity-50"
            >
              <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
              Refresh
            </button>
          }
        >
          <div className="space-y-4 p-4">
            {/* Model pills */}
            <div>
              <p className="mb-2 font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">Model</p>
              <div className="flex flex-wrap gap-2">
                {["__all__", ...models].map((m) => {
                  const count =
                    m === "__all__"
                      ? results.length
                      : results.filter((r) => r.model_name === m).length;
                  return (
                    <button
                      key={m}
                      onClick={() => setSelectedModel(m)}
                      className={cn(
                        "inline-flex h-8 items-center gap-1.5 rounded-[7px] border px-3 text-[12px] font-medium transition",
                        selectedModel === m
                          ? "border-[var(--ds-gray-1000)] bg-[var(--ds-gray-1000)] text-[var(--ds-background-100)]"
                          : "border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] text-[var(--ds-gray-900)] hover:bg-[var(--ds-gray-100)]",
                      )}
                    >
                      {m === "__all__" ? "All models" : m}
                      <span
                        className={cn(
                          "inline-flex h-4 min-w-[20px] items-center justify-center rounded-full px-1 font-mono text-[10px]",
                          selectedModel === m
                            ? "bg-[var(--ds-gray-700)] text-[var(--ds-background-100)]"
                            : "bg-[var(--ds-gray-200)] text-[var(--ds-gray-800)]",
                        )}
                      >
                        {count.toLocaleString()}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Sliders */}
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <p className="font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">
                    Classification threshold
                  </p>
                  <span className="font-mono text-[13px] font-semibold tabular-nums text-[var(--ds-gray-900)]">
                    {threshold} CL
                  </span>
                </div>
                <input
                  type="range" min={0} max={60} step={1}
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                  className="w-full accent-[var(--ds-gray-1000)]"
                />
                <div className="mt-1 flex justify-between font-mono text-[10px] text-[var(--ds-gray-500)]">
                  <span>0</span><span>15</span><span>30</span><span>45</span><span>60</span>
                </div>
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between">
                  <p className="font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">
                    Histogram bins
                  </p>
                  <span className="font-mono text-[13px] font-semibold tabular-nums text-[var(--ds-gray-900)]">
                    {binCount}
                  </span>
                </div>
                <input
                  type="range" min={8} max={50} step={2}
                  value={binCount}
                  onChange={(e) => setBinCount(Number(e.target.value))}
                  className="w-full accent-[var(--ds-gray-1000)]"
                />
                <div className="mt-1 flex justify-between font-mono text-[10px] text-[var(--ds-gray-500)]">
                  <span>8</span><span>20</span><span>32</span><span>44</span>
                </div>
              </div>
            </div>

            {/* Summary chips */}
            <div className="flex flex-wrap gap-2">
              {[
                { label: "Total", value: filtered.length.toLocaleString() },
                { label: "Labeled", value: labeled.length.toLocaleString() },
                {
                  label: "Coverage",
                  value:
                    filtered.length > 0
                      ? `${((labeled.length / filtered.length) * 100).toFixed(1)}%`
                      : "—",
                },
                { label: "Models", value: models.length.toString() },
                { label: "Clip range", value: "[−20, 130] CL" },
              ].map((s) => (
                <div
                  key={s.label}
                  className="rounded-[6px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-3 py-1.5"
                >
                  <p className="font-mono text-[10px] uppercase text-[var(--ds-gray-600)]">{s.label}</p>
                  <p className="font-mono text-[13px] font-semibold tabular-nums text-[var(--ds-gray-1000)]">
                    {s.value}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </SectionCard>

        {error && (
          <div className="flex items-center gap-2 rounded-[8px] border border-[var(--ds-red-400)] bg-[var(--ds-red-100)] px-4 py-3 text-[13px] text-[var(--ds-red-700)]">
            <AlertCircle className="h-4 w-4 shrink-0" />
            Cannot reach API: {error}
          </div>
        )}

        {/* ── 02 Regression metrics ────────────────────────────────────────── */}
        {metrics && (
          <>
            <SectionCard
              eyebrow="02 / Regression metrics"
              title={`Evaluated on ${metrics.n.toLocaleString()} labeled pairs`}
            >
              <div className="grid grid-cols-2 gap-3 p-4 sm:grid-cols-4">
                <MetricCard
                  eyebrow="MAE"
                  value={`${metrics.mae.toFixed(2)} CL`}
                  sub="Mean absolute error"
                  accent="var(--ds-blue-700)"
                />
                <MetricCard
                  eyebrow="RMSE"
                  value={`${metrics.rmse.toFixed(2)} CL`}
                  sub="Root mean squared error"
                  accent="var(--ds-blue-700)"
                />
                <MetricCard
                  eyebrow="R²"
                  value={metrics.r2.toFixed(4)}
                  sub="Coefficient of determination"
                  accent={
                    metrics.r2 >= 0.8
                      ? "var(--ds-green-700)"
                      : metrics.r2 >= 0.6
                      ? "var(--ds-amber-700)"
                      : "var(--ds-red-700)"
                  }
                />
                <MetricCard
                  eyebrow="Pearson r"
                  value={metrics.pearson.toFixed(4)}
                  sub="Linear correlation"
                  accent={
                    metrics.pearson >= 0.9
                      ? "var(--ds-green-700)"
                      : metrics.pearson >= 0.7
                      ? "var(--ds-amber-700)"
                      : "var(--ds-red-700)"
                  }
                />
              </div>
            </SectionCard>

            {/* ── 03 Classification metrics ─────────────────────────────────── */}
            <SectionCard
              eyebrow="03 / Classification metrics"
              title={`Binary at threshold ${threshold} CL`}
            >
              <div className="grid grid-cols-2 gap-3 p-4 sm:grid-cols-4">
                <MetricCard
                  eyebrow="Accuracy"
                  value={`${(metrics.accuracy * 100).toFixed(1)}%`}
                  sub={`${(metrics.tp + metrics.tn).toLocaleString()} / ${metrics.n.toLocaleString()} correct`}
                  accent={
                    metrics.accuracy >= 0.85
                      ? "var(--ds-green-700)"
                      : metrics.accuracy >= 0.70
                      ? "var(--ds-amber-700)"
                      : "var(--ds-red-700)"
                  }
                />
                <MetricCard
                  eyebrow="F1 Score"
                  value={metrics.f1.toFixed(4)}
                  sub="Harmonic mean prec + rec"
                  accent={
                    metrics.f1 >= 0.85
                      ? "var(--ds-green-700)"
                      : metrics.f1 >= 0.70
                      ? "var(--ds-amber-700)"
                      : "var(--ds-red-700)"
                  }
                />
                <MetricCard
                  eyebrow="Sensitivity"
                  value={`${(metrics.sensitivity * 100).toFixed(1)}%`}
                  sub="True positive rate (recall)"
                  accent={
                    metrics.sensitivity >= 0.85
                      ? "var(--ds-green-700)"
                      : metrics.sensitivity >= 0.70
                      ? "var(--ds-amber-700)"
                      : "var(--ds-red-700)"
                  }
                />
                <MetricCard
                  eyebrow="Specificity"
                  value={`${(metrics.specificity * 100).toFixed(1)}%`}
                  sub="True negative rate"
                  accent={
                    metrics.specificity >= 0.85
                      ? "var(--ds-green-700)"
                      : metrics.specificity >= 0.70
                      ? "var(--ds-amber-700)"
                      : "var(--ds-red-700)"
                  }
                />
              </div>
            </SectionCard>
          </>
        )}

        {/* ── Charts 2×2 ──────────────────────────────────────────────────── */}
        {labeled.length > 0 && (
          <div className="grid gap-4 xl:grid-cols-2">

            {/* 04 Scatter */}
            <SectionCard
              eyebrow="04 / Scatter"
              title="Predicted vs Actual"
              sub={
                scatterData.sampled
                  ? `Showing ${MAX_SCATTER.toLocaleString()} random samples of ${labeled.length.toLocaleString()}`
                  : undefined
              }
            >
              <div className="p-4">
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart margin={{ top: 8, right: 20, bottom: 28, left: 8 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--ds-gray-200)" />
                    <XAxis
                      dataKey="x" type="number" name="Actual"
                      domain={scatterDomain}
                      tick={{ fontSize: 11, fontFamily: "monospace" }}
                      label={{ value: "Actual CL", position: "insideBottom", offset: -18, fontSize: 11 }}
                    />
                    <YAxis
                      dataKey="y" type="number" name="Predicted"
                      domain={scatterDomain}
                      tick={{ fontSize: 11, fontFamily: "monospace" }}
                      label={{ value: "Predicted CL", angle: -90, position: "insideLeft", offset: 14, fontSize: 11 }}
                    />
                    <Tooltip
                      cursor={{ strokeDasharray: "3 3" }}
                      content={({ active, payload }) => {
                        if (!active || !payload?.[0]) return null;
                        const d = payload[0].payload;
                        return (
                          <ChartTip>
                            <p className="max-w-[160px] truncate font-mono text-[var(--ds-gray-700)]">{d.name}</p>
                            <p className="mt-1">Actual: <strong>{Number(d.x).toFixed(1)}</strong></p>
                            <p>Predicted: <strong>{Number(d.y).toFixed(1)}</strong></p>
                            <p>Error: <strong>±{Math.abs(Number(d.x) - Number(d.y)).toFixed(1)}</strong></p>
                          </ChartTip>
                        );
                      }}
                    />
                    {/* Identity line (y = x) */}
                    <Scatter
                      name="y=x"
                      data={refLineData}
                      line={{ stroke: "var(--ds-blue-700)", strokeDasharray: "5 4", strokeWidth: 1.5 }}
                      shape={() => <g />}
                      legendType="none"
                    />
                    <Scatter
                      name="Results"
                      data={scatterData.data}
                      fill="var(--ds-blue-700)"
                      fillOpacity={0.5}
                      r={3}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            </SectionCard>

            {/* 05 Confusion matrix */}
            {metrics && (
              <SectionCard
                eyebrow="05 / Confusion matrix"
                title={`Binary classification at ${threshold} CL`}
              >
                <div className="flex items-center justify-center p-6">
                  <ConfusionMatrix
                    tp={metrics.tp} fp={metrics.fp}
                    tn={metrics.tn} fn={metrics.fn}
                    threshold={threshold}
                  />
                </div>
              </SectionCard>
            )}

            {/* 06 Residuals */}
            <SectionCard
              eyebrow="06 / Residuals"
              title="Error distribution (predicted − actual)"
            >
              <div className="p-4">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart
                    data={residualBins}
                    margin={{ top: 8, right: 20, bottom: 28, left: 8 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--ds-gray-200)" vertical={false} />
                    <XAxis
                      dataKey="bin"
                      tick={{ fontSize: 10, fontFamily: "monospace" }}
                      label={{ value: "Error (CL)", position: "insideBottom", offset: -18, fontSize: 11 }}
                      interval="preserveStartEnd"
                    />
                    <YAxis tick={{ fontSize: 11, fontFamily: "monospace" }} />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.[0]) return null;
                        return (
                          <ChartTip>
                            <p>Bin: <strong>{payload[0].payload.bin}</strong></p>
                            <p>Count: <strong>{payload[0].value}</strong></p>
                          </ChartTip>
                        );
                      }}
                    />
                    <Bar
                      dataKey="count"
                      fill="var(--ds-blue-700)"
                      fillOpacity={0.75}
                      radius={[2, 2, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </SectionCard>

            {/* 07 Distribution */}
            <SectionCard
              eyebrow="07 / Distribution"
              title="Predicted vs actual centiloid distribution"
            >
              <div className="p-4">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart
                    data={distBins}
                    margin={{ top: 8, right: 20, bottom: 28, left: 8 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--ds-gray-200)" vertical={false} />
                    <XAxis
                      dataKey="bin"
                      tick={{ fontSize: 10, fontFamily: "monospace" }}
                      label={{ value: "Centiloid", position: "insideBottom", offset: -18, fontSize: 11 }}
                      interval="preserveStartEnd"
                    />
                    <YAxis tick={{ fontSize: 11, fontFamily: "monospace" }} />
                    <Legend
                      wrapperStyle={{ fontSize: 11, fontFamily: "monospace", paddingTop: 8 }}
                      formatter={(v) => (v === "actual" ? "Ground truth" : "Predicted")}
                    />
                    <Tooltip
                      content={({ active, payload }) => {
                        if (!active || !payload?.length) return null;
                        return (
                          <ChartTip>
                            <p>Bin: <strong>{payload[0]?.payload?.bin}</strong></p>
                            {payload.map((p) => (
                              <p key={p.name} style={{ color: p.color as string }}>
                                {p.name === "actual" ? "Ground truth" : "Predicted"}:{" "}
                                <strong>{p.value}</strong>
                              </p>
                            ))}
                          </ChartTip>
                        );
                      }}
                    />
                    <Bar dataKey="actual" fill="var(--ds-green-700)" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
                    <Bar dataKey="pred" fill="var(--ds-blue-700)" fillOpacity={0.7} radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </SectionCard>
          </div>
        )}

        {/* Empty states */}
        {!loading && labeled.length === 0 && filtered.length > 0 && (
          <div className="rounded-[8px] border border-[var(--ds-amber-400)] bg-[var(--ds-amber-100)] px-4 py-6 text-center">
            <p className="text-[13px] font-semibold text-[var(--ds-amber-900)]">No labeled results found</p>
            <p className="mt-1 font-mono text-[11px] text-[var(--ds-amber-700)]">
              Run inference with ground-truth labels to see metrics and charts.
            </p>
          </div>
        )}

        {!loading && results.length === 0 && !error && (
          <div className="flex flex-col items-center justify-center gap-3 rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] py-16">
            <Activity className="h-8 w-8 text-[var(--ds-gray-400)]" />
            <p className="text-[13px] font-medium text-[var(--ds-gray-700)]">No inference results yet</p>
            <p className="font-mono text-[11px] text-[var(--ds-gray-600)]">
              Run inferences from the main page first
            </p>
            <Link
              href="/"
              className="mt-1 inline-flex h-8 items-center gap-1.5 rounded-[7px] border border-[var(--ds-gray-1000)] bg-[var(--ds-gray-1000)] px-3 text-[12px] font-medium text-[var(--ds-background-100)] transition hover:bg-black"
            >
              <Brain className="h-3.5 w-3.5" />
              Go to inference
            </Link>
          </div>
        )}

        {/* ── 08 Results table ─────────────────────────────────────────────── */}
        {filtered.length > 0 && (
          <SectionCard
            eyebrow="08 / Results"
            title="All inference records"
            action={
              <span className="inline-flex h-6 items-center rounded-[5px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-2 font-mono text-[11px] text-[var(--ds-gray-800)]">
                {filtered.length.toLocaleString()} records
              </span>
            }
          >
            <DataTable rows={tableRows} columns={tableColumns} />
          </SectionCard>
        )}
      </main>
    </div>
  );
}
