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
  Loader2,
  RefreshCw,
  Settings2,
  Shuffle,
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
import { type InferenceResult, getFolds, getModels, getResults } from "@/lib/api";
import { DataTable, type TableColumn } from "@/components/ui/data-table";

/* ─── clip ───────────────────────────────────────────────────────────────────
   Both model output and ground-truth labels are clamped to this range for
   all analysis. Values outside [-20, 130] CL are physiologically implausible.
──────────────────────────────────────────────────────────────────────────── */
const CL_MIN = -20;
const CL_MAX = 130;
const clip = (v: number) => Math.max(CL_MIN, Math.min(CL_MAX, v));

const DEFAULT_THRESHOLDS = [18];
const DEFAULT_BINS = 24;
const MAX_SCATTER = 300;

/* ─── types ──────────────────────────────────────────────────────────────── */
interface LabeledPair {
  pred: number;
  actual: number;
  filename: string;
  model: string;
}

interface FusedResult {
  filename: string;
  pred: number;
  actual: number | null;
  modelPreds: Record<string, number>;
}

/* ─── pure helpers ───────────────────────────────────────────────────────── */
function parsedLabel(r: { label: string | null }): number | null {
  if (!r.label) return null;
  const n = parseFloat(r.label);
  return isNaN(n) ? null : clip(n);
}

function fmtDate(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

function buildHistogram(values: number[], bins: number) {
  if (!values.length) return [];
  const min = Math.min(...values), max = Math.max(...values);
  const step = (max - min) / bins || 1;
  const result = Array.from({ length: bins }, (_, i) => ({
    bin: (min + i * step).toFixed(0), count: 0,
  }));
  for (const v of values) result[Math.min(Math.floor((v - min) / step), bins - 1)].count++;
  return result;
}

function buildDualHistogram(actuals: number[], preds: number[], bins: number) {
  if (!actuals.length && !preds.length) return [];
  const all = [...actuals, ...preds];
  const min = Math.min(...all), max = Math.max(...all);
  const step = (max - min) / bins || 1;
  const result = Array.from({ length: bins }, (_, i) => ({
    bin: (min + i * step).toFixed(0), actual: 0, pred: 0,
  }));
  for (const v of actuals) result[Math.min(Math.floor((v - min) / step), bins - 1)].actual++;
  for (const v of preds)   result[Math.min(Math.floor((v - min) / step), bins - 1)].pred++;
  return result;
}

function classIndex(value: number, thresholds: number[]) {
  for (let i = 0; i < thresholds.length; i++) {
    if (value <= thresholds[i]) return i;
  }
  return thresholds.length;
}

function computeMetrics(pairs: LabeledPair[], thresholds: number[]) {
  const n = pairs.length;
  if (!n) return null;

  let sumAE = 0, sumSE = 0, sumActual = 0, sumPred = 0;
  for (const { pred, actual } of pairs) {
    sumAE += Math.abs(pred - actual);
    sumSE += (pred - actual) ** 2;
    sumActual += actual;
    sumPred += pred;
  }
  const mae = sumAE / n, mse = sumSE / n, rmse = Math.sqrt(mse);
  const mA = sumActual / n, mP = sumPred / n;
  let ssTot = 0, cov = 0, vA = 0, vP = 0;
  for (const { pred, actual } of pairs) {
    ssTot += (actual - mA) ** 2;
    cov   += (pred - mP) * (actual - mA);
    vA    += (actual - mA) ** 2;
    vP    += (pred - mP) ** 2;
  }
  const r2 = ssTot > 0 ? 1 - sumSE / ssTot : 0;
  const pearson = vA > 0 && vP > 0 ? cov / Math.sqrt(vA * vP) : 0;

  const classCount = thresholds.length + 1;
  const confusion = Array.from({ length: classCount }, () => new Array(classCount).fill(0));
  let correct = 0;
  for (const { pred, actual } of pairs) {
    const actualClass = classIndex(actual, thresholds);
    const predClass = classIndex(pred, thresholds);
    confusion[actualClass][predClass]++;
    if (actualClass === predClass) correct++;
  }
  const accuracy = correct / n;
  const classF1 = confusion.map((row, i) => {
    const tp = row[i];
    const fp = confusion.reduce((sum, r, ri) => sum + (ri === i ? 0 : r[i]), 0);
    const fn = row.reduce((sum, v, pi) => sum + (pi === i ? 0 : v), 0);
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    return precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  });
  const macroF1 = classF1.reduce((a, b) => a + b, 0) / classCount;
  return { mae, mse, rmse, r2, pearson, accuracy, correct, classF1, macroF1, confusion, classCount, n };
}

/* ─── fusion helpers ─────────────────────────────────────────────────────── */

/** Project a vector onto the probability simplex (sum=1, x≥0). Duchi et al. 2008 */
function projectSimplex(v: number[]): number[] {
  const n = v.length;
  const u = [...v].sort((a, b) => b - a);
  let cumsum = 0, rho = 0;
  for (let i = 0; i < n; i++) {
    cumsum += u[i];
    if (u[i] - (cumsum - 1) / (i + 1) > 0) rho = i;
  }
  const theta = (u.slice(0, rho + 1).reduce((a, b) => a + b, 0) - 1) / (rho + 1);
  return v.map(x => Math.max(x - theta, 0));
}

function optimizeWeights(
  results: InferenceResult[],
  selModels: string[],
  target: "mae" | "rmse",
): number[] {
  const n = selModels.length;

  // Build labeled sample matrix [filename → {preds[n], actual}]
  const byFile = new Map<string, { preds: number[]; actual: number }>();
  for (const r of results) {
    const mi = selModels.indexOf(r.model_name);
    if (mi === -1) continue;
    const actual = parsedLabel(r);
    if (actual === null) continue;
    if (!byFile.has(r.filename))
      byFile.set(r.filename, { preds: new Array(n).fill(NaN), actual });
    byFile.get(r.filename)!.preds[mi] = clip(r.centiloid);
  }
  const samples = [...byFile.values()].filter(s => s.preds.every(p => !isNaN(p)));
  if (!samples.length) return selModels.map(() => 1 / n);

  const loss = (w: number[]) => {
    const s = w.reduce((a, b) => a + b, 0) || 1;
    const nw = w.map(x => x / s);
    let L = 0;
    for (const { preds, actual } of samples) {
      const e = nw.reduce((acc, wi, i) => acc + wi * preds[i], 0) - actual;
      L += target === "mae" ? Math.abs(e) : e * e;
    }
    return L / samples.length;
  };

  // Exact grid search for 2 models
  if (n === 2) {
    let best = Infinity, bestW = [0.5, 0.5];
    for (let i = 0; i <= 200; i++) {
      const w0 = i / 200, L = loss([w0, 1 - w0]);
      if (L < best) { best = L; bestW = [w0, 1 - w0]; }
    }
    return bestW;
  }

  // Projected gradient descent for N > 2
  let w = new Array(n).fill(1 / n);
  const eps = 1e-5;
  for (let it = 0; it < 800; it++) {
    const lr = 0.05 / (1 + it * 0.008);
    const g = w.map((_, i) => {
      const wp = [...w]; wp[i] = Math.min(1, w[i] + eps);
      const wm = [...w]; wm[i] = Math.max(0, w[i] - eps);
      return (loss(wp) - loss(wm)) / (2 * eps);
    });
    w = projectSimplex(w.map((wi, i) => wi - lr * g[i]));
  }
  return w;
}

function computeFused(
  results: InferenceResult[],
  selModels: string[],
  normW: Record<string, number>,
): FusedResult[] {
  if (selModels.length < 2) return [];
  const byFile = new Map<string, { preds: Map<string, number>; label: string | null }>();
  for (const r of results) {
    if (!selModels.includes(r.model_name)) continue;
    if (!byFile.has(r.filename))
      byFile.set(r.filename, { preds: new Map(), label: r.label });
    const e = byFile.get(r.filename)!;
    e.preds.set(r.model_name, clip(r.centiloid));
    if (r.label) e.label = r.label;
  }
  const out: FusedResult[] = [];
  for (const [filename, { preds, label }] of byFile) {
    if (!selModels.every(m => preds.has(m))) continue;
    const pred = clip(selModels.reduce((s, m) => s + (normW[m] ?? 0) * preds.get(m)!, 0));
    const lv = label ? parseFloat(label) : NaN;
    const actual = isNaN(lv) ? null : clip(lv);
    const modelPreds: Record<string, number> = {};
    selModels.forEach(m => { modelPreds[m] = preds.get(m)!; });
    out.push({ filename, pred, actual, modelPreds });
  }
  return out;
}

/* ─── sub-components ─────────────────────────────────────────────────────── */
function metricTone(good: boolean, mid: boolean) {
  if (good) return "var(--ds-green-700)";
  if (mid) return "var(--ds-amber-700)";
  return "var(--ds-red-700)";
}

function errorTone(error: number) {
  return metricTone(error < 10, error < 25);
}

function scatterErrorColor(error: number) {
  const ratio = Math.max(0, Math.min(1, error / 70));
  const hue = 145 - ratio * 145;
  const lightness = 38 + ratio * 4;
  return `hsl(${hue} 72% ${lightness}%)`;
}

function diagonalDistanceColor(distance: number, maxDistance: number) {
  const ratio = maxDistance > 0 ? Math.max(0, Math.min(1, distance / maxDistance)) : 0;
  const hue = 145 - ratio * 145;
  const lightness = 38 + ratio * 4;
  return `hsl(${hue} 72% ${lightness}%)`;
}

function MetricDot({ color }: { color: string }) {
  return (
    <span
      className="mt-0.5 h-2.5 w-2.5 shrink-0 rounded-full border border-black shadow-[inset_0_1px_0_rgb(255_255_255_/_0.55)]"
      style={{ background: color }}
    />
  );
}

function ErrorValue({ value }: { value: number }) {
  return (
    <span className="inline-flex items-center justify-end gap-2 font-mono text-[12px] font-semibold tabular-nums text-[var(--ds-gray-1000)]">
      <MetricDot color={errorTone(value)} />
      +/-{value.toFixed(1)}
    </span>
  );
}

function ScatterPoint(props: any) {
  const { cx, cy, payload } = props;
  if (typeof cx !== "number" || typeof cy !== "number") return <g />;
  const error = Math.abs(Number(payload.x) - Number(payload.y));
  return (
    <circle
      cx={cx}
      cy={cy}
      r={4.5}
      fill={scatterErrorColor(error)}
      stroke="var(--ds-background-100)"
      strokeWidth={0.75}
      opacity={0.78}
    />
  );
}

function StatusDot({ color, pulse }: { color: string; pulse?: boolean }) {
  return (
    <span
      className={cn("inline-block h-2 w-2 shrink-0 rounded-full border border-[var(--ds-gray-alpha-500)]", pulse && "animate-pulse")}
      style={{ background: color }}
    />
  );
}

function MetricCard({ eyebrow, value, sub, tone = "var(--ds-gray-1000)" }: {
  eyebrow: string; value: string; sub?: string; tone?: string;
}) {
  return (
    <div className="depth-surface rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] p-4">
      <div className="flex items-center gap-2">
        <MetricDot color={tone} />
        <p className="font-mono text-[11px] uppercase tracking-normal text-[var(--ds-gray-700)]">{eyebrow}</p>
      </div>
      <p className="mt-2 text-[28px] font-semibold tabular-nums leading-8 text-[var(--ds-gray-1000)]">{value}</p>
      {sub && <p className="mt-1 text-[12px] leading-5 text-[var(--ds-gray-700)]">{sub}</p>}
    </div>
  );
}

function classLabel(index: number, thresholds: number[]) {
  if (thresholds.length === 0) return "All";
  if (index === 0) return `<=${thresholds[0]}`;
  if (index === thresholds.length) return `>${thresholds[thresholds.length - 1]}`;
  return `${thresholds[index - 1] + 1}-${thresholds[index]}`;
}

function ConfusionMatrix({ confusion, thresholds }: {
  confusion: number[][]; thresholds: number[];
}) {
  const total = confusion.flat().reduce((a, b) => a + b, 0);
  const pct = (x: number) => total > 0 ? `${((x / total) * 100).toFixed(1)}%` : "—";
  const size = confusion.length;
  return (
    <div className="flex flex-col gap-2">
      <div className="grid gap-2 text-center" style={{ gridTemplateColumns: `88px repeat(${size}, minmax(74px, 1fr))` }}>
        <div />
        {confusion.map((_, i) => (
          <div key={i} className="font-mono text-[10px] uppercase leading-tight text-[var(--ds-gray-600)]">
            Pred C{i + 1}<br />
            <span className="text-[var(--ds-gray-500)]">{classLabel(i, thresholds)}</span>
          </div>
        ))}
      </div>
      {confusion.map((row, actualIndex) => (
        <div key={actualIndex} className="grid items-center gap-2" style={{ gridTemplateColumns: `88px repeat(${size}, minmax(74px, 1fr))` }}>
          <div className="pr-2 text-right font-mono text-[10px] uppercase text-[var(--ds-gray-600)]">Actual C{actualIndex + 1}</div>
          {row.map((val, predIndex) => {
            const distance = Math.abs(actualIndex - predIndex);
            const cellColor = diagonalDistanceColor(distance, size - 1);
            const lightRatio = size > 1 ? Math.max(0, Math.min(1, distance / (size - 1))) : 0;
            return (
              <div key={predIndex} className="rounded-[6px] border p-3 text-center"
                style={{
                  background: `color-mix(in srgb, ${cellColor} ${18 + lightRatio * 22}%, var(--ds-background-100))`,
                  borderColor: cellColor,
                }}>
                <p className="mb-1 font-mono text-[10px] uppercase text-[var(--ds-gray-900)]">{distance === 0 ? "OK" : `D${distance}`}</p>
                <p className="text-[22px] font-semibold tabular-nums leading-none text-[var(--ds-gray-1000)]">{val.toLocaleString()}</p>
                <p className="mt-1 font-mono text-[10px] text-[var(--ds-gray-800)]">{pct(val)}</p>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

function ThresholdControls({
  thresholds,
  setThresholds,
}: {
  thresholds: number[];
  setThresholds: (value: number[]) => void;
}) {
  const sorted = [...thresholds].sort((a, b) => a - b);
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <p className="font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">Classification cutpoints</p>
        <button
          className="inline-flex h-7 items-center rounded-[6px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2 text-[12px] font-medium text-[var(--ds-gray-900)] transition hover:bg-[var(--ds-gray-100)] disabled:opacity-50"
          disabled={sorted.length >= 3}
          onClick={() => setThresholds([...sorted, sorted.at(-1) !== undefined ? Math.min(sorted.at(-1)! + 18, CL_MAX) : 18])}
          type="button"
        >
          Add point
        </button>
      </div>
      <div className="space-y-3">
        {sorted.map((value, index) => (
          <div key={index}>
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="font-mono text-[11px] text-[var(--ds-gray-700)]">Point {index + 1}</span>
              <div className="flex items-center gap-2">
                <span className="font-mono text-[13px] font-semibold tabular-nums text-[var(--ds-gray-900)]">{value} CL</span>
                {sorted.length > 1 && (
                  <button
                    className="rounded-[5px] px-1.5 py-1 text-[11px] text-[var(--ds-gray-700)] hover:bg-[var(--ds-gray-100)]"
                    onClick={() => setThresholds(sorted.filter((_, i) => i !== index))}
                    type="button"
                  >
                    Remove
                  </button>
                )}
              </div>
            </div>
            <input
              type="range"
              min={CL_MIN}
              max={CL_MAX}
              step={1}
              value={value}
              onChange={e => {
                const next = [...sorted];
                next[index] = Number(e.target.value);
                setThresholds([...new Set(next)].sort((a, b) => a - b));
              }}
              className="w-full accent-[var(--ds-gray-1000)]"
            />
          </div>
        ))}
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-[var(--ds-gray-500)]">
        <span>-20</span><span>0</span><span>50</span><span>100</span><span>130</span>
      </div>
    </div>
  );
}

function ChartTip({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-[6px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] p-2 text-[11px] shadow-sm">
      {children}
    </div>
  );
}

function SectionCard({ eyebrow, title, sub, children, action }: {
  eyebrow: string; title: string; sub?: string;
  children: React.ReactNode; action?: React.ReactNode;
}) {
  return (
    <section className="overflow-hidden rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)]">
      <div className="flex flex-col gap-3 border-b border-[var(--ds-gray-alpha-400)] p-4 sm:flex-row sm:items-end sm:justify-between">
        <div className="min-w-0">
          <p className="mb-1 font-mono text-[11px] uppercase tracking-normal text-[var(--ds-gray-700)]">{eyebrow}</p>
          <h2 className="text-[18px] font-semibold leading-6 text-[var(--ds-gray-1000)]">{title}</h2>
          {sub && <p className="mt-1 max-w-2xl text-[13px] leading-5 text-[var(--ds-gray-900)]">{sub}</p>}
        </div>
        {action ? <div className="flex shrink-0 items-center gap-2">{action}</div> : null}
      </div>
      {children}
    </section>
  );
}

/* ─── page ────────────────────────────────────────────────────────────────── */
export default function DataAnalysisPage() {
  const defaultApi = process.env.NEXT_PUBLIC_API_BASE ?? "https://sezer-muhammed-mri-inference-api.hf.space";
  const [apiBase, setApiBase] = useState(defaultApi);
  const [apiBaseInput, setApiBaseInput] = useState(defaultApi);
  const [showSettings, setShowSettings] = useState(false);

  const [results, setResults] = useState<InferenceResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>([]);
  const [availableFolds, setAvailableFolds] = useState<number[]>([]);

  // ── analysis controls
  const [selectedModel, setSelectedModel] = useState("__all__");
  const [selectedFold, setSelectedFold] = useState<number | "__all__">("__all__");
  const [thresholds, setThresholds] = useState<number[]>(DEFAULT_THRESHOLDS);
  const [binCount, setBinCount] = useState(DEFAULT_BINS);
  const [tablePage, setTablePage] = useState(0);

  // ── fusion state
  const [fusionModels, setFusionModels] = useState<string[]>([]);
  const [fusionWeights, setFusionWeights] = useState<Record<string, number>>({});
  const [fusionOptTarget, setFusionOptTarget] = useState<"mae" | "rmse">("mae");
  const [isOptimizing, setIsOptimizing] = useState(false);

  const fusionActive = fusionModels.length >= 2;

  // Keep weights in sync when fusionModels changes
  useEffect(() => {
    setFusionWeights(prev => {
      const next: Record<string, number> = {};
      for (const m of fusionModels) next[m] = prev[m] ?? 50;
      return next;
    });
  }, [fusionModels]);

  // If selected model disappears (e.g., fused was deactivated), reset
  useEffect(() => {
    if (selectedModel === "__fused__" && !fusionActive) setSelectedModel("__all__");
  }, [fusionActive, selectedModel]);

  /* ── data loading ─────────────────────────────────────────────────────── */
  const loadData = useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const [res, mods] = await Promise.all([
        getResults(apiBase),
        getModels(apiBase).catch(() => [] as string[]),
      ]);
      setResults(res);
      const fromResults = [...new Set(res.map(r => r.model_name))];
      setModels([...new Set([...mods, ...fromResults])].sort());
      const folds = await getFolds(apiBase).catch(() => (
        [...new Set(res.map(r => r.fold).filter((f): f is number => typeof f === "number"))]
      ));
      setAvailableFolds(folds.sort((a, b) => a - b));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [apiBase]);

  useEffect(() => { loadData(); }, [loadData]);
  useEffect(() => { setTablePage(0); }, [selectedModel, selectedFold, fusionModels, fusionWeights]);

  const foldResults = useMemo(
    () => selectedFold === "__all__" ? results : results.filter(r => r.fold === selectedFold),
    [results, selectedFold],
  );

  /* ── normalized weights ────────────────────────────────────────────────── */
  const normWeights = useMemo<Record<string, number>>(() => {
    const total = fusionModels.reduce((s, m) => s + (fusionWeights[m] ?? 50), 0) || 1;
    return Object.fromEntries(fusionModels.map(m => [m, (fusionWeights[m] ?? 50) / total]));
  }, [fusionModels, fusionWeights]);

  /* ── fused dataset ─────────────────────────────────────────────────────── */
  const fusedData = useMemo(
    () => computeFused(foldResults, fusionModels, normWeights),
    [foldResults, fusionModels, normWeights],
  );

  /* ── labeled pairs for analysis ────────────────────────────────────────── */
  const labeled = useMemo((): LabeledPair[] => {
    if (selectedModel === "__fused__") {
      return fusedData
        .filter(f => f.actual !== null)
        .map(f => ({ pred: f.pred, actual: f.actual!, filename: f.filename, model: "Fused" }));
    }
    const src = selectedModel === "__all__"
      ? foldResults
      : foldResults.filter(r => r.model_name === selectedModel);
    const pairs: LabeledPair[] = [];
    for (const r of src) {
      const actual = parsedLabel(r);
      if (actual !== null) pairs.push({ pred: clip(r.centiloid), actual, filename: r.filename, model: r.model_name });
    }
    return pairs;
  }, [foldResults, selectedModel, fusedData]);

  const metrics = useMemo(() => computeMetrics(labeled, thresholds), [labeled, thresholds]);

  /* ── scatter ───────────────────────────────────────────────────────────── */
  const scatterData = useMemo(() => {
    const data = labeled.map(p => ({ x: p.actual, y: p.pred, name: p.filename }));
    if (data.length <= MAX_SCATTER) return { data, sampled: false };
    return { data: [...data].sort(() => Math.random() - 0.5).slice(0, MAX_SCATTER), sampled: true };
  }, [labeled]);

  const refLineData = [{ x: CL_MIN, y: CL_MIN }, { x: CL_MAX, y: CL_MAX }];

  /* ── histograms ────────────────────────────────────────────────────────── */
  const residuals = useMemo(() => labeled.map(p => p.pred - p.actual), [labeled]);
  const residualBins = useMemo(() => buildHistogram(residuals, binCount), [residuals, binCount]);
  const distBins = useMemo(
    () => buildDualHistogram(labeled.map(p => p.actual), labeled.map(p => p.pred), binCount),
    [labeled, binCount],
  );

  /* ── table ─────────────────────────────────────────────────────────────── */
  type IndividualRow = Omit<InferenceResult, "id"> & { id: string };
  const individualRows = useMemo<IndividualRow[]>(() => {
    const src = selectedModel === "__all__" ? foldResults
      : selectedModel === "__fused__" ? []
      : foldResults.filter(r => r.model_name === selectedModel);
    return src.map(r => ({ ...r, id: String(r.id) }));
  }, [foldResults, selectedModel]);

  type FusedRow = { id: string; filename: string; pred: number; actual: number | null } & Record<string, number | string | null>;
  const fusedRows = useMemo<FusedRow[]>(() => {
    if (selectedModel !== "__fused__") return [];
    return fusedData.map(f => ({
      id: f.filename,
      filename: f.filename,
      pred: f.pred,
      actual: f.actual,
      ...f.modelPreds,
    }));
  }, [fusedData, selectedModel]);

  const individualColumns = useMemo((): TableColumn<IndividualRow>[] => [
    { key: "filename", header: "File", render: r => <span className="block max-w-[200px] truncate font-medium" title={r.filename}>{r.filename}</span> },
    { key: "model_name", header: "Model", render: r => <span className="inline-flex h-6 items-center rounded-[5px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-2 font-mono text-[11px]">{r.model_name}</span> },
    { key: "centiloid", header: "Predicted", align: "right", render: r => <span className="font-mono text-[12px] font-semibold tabular-nums">{clip(r.centiloid).toFixed(1)}</span> },
    {
      key: "label", header: "Actual (clipped)", align: "right",
      render: r => { const v = parsedLabel(r); return v !== null ? <span className="font-mono text-[12px] tabular-nums">{v.toFixed(1)}</span> : <span className="text-[var(--ds-gray-500)]">—</span>; },
    },
    {
      key: "fold", header: "Fold", align: "right",
      render: r => r.fold !== null ? <span className="font-mono text-[12px] tabular-nums">{r.fold}</span> : <span className="text-[var(--ds-gray-500)]">—</span>,
    },
    {
      key: "delta", header: "Δ Error", align: "right",
      render: r => {
        const v = parsedLabel(r);
        if (v === null) return <span className="text-[var(--ds-gray-500)]">—</span>;
        const d = Math.abs(clip(r.centiloid) - v);
        return <ErrorValue value={d} />;
      },
    },
    { key: "created_at", header: "Date", render: r => <span className="whitespace-nowrap font-mono text-[11px] text-[var(--ds-gray-700)]">{fmtDate(r.created_at)}</span> },
  ], []);

  const fusedColumns = useMemo((): TableColumn<FusedRow>[] => [
    { key: "filename", header: "File", render: r => <span className="block max-w-[180px] truncate font-medium" title={r.filename}>{r.filename}</span> },
    { key: "pred", header: "Fused pred", align: "right", render: r => <span className="font-mono text-[12px] font-semibold tabular-nums text-[var(--ds-gray-1000)]">{Number(r.pred).toFixed(1)}</span> },
    ...fusionModels.map(m => ({
      key: m, header: m.slice(0, 18), align: "right" as const,
      render: (r: FusedRow) => <span className="font-mono text-[11px] tabular-nums text-[var(--ds-gray-700)]">{typeof r[m] === "number" ? Number(r[m]).toFixed(1) : "—"}</span>,
    })),
    {
      key: "actual", header: "Actual", align: "right",
      render: r => r.actual !== null ? <span className="font-mono text-[12px] tabular-nums">{Number(r.actual).toFixed(1)}</span> : <span className="text-[var(--ds-gray-500)]">—</span>,
    },
    {
      key: "delta", header: "Δ Error", align: "right",
      render: r => {
        if (r.actual === null) return <span className="text-[var(--ds-gray-500)]">—</span>;
        const d = Math.abs(Number(r.pred) - Number(r.actual));
        return <ErrorValue value={d} />;
      },
    },
  ], [fusionModels]);

  /* ── auto-optimize ─────────────────────────────────────────────────────── */
  const handleOptimize = useCallback(() => {
    if (fusionModels.length < 2 || isOptimizing) return;
    setIsOptimizing(true);
    setTimeout(() => {
      try {
        const optW = optimizeWeights(foldResults, fusionModels, fusionOptTarget);
        const next: Record<string, number> = {};
        fusionModels.forEach((m, i) => { next[m] = Math.round(optW[i] * 10000) / 100; });
        setFusionWeights(next);
      } finally {
        setIsOptimizing(false);
      }
    }, 16);
  }, [foldResults, fusionModels, fusionOptTarget, isOptimizing]);

  /* ── summary count chips ───────────────────────────────────────────────── */
  const totalFiltered = selectedModel === "__fused__" ? fusedData.length
    : selectedModel === "__all__" ? foldResults.length
    : foldResults.filter(r => r.model_name === selectedModel).length;

  const tableRowsAll = selectedModel === "__fused__" ? fusedRows : individualRows;
  const tableLimit = 20;
  const tablePageCount = Math.max(1, Math.ceil(tableRowsAll.length / tableLimit));
  const safeTablePage = Math.min(tablePage, tablePageCount - 1);
  const tableStart = safeTablePage * tableLimit;
  const tableRowsPage = tableRowsAll.slice(tableStart, tableStart + tableLimit);

  /* ═══════════════════════════════════════════════════════════════════════ */
  /* render                                                                   */
  /* ═══════════════════════════════════════════════════════════════════════ */
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
          <Link href="/" className="inline-flex h-8 items-center gap-1.5 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2.5 text-[12px] font-medium text-[var(--ds-gray-900)] transition hover:bg-[var(--ds-gray-100)]">
            <Brain aria-hidden className="h-3.5 w-3.5" />Inference
          </Link>
          <div className="hidden items-center gap-2 md:flex">
            <StatusDot color={error ? "var(--ds-red-700)" : loading ? "var(--ds-amber-700)" : "var(--ds-green-700)"} pulse={loading} />
            <span className="font-mono text-[12px] text-[var(--ds-gray-700)]">{apiBase}</span>
          </div>
          <button onClick={() => { setApiBaseInput(apiBase); setShowSettings(v => !v); }}
            className="inline-flex h-8 items-center gap-1.5 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2.5 text-[12px] font-medium text-[var(--ds-gray-900)] transition hover:bg-[var(--ds-gray-100)]">
            <Settings2 aria-hidden className="h-3.5 w-3.5" />API
            {showSettings ? <ChevronUp aria-hidden className="h-3 w-3" /> : <ChevronDown aria-hidden className="h-3 w-3" />}
          </button>
        </div>
      </header>

      {showSettings && (
        <div className="mx-auto mt-2 max-w-[1440px]">
          <div className="rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] p-3">
            <p className="mb-2 font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">API base URL</p>
            <div className="flex gap-2">
              <input className="h-9 flex-1 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-3 font-mono text-[13px] outline-none focus:border-[var(--ds-blue-700)]"
                value={apiBaseInput} onChange={e => setApiBaseInput(e.target.value)} placeholder="http://localhost:7860" />
              <button onClick={() => { setApiBase(apiBaseInput.replace(/\/$/, "")); setShowSettings(false); }}
                className="inline-flex h-9 items-center rounded-[7px] border border-[var(--ds-gray-1000)] bg-[var(--ds-gray-1000)] px-3 text-[13px] font-medium text-[var(--ds-background-100)] transition hover:bg-black">
                Save &amp; reconnect
              </button>
            </div>
          </div>
        </div>
      )}

      <main className="mx-auto max-w-[1440px] space-y-4 py-4">

        {/* ── 01 Controls ─────────────────────────────────────────────────── */}
        <SectionCard eyebrow="01 / Controls" title="Model & parameters"
          action={
            <button onClick={loadData} disabled={loading}
              className="inline-flex h-8 items-center gap-1.5 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2.5 text-[12px] font-medium text-[var(--ds-gray-900)] transition hover:bg-[var(--ds-gray-100)] disabled:opacity-50">
              <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />Refresh
            </button>
          }
        >
          <div className="space-y-4 p-4">
            {/* Model pills */}
            <div>
              <p className="mb-2 font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">View model</p>
              <div className="flex flex-wrap gap-2">
                {["__all__", ...models, ...(fusionActive ? ["__fused__"] : [])].map(m => {
                  const label = m === "__all__" ? "All models" : m === "__fused__" ? "Fused" : m;
                  const count = m === "__all__" ? foldResults.length
                    : m === "__fused__" ? fusedData.length
                    : foldResults.filter(r => r.model_name === m).length;
                  return (
                    <button key={m} onClick={() => setSelectedModel(m)}
                      className={cn(
                        "inline-flex h-8 items-center gap-1.5 rounded-[7px] border px-3 text-[12px] font-medium transition",
                        selectedModel === m
                          ? "border-[var(--ds-gray-1000)] bg-[var(--ds-gray-1000)] text-[var(--ds-background-100)]"
                          : "border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] text-[var(--ds-gray-900)] hover:bg-[var(--ds-gray-100)]",
                      )}>
                      {label}
                      <span className={cn("inline-flex h-4 min-w-[20px] items-center justify-center rounded-full px-1 font-mono text-[10px]",
                        selectedModel === m ? "bg-white/20 text-white" : "bg-[var(--ds-gray-200)] text-[var(--ds-gray-800)]")}>
                        {count.toLocaleString()}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            {availableFolds.length > 0 && (
              <div>
                <p className="mb-2 font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">Fold</p>
                <div className="flex flex-wrap gap-2">
                  {(["__all__", ...availableFolds] as const).map(f => {
                    const selected = selectedFold === f;
                    const count = f === "__all__"
                      ? results.length
                      : results.filter(r => r.fold === f).length;
                    return (
                      <button
                        key={String(f)}
                        onClick={() => setSelectedFold(f)}
                        className={cn(
                          "inline-flex h-8 items-center gap-1.5 rounded-[7px] border px-3 text-[12px] font-medium transition",
                          selected
                            ? "border-[var(--ds-gray-1000)] bg-[var(--ds-gray-1000)] text-[var(--ds-background-100)]"
                            : "border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] text-[var(--ds-gray-900)] hover:bg-[var(--ds-gray-100)]",
                        )}
                      >
                        {f === "__all__" ? "All folds" : `Fold ${f}`}
                        <span className={cn("inline-flex h-4 min-w-[20px] items-center justify-center rounded-full px-1 font-mono text-[10px]",
                          selected ? "bg-white/20 text-white" : "bg-[var(--ds-gray-200)] text-[var(--ds-gray-800)]")}>
                          {count.toLocaleString()}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Sliders */}
            <div className="grid gap-4 sm:grid-cols-2">
              <ThresholdControls thresholds={thresholds} setThresholds={setThresholds} />
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <p className="font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">Histogram bins</p>
                  <span className="font-mono text-[13px] font-semibold tabular-nums text-[var(--ds-gray-900)]">{binCount}</span>
                </div>
                <input type="range" min={8} max={50} step={2} value={binCount}
                  onChange={e => setBinCount(Number(e.target.value))} className="w-full accent-[var(--ds-gray-1000)]" />
                <div className="mt-1 flex justify-between font-mono text-[10px] text-[var(--ds-gray-500)]">
                  <span>8</span><span>20</span><span>32</span><span>44</span>
                </div>
              </div>
            </div>

            {/* Summary chips */}
            <div className="flex flex-wrap gap-2">
              {[
                { label: "Total", value: totalFiltered.toLocaleString() },
                { label: "Labeled", value: labeled.length.toLocaleString() },
                { label: "Coverage", value: totalFiltered > 0 ? `${((labeled.length / totalFiltered) * 100).toFixed(1)}%` : "—" },
                { label: "Models", value: models.length.toString() },
                { label: "Fold", value: selectedFold === "__all__" ? "All" : String(selectedFold) },
                { label: "Clip range", value: "[−20, 130] CL" },
              ].map(s => (
                <div key={s.label} className="rounded-[6px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-3 py-1.5">
                  <p className="font-mono text-[10px] uppercase text-[var(--ds-gray-600)]">{s.label}</p>
                  <p className="font-mono text-[13px] font-semibold tabular-nums text-[var(--ds-gray-1000)]">{s.value}</p>
                </div>
              ))}
            </div>
          </div>
        </SectionCard>

        {error && (
          <div className="rounded-[8px] border border-[var(--ds-red-400)] bg-[var(--ds-red-100)] p-3 text-[var(--ds-red-900)]">
            <div className="flex items-start gap-2">
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
              <div>
                <p className="text-[13px] font-semibold">Cannot reach API</p>
                <p className="mt-1 font-mono text-[11px] leading-5 opacity-80">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* ── 02 Model Fusion ──────────────────────────────────────────────── */}
        <SectionCard eyebrow="02 / Model Fusion" title="Weighted ensemble"
          action={
            <div className="flex items-center gap-1.5">
              <span className="font-mono text-[11px] text-[var(--ds-gray-600)]">
                {fusionActive ? `${fusedData.length.toLocaleString()} samples · ${fusedData.filter(f => f.actual !== null).length.toLocaleString()} labeled` : "select 2+ models"}
              </span>
            </div>
          }
        >
          <div className="space-y-4 p-4">
            {/* Model checkboxes */}
            <div>
              <p className="mb-2 font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">Models to fuse</p>
              <div className="flex flex-wrap gap-2">
                {models.map(m => {
                  const checked = fusionModels.includes(m);
                  return (
                    <button key={m} onClick={() => setFusionModels(prev => checked ? prev.filter(x => x !== m) : [...prev, m])}
                      className={cn(
                        "inline-flex h-8 items-center gap-2 rounded-[7px] border px-2.5 text-[12px] font-medium leading-none outline-none transition focus-visible:shadow-[var(--ds-focus-ring)]",
                        checked
                          ? "border-[var(--ds-gray-1000)] bg-[var(--ds-gray-1000)] text-[var(--ds-background-100)]"
                          : "border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] text-[var(--ds-gray-1000)] hover:border-[var(--ds-gray-alpha-500)] hover:bg-[var(--ds-gray-100)]",
                      )}>
                      <span className={cn("h-2.5 w-2.5 rounded-full border border-black transition", checked ? "bg-[var(--ds-background-100)]" : "bg-transparent")} />
                      {m}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Weight controls — only when 2+ selected */}
            {fusionActive && (
              <div className="space-y-3">
                <p className="font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">Weights</p>
                {fusionModels.map(m => {
                  const raw = fusionWeights[m] ?? 50;
                  const pct = (normWeights[m] * 100).toFixed(1);
                  return (
                    <div key={m} className="grid items-center gap-3 rounded-[7px] border border-[var(--ds-gray-alpha-300)] bg-[var(--ds-background-200)] p-3 md:grid-cols-[1fr_120px_64px]">
                      <div>
                        <p className="mb-1 font-mono text-[11px] text-[var(--ds-gray-800)] truncate">{m}</p>
                        <input type="range" min={0} max={100} step={1} value={raw}
                          onChange={e => setFusionWeights(prev => ({ ...prev, [m]: Number(e.target.value) }))}
                          className="w-full accent-[var(--ds-gray-1000)]" />
                      </div>
                      <div className="relative">
                        <input type="number" min={0} max={100} step={0.1}
                          value={raw}
                          onChange={e => setFusionWeights(prev => ({ ...prev, [m]: Math.max(0, Number(e.target.value)) }))}
                          className="h-8 w-full rounded-[6px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] pr-8 pl-2 font-mono text-[12px] tabular-nums outline-none focus:border-[var(--ds-gray-1000)]" />
                        <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 font-mono text-[11px] text-[var(--ds-gray-600)]">raw</span>
                      </div>
                      <div className="rounded-[6px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-2 py-1 text-center">
                        <p className="font-mono text-[12px] font-semibold tabular-nums text-[var(--ds-gray-1000)]">{pct}%</p>
                      </div>
                    </div>
                  );
                })}

                {/* Auto-optimize row */}
                <div className="flex items-center gap-2 pt-1">
                  <select value={fusionOptTarget} onChange={e => setFusionOptTarget(e.target.value as "mae" | "rmse")}
                    className="h-8 rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2.5 font-mono text-[12px] outline-none transition focus-visible:shadow-[var(--ds-focus-ring)]">
                    <option value="mae">Minimize MAE</option>
                    <option value="rmse">Minimize RMSE</option>
                  </select>
                  <button onClick={handleOptimize}
                    disabled={isOptimizing || labeled.length === 0}
                    className="inline-flex h-8 items-center justify-center gap-2 rounded-[7px] border border-[var(--ds-gray-1000)] bg-[var(--ds-gray-1000)] px-2.5 text-[12px] font-medium leading-none text-[var(--ds-background-100)] outline-none transition hover:border-black hover:bg-black focus-visible:shadow-[var(--ds-focus-ring)] disabled:pointer-events-none disabled:opacity-50">
                    {isOptimizing
                      ? <><Loader2 className="h-3.5 w-3.5 animate-spin" />Optimizing…</>
                      : <><Shuffle className="h-3.5 w-3.5" />Auto-optimize weights</>}
                  </button>
                  {labeled.length === 0 && selectedModel !== "__fused__" && (
                    <span className="font-mono text-[11px] text-[var(--ds-gray-700)]">
                      Switch to Fused view first (needs labeled fused data)
                    </span>
                  )}
                </div>

                {/* Normalized weight summary bar */}
                <div>
                  <p className="mb-1.5 font-mono text-[10px] uppercase text-[var(--ds-gray-600)]">Normalized weights</p>
                  <div className="flex h-5 w-full overflow-hidden rounded-full border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)]">
                    {fusionModels.map((m, i) => {
                      const COLORS = ["var(--ds-gray-1000)", "var(--ds-gray-800)", "var(--ds-gray-600)", "var(--ds-gray-500)", "var(--ds-gray-400)"];
                      return (
                        <div key={m} style={{ width: `${(normWeights[m] * 100).toFixed(2)}%`, background: COLORS[i % COLORS.length] }}
                          title={`${m}: ${(normWeights[m] * 100).toFixed(1)}%`} />
                      );
                    })}
                  </div>
                  <div className="mt-1 flex flex-wrap gap-3">
                    {fusionModels.map((m, i) => {
                      const COLORS = ["var(--ds-gray-1000)", "var(--ds-gray-800)", "var(--ds-gray-600)", "var(--ds-gray-500)", "var(--ds-gray-400)"];
                      return (
                        <div key={m} className="flex items-center gap-1">
                          <span className="h-2 w-2 rounded-full border border-black" style={{ background: COLORS[i % COLORS.length] }} />
                          <span className="font-mono text-[10px] text-[var(--ds-gray-700)]">{m.slice(0, 20)} {(normWeights[m] * 100).toFixed(1)}%</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>
        </SectionCard>

        {/* ── 03 Regression metrics ────────────────────────────────────────── */}
        {metrics && (
          <>
            <SectionCard eyebrow="03 / Regression metrics" title={`Evaluated on ${metrics.n.toLocaleString()} labeled pairs`}>
              <div className="grid grid-cols-2 gap-3 p-4 sm:grid-cols-4">
                <MetricCard eyebrow="MAE" value={`${metrics.mae.toFixed(2)} CL`} sub="Mean absolute error" tone={metricTone(metrics.mae < 10, metrics.mae < 25)} />
                <MetricCard eyebrow="RMSE" value={`${metrics.rmse.toFixed(2)} CL`} sub="Root mean squared error" tone={metricTone(metrics.rmse < 12, metrics.rmse < 30)} />
                <MetricCard eyebrow="R²" value={metrics.r2.toFixed(4)} sub="Coefficient of determination"
                  tone={metricTone(metrics.r2 >= 0.8, metrics.r2 >= 0.6)} />
                <MetricCard eyebrow="Pearson r" value={metrics.pearson.toFixed(4)} sub="Linear correlation"
                  tone={metricTone(metrics.pearson >= 0.9, metrics.pearson >= 0.7)} />
              </div>
            </SectionCard>

            {/* ── 04 Classification metrics ─────────────────────────────────── */}
            <SectionCard eyebrow="04 / Classification metrics" title={`${metrics.classCount}-class classification`}>
              <div className="grid grid-cols-2 gap-3 p-4 sm:grid-cols-4">
                <MetricCard eyebrow="Accuracy" value={`${(metrics.accuracy * 100).toFixed(1)}%`}
                  sub={`${metrics.correct.toLocaleString()} / ${metrics.n.toLocaleString()} correct`}
                  tone={metricTone(metrics.accuracy >= 0.85, metrics.accuracy >= 0.70)} />
                <MetricCard eyebrow="Macro F1" value={metrics.macroF1.toFixed(4)} sub="Mean class F score"
                  tone={metricTone(metrics.macroF1 >= 0.85, metrics.macroF1 >= 0.70)} />
                {metrics.classF1.map((score, i) => (
                  <MetricCard key={i} eyebrow={`F${i + 1}`} value={score.toFixed(4)} sub={`Class ${i + 1}: ${classLabel(i, thresholds)} CL`}
                    tone={metricTone(score >= 0.85, score >= 0.70)} />
                ))}
              </div>
            </SectionCard>
          </>
        )}

        {/* ── Charts 2×2 ──────────────────────────────────────────────────── */}
        {labeled.length > 0 && (
          <div className="grid gap-4 xl:grid-cols-2">
            {/* 05 Scatter */}
            <SectionCard eyebrow="05 / Scatter" title="Predicted vs Actual"
              sub={scatterData.sampled ? `Showing ${MAX_SCATTER.toLocaleString()} random samples of ${labeled.length.toLocaleString()}` : undefined}>
              <div className="p-4">
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart margin={{ top: 8, right: 20, bottom: 28, left: 8 }}>
                    <CartesianGrid stroke="var(--ds-gray-alpha-300)" vertical={false} />
                    <XAxis dataKey="x" type="number" name="Actual" domain={[CL_MIN, CL_MAX]}
                      axisLine={false}
                      tick={{ fontSize: 11, fontFamily: "monospace" }}
                      tickLine={false}
                      tickMargin={10}
                      label={{ value: "Actual CL", position: "insideBottom", offset: -18, fontSize: 11 }} />
                    <YAxis dataKey="y" type="number" name="Predicted" domain={[CL_MIN, CL_MAX]}
                      axisLine={false}
                      tick={{ fontSize: 11, fontFamily: "monospace" }}
                      tickLine={false}
                      label={{ value: "Predicted CL", angle: -90, position: "insideLeft", offset: 14, fontSize: 11 }} />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }}
                      content={({ active, payload }) => {
                        if (!active || !payload?.[0]) return null;
                        const d = payload[0].payload;
                        return <ChartTip>
                          <p className="max-w-[160px] truncate font-mono text-[var(--ds-gray-700)]">{d.name}</p>
                          <p className="mt-1">Actual: <strong>{Number(d.x).toFixed(1)}</strong></p>
                          <p>Predicted: <strong>{Number(d.y).toFixed(1)}</strong></p>
                          <p>Error: <strong>±{Math.abs(Number(d.x) - Number(d.y)).toFixed(1)}</strong></p>
                        </ChartTip>;
                      }} />
                    <Scatter name="y=x" data={refLineData}
                      line={{ stroke: "var(--ds-green-700)", strokeDasharray: "5 4", strokeWidth: 1.5 }}
                      shape={() => <g />} legendType="none" />
                    <Scatter name="Results" data={scatterData.data}
                      shape={<ScatterPoint />} />
                  </ScatterChart>
                </ResponsiveContainer>
                <div className="mt-2 flex items-center justify-end gap-2 font-mono text-[10px] text-[var(--ds-gray-600)]">
                  <span>low error</span>
                  <div className="h-2 w-24 rounded-full border border-[var(--ds-gray-alpha-400)] bg-[linear-gradient(90deg,hsl(145_72%_38%),hsl(55_72%_40%),hsl(0_72%_42%))]" />
                  <span>high error</span>
                </div>
              </div>
            </SectionCard>

            {/* 06 Confusion matrix */}
            {metrics && (
              <SectionCard eyebrow="06 / Confusion matrix" title={`${metrics.classCount} × ${metrics.classCount} classification`}>
                <div className="overflow-x-auto p-6">
                  <ConfusionMatrix confusion={metrics.confusion} thresholds={thresholds} />
                </div>
              </SectionCard>
            )}

            {/* 07 Residuals */}
            <SectionCard eyebrow="07 / Residuals" title="Error distribution (predicted − actual)">
              <div className="p-4">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={residualBins} margin={{ top: 8, right: 20, bottom: 28, left: 8 }}>
                    <CartesianGrid stroke="var(--ds-gray-alpha-300)" vertical={false} />
                    <XAxis dataKey="bin" tick={{ fontSize: 10, fontFamily: "monospace" }}
                      axisLine={false}
                      label={{ value: "Error (CL)", position: "insideBottom", offset: -18, fontSize: 11 }}
                      interval="preserveStartEnd"
                      tickLine={false}
                      tickMargin={10} />
                    <YAxis axisLine={false} tick={{ fontSize: 11, fontFamily: "monospace" }} tickLine={false} />
                    <Tooltip content={({ active, payload }) => !active || !payload?.[0] ? null :
                      <ChartTip><p>Bin: <strong>{payload[0].payload.bin}</strong></p><p>Count: <strong>{payload[0].value}</strong></p></ChartTip>} />
                    <Bar dataKey="count"
                      fill="var(--ds-gray-1000)"
                      fillOpacity={0.78} radius={[6, 6, 0, 0]} stroke="var(--ds-gray-1000)" strokeWidth={1} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </SectionCard>

            {/* 08 Distribution */}
            <SectionCard eyebrow="08 / Distribution" title="Predicted vs actual centiloid distribution">
              <div className="p-4">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={distBins} margin={{ top: 8, right: 20, bottom: 28, left: 8 }}>
                    <CartesianGrid stroke="var(--ds-gray-alpha-300)" vertical={false} />
                    <XAxis dataKey="bin" tick={{ fontSize: 10, fontFamily: "monospace" }}
                      axisLine={false}
                      label={{ value: "Centiloid", position: "insideBottom", offset: -18, fontSize: 11 }}
                      interval="preserveStartEnd"
                      tickLine={false}
                      tickMargin={10} />
                    <YAxis axisLine={false} tick={{ fontSize: 11, fontFamily: "monospace" }} tickLine={false} />
                    <Legend wrapperStyle={{ fontSize: 11, fontFamily: "monospace", paddingTop: 8 }}
                      formatter={v => v === "actual" ? "Ground truth" : "Predicted"} />
                    <Tooltip content={({ active, payload }) => !active || !payload?.length ? null :
                      <ChartTip>
                        <p>Bin: <strong>{payload[0]?.payload?.bin}</strong></p>
                        {payload.map(p => <p key={p.name} style={{ color: p.color as string }}>{p.name === "actual" ? "Ground truth" : "Predicted"}: <strong>{p.value}</strong></p>)}
                      </ChartTip>} />
                    <Bar dataKey="actual" fill="var(--ds-green-700)" fillOpacity={0.72} radius={[6, 6, 0, 0]} stroke="var(--ds-gray-1000)" strokeWidth={1} />
                    <Bar dataKey="pred"
                      fill="var(--ds-gray-1000)"
                      fillOpacity={0.72} radius={[6, 6, 0, 0]} stroke="var(--ds-gray-1000)" strokeWidth={1} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </SectionCard>
          </div>
        )}

        {/* Empty states */}
        {!loading && labeled.length === 0 && totalFiltered > 0 && (
          <div className="rounded-[8px] border border-[var(--ds-amber-400)] bg-[var(--ds-amber-100)] px-4 py-6 text-center text-[var(--ds-amber-900)]">
            <p className="text-[13px] font-semibold">No labeled results found for this selection</p>
            <p className="mt-1 text-[12px] leading-5 opacity-80">
              Run inference with ground-truth labels to see metrics and charts.
            </p>
          </div>
        )}
        {!loading && results.length === 0 && !error && (
          <div className="flex flex-col items-center justify-center gap-3 rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] py-16">
            <Activity className="h-8 w-8 text-[var(--ds-gray-400)]" />
            <p className="text-[13px] font-medium text-[var(--ds-gray-700)]">No inference results yet</p>
            <Link href="/" className="mt-1 inline-flex h-8 items-center gap-1.5 rounded-[7px] border border-[var(--ds-gray-1000)] bg-[var(--ds-gray-1000)] px-3 text-[12px] font-medium text-[var(--ds-background-100)] transition hover:bg-black">
              <Brain className="h-3.5 w-3.5" />Go to inference
            </Link>
          </div>
        )}

        {/* ── 09 Results table ─────────────────────────────────────────────── */}
        {(individualRows.length > 0 || fusedRows.length > 0) && (
          <SectionCard eyebrow="09 / Results" title={selectedModel === "__fused__" ? "Fused model predictions" : "All inference records"}
            action={
              <span className="inline-flex h-6 items-center rounded-[5px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-200)] px-2 font-mono text-[11px] text-[var(--ds-gray-800)]">
                {tableRowsAll.length
                  ? `${tableStart + 1}-${Math.min(tableStart + tableRowsPage.length, tableRowsAll.length)} / ${tableRowsAll.length.toLocaleString()} records`
                  : "0 records"}
              </span>
            }>
            {selectedModel === "__fused__"
              ? <DataTable rows={tableRowsPage as typeof fusedRows} columns={fusedColumns} />
              : <DataTable rows={tableRowsPage as typeof individualRows} columns={individualColumns} />}
            {tableRowsAll.length > tableLimit && (
              <div className="flex items-center justify-end gap-2 border-t border-[var(--ds-gray-alpha-300)] p-3">
                <button
                  onClick={() => setTablePage(Math.max(0, safeTablePage - 1))}
                  disabled={safeTablePage === 0}
                  className="inline-flex h-8 items-center rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2.5 text-[12px] font-medium text-[var(--ds-gray-900)] transition hover:bg-[var(--ds-gray-100)] disabled:opacity-50"
                >
                  Prev
                </button>
                <button
                  onClick={() => setTablePage(Math.min(tablePageCount - 1, safeTablePage + 1))}
                  disabled={safeTablePage >= tablePageCount - 1}
                  className="inline-flex h-8 items-center rounded-[7px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] px-2.5 text-[12px] font-medium text-[var(--ds-gray-900)] transition hover:bg-[var(--ds-gray-100)] disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            )}
          </SectionCard>
        )}
      </main>
    </div>
  );
}
