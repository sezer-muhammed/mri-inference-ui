"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { cn } from "@/lib/cn";

/* Niivue slice type constants (SLICE_TYPE enum) */
const SLICE = { AXIAL: 0, CORONAL: 1, SAGITTAL: 2, MULTIPLANAR: 3, RENDER: 4 } as const;
type SliceKey = keyof typeof SLICE;

const MODES: { key: SliceKey; label: string }[] = [
  { key: "MULTIPLANAR", label: "4-panel" },
  { key: "RENDER", label: "3D render" },
  { key: "AXIAL", label: "Axial" },
  { key: "CORONAL", label: "Coronal" },
  { key: "SAGITTAL", label: "Sagittal" },
];

type Props = { file: File | null };

export function NiiViewer({ file }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nvRef = useRef<any>(null);
  const objectUrlRef = useRef<string | null>(null);

  const [ready, setReady] = useState(false);
  const [loading, setLoading] = useState(false);
  const [hasVolume, setHasVolume] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<SliceKey>("MULTIPLANAR");

  /* Initialize Niivue once */
  useEffect(() => {
    if (!canvasRef.current) return;
    let cancelled = false;

    (async () => {
      try {
        const { Niivue } = await import("@niivue/niivue");
        if (cancelled) return;

        const nv = new Niivue({
          backColor: [0.07, 0.08, 0.10, 1.0],
          show3Dcrosshair: true,
          crosshairWidth: 1.0,
          isColorbar: false,
          isOrientCube: true,
          isNearestInterpolation: false,
        });

        nv.attachToCanvas(canvasRef.current!);
        nv.setSliceType(SLICE.MULTIPLANAR);
        nvRef.current = nv;
        setReady(true);
      } catch (err: any) {
        if (!cancelled) setError("WebGL init failed: " + (err?.message ?? String(err)));
      }
    })();

    return () => { cancelled = true; };
  }, []);

  /* Load volume when file changes */
  useEffect(() => {
    if (!file || !ready || !nvRef.current) return;

    const nv = nvRef.current;
    let mounted = true;

    (async () => {
      setLoading(true);
      setError(null);

      /* Revoke any previous object URL */
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }

      try {
        const url = URL.createObjectURL(file);
        objectUrlRef.current = url;
        await nv.loadVolumes([{ url, name: file.name }]);
        if (mounted) setHasVolume(true);
      } catch (err: any) {
        if (mounted) {
          setError("Cannot preview this file: " + (err?.message ?? "unsupported format"));
          setHasVolume(false);
        }
      } finally {
        if (mounted) setLoading(false);
      }
    })();

    return () => { mounted = false; };
  }, [file, ready]);

  /* Cleanup object URL on unmount */
  useEffect(() => {
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  const changeMode = useCallback((key: SliceKey) => {
    if (!nvRef.current) return;
    setMode(key);
    nvRef.current.setSliceType(SLICE[key]);
  }, []);

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-[8px] border border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)]">
      {/* Toolbar */}
      <div className="flex shrink-0 items-center gap-1 border-b border-[var(--ds-gray-alpha-300)] bg-[var(--ds-background-100)] px-2 py-2">
        <p className="mr-2 font-mono text-[11px] uppercase text-[var(--ds-gray-700)]">View</p>
        {MODES.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => changeMode(key)}
            className={cn(
              "inline-flex h-7 items-center rounded-[6px] border px-2.5 font-mono text-[11px] uppercase transition",
              mode === key
                ? "border-[var(--ds-gray-1000)] bg-[var(--ds-gray-1000)] text-[var(--ds-background-100)]"
                : "border-[var(--ds-gray-alpha-400)] bg-[var(--ds-background-100)] text-[var(--ds-gray-900)] hover:bg-[var(--ds-gray-100)]",
            )}
          >
            {label}
          </button>
        ))}
        {hasVolume && (
          <span className="ml-auto font-mono text-[11px] text-[var(--ds-gray-700)]">
            {file?.name}
          </span>
        )}
      </div>

      {/* Canvas area */}
      <div className="relative flex-1 bg-[#12131a]">
        {/* Empty state */}
        {!hasVolume && !loading && !error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <div className="grid h-12 w-12 place-items-center rounded-[10px] border border-[var(--ds-gray-alpha-300)] bg-[var(--ds-gray-alpha-100)]">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-[var(--ds-gray-600)]">
                <ellipse cx="12" cy="12" rx="10" ry="6" />
                <ellipse cx="12" cy="12" rx="10" ry="6" transform="rotate(60 12 12)" />
                <ellipse cx="12" cy="12" rx="10" ry="6" transform="rotate(120 12 12)" />
              </svg>
            </div>
            <div className="text-center">
              <p className="text-[13px] font-medium text-[var(--ds-gray-700)]">No volume loaded</p>
              <p className="mt-0.5 font-mono text-[11px] text-[var(--ds-gray-600)]">
                Select a .nii or .nii.gz file
              </p>
            </div>
          </div>
        )}

        {/* Loading overlay */}
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#12131a]/80 backdrop-blur-sm">
            <div className="flex items-center gap-3">
              <div className="h-4 w-4 animate-spin rounded-full border-2 border-[var(--ds-blue-700)] border-t-transparent" />
              <span className="font-mono text-[12px] text-[var(--ds-gray-600)]">Loading volume…</span>
            </div>
          </div>
        )}

        {/* Error overlay */}
        {error && !loading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center">
            <div className="grid h-10 w-10 place-items-center rounded-[8px] border border-[var(--ds-red-400)] bg-[var(--ds-red-100)]">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-[var(--ds-red-700)]">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
            </div>
            <p className="text-[12px] text-[var(--ds-red-700)]">{error}</p>
            <p className="font-mono text-[11px] text-[var(--ds-gray-600)]">
              .nii.tar files may not be previewable — inference will still work
            </p>
          </div>
        )}

        <canvas
          ref={canvasRef}
          className="h-full w-full"
          style={{ display: !hasVolume && !loading ? "none" : "block" }}
        />
      </div>
    </div>
  );
}
