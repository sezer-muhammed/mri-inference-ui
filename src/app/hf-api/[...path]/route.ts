import { NextRequest, NextResponse } from "next/server";

const HF_BASE = "https://sezer-muhammed-mri-inference-api.hf.space";

async function proxy(request: NextRequest, params: Promise<{ path: string[] }>) {
  const { path } = await params;
  const url = `${HF_BASE}/${path.join("/")}${request.nextUrl.search}`;

  const contentType = request.headers.get("content-type");
  const body = request.method !== "GET" && request.method !== "HEAD"
    ? await request.arrayBuffer()
    : undefined;

  const res = await fetch(url, {
    method: request.method,
    body,
    headers: contentType ? { "content-type": contentType } : {},
  });

  const resBody = await res.arrayBuffer();
  return new NextResponse(resBody, {
    status: res.status,
    headers: { "content-type": res.headers.get("content-type") ?? "application/json" },
  });
}

export const GET = (req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) =>
  proxy(req, params);

export const POST = (req: NextRequest, { params }: { params: Promise<{ path: string[] }> }) =>
  proxy(req, params);
