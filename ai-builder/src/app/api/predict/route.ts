import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  const body = await req.json();
  const { slug, pixels } = body as { slug: string; pixels: number[] };
  const baseUrl = process.env.MODAL_BASE_URL;
  if (!baseUrl) {
    return NextResponse.json({ error: 'Missing MODAL_BASE_URL' }, { status: 500 });
  }
  const res = await fetch(`${baseUrl}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ slug, pixels }),
  });
  const data = await res.json();
  return NextResponse.json(data);
}
