import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { slug, pixels } = body as { slug: string; pixels: number[] };
    const predictUrl = process.env.MODAL_PREDICT_URL;
    if (!predictUrl) {
      return NextResponse.json({ error: 'Missing MODAL_PREDICT_URL' }, { status: 500 });
    }
    const res = await fetch(predictUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slug, pixels }),
    });
    if (!res.ok) {
      const text = await res.text();
      return NextResponse.json({ error: 'Modal predict failed', detail: text }, { status: 500 });
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch (err: any) {
    return NextResponse.json({ error: 'Unexpected error', detail: String(err?.message || err) }, { status: 500 });
  }
}
