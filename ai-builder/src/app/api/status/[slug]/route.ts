import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest, { params }: { params: { slug: string } }) {
  try {
    const jobId = params.slug;
    const statusUrl = process.env.MODAL_STATUS_URL;
    if (!statusUrl) {
      return NextResponse.json({ error: 'Missing MODAL_STATUS_URL' }, { status: 500 });
    }

    const url = new URL(statusUrl);
    url.searchParams.set('jobId', jobId);

    const res = await fetch(url.toString());
    if (!res.ok) {
      const text = await res.text();
      return NextResponse.json({ error: 'Modal status failed', detail: text }, { status: 500 });
    }
    const data = await res.json();
    return NextResponse.json(data);
  } catch (err: any) {
    return NextResponse.json({ error: 'Unexpected error', detail: String(err?.message || err) }, { status: 500 });
  }
}
