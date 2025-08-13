import { NextRequest, NextResponse } from 'next/server';

export async function GET(req: NextRequest, { params }: { params: { slug: string } }) {
  const jobId = params.slug;
  const baseUrl = process.env.MODAL_BASE_URL;
  if (!baseUrl) {
    return NextResponse.json({ error: 'Missing MODAL_BASE_URL' }, { status: 500 });
  }

  const res = await fetch(`${baseUrl}/status?jobId=${encodeURIComponent(jobId)}`);
  const data = await res.json();
  return NextResponse.json(data);
}
