import { NextResponse } from 'next/server'

export const dynamic = 'force-dynamic'

function resolveStartUrl(): string {
	const direct = process.env.MODAL_START_URL
	if (direct) return direct
	const base = process.env.MODAL_APP_URL
	if (base) return `${base.replace(/\/$/, '')}/start`
	throw new Error('Missing Modal endpoint configuration. Set MODAL_START_URL or MODAL_APP_URL.')
}

export async function POST(req: Request) {
	try {
		const body = await req.json().catch(() => ({})) as { prompt?: string }
		const prompt = body?.prompt || ''
		if (!prompt) {
			return NextResponse.json({ error: 'Missing prompt' }, { status: 400 })
		}

		const url = resolveStartUrl()
		const res = await fetch(url, {
			method: 'POST',
			headers: { 'content-type': 'application/json' },
			body: JSON.stringify({ prompt }),
			// Avoid Next.js fetch caching for dynamic job creation
			cache: 'no-store',
		})

		const data = await res.json().catch(() => ({}))
		if (!res.ok) {
			return NextResponse.json({ error: data?.error || 'Upstream error', status: res.status }, { status: 502 })
		}

		return NextResponse.json(data)
	} catch (error: unknown) {
		const message = error instanceof Error ? error.message : 'Unknown error'
		return NextResponse.json({ error: message }, { status: 500 })
	}
}