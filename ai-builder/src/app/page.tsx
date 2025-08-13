'use client';
import { useState } from 'react';

export default function HomePage() {
  const [prompt, setPrompt] = useState('build a classifier for mnist');
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('');
  const [error, setError] = useState<string>('');

  async function startBuild(e: React.FormEvent) {
    e.preventDefault();
    setError('');
    setStatus('Starting...');
    try {
      const res = await fetch('/api/build', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      if (!res.ok) {
        const text = await res.text();
        setError(`Build failed: ${text}`);
        setStatus('');
        return;
      }
      const data = await res.json();
      if (!data.jobId) {
        setError('No job id returned');
        setStatus('');
        return;
      }
      setJobId(data.jobId);
      setStatus('Queued');

      const interval = setInterval(async () => {
        try {
          const sres = await fetch(`/api/status/${data.jobId}`);
          if (!sres.ok) return;
          const sdata = await sres.json();
          setStatus(sdata.status || '');
          if (sdata.status === 'completed' && sdata.slug) {
            clearInterval(interval);
            window.location.href = `/demo/${sdata.slug}`;
          }
          if (sdata.status === 'failed') {
            clearInterval(interval);
          }
        } catch {}
      }, 2000);
    } catch (err: any) {
      setError(String(err?.message || err));
      setStatus('');
    }
  }

  return (
    <form onSubmit={startBuild} style={{ display: 'grid', gap: 12 }}>
      <label style={{ display: 'grid', gap: 6 }}>
        Prompt
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
          style={{ width: '100%', fontFamily: 'inherit' }}
        />
      </label>
      <button type="submit">Build project</button>
      {jobId && <div>Job: {jobId}</div>}
      {status && <div>Status: {status}</div>}
      {error && <div style={{ color: 'crimson' }}>{error}</div>}
    </form>
  );
}
