'use client';
import { useState } from 'react';

export default function HomePage() {
  const [prompt, setPrompt] = useState('build a classifier for mnist');
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('');

  async function startBuild(e: React.FormEvent) {
    e.preventDefault();
    setStatus('Starting...');
    const res = await fetch('/api/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt }),
    });
    const data = await res.json();
    setJobId(data.jobId);
    setStatus('Queued');

    const interval = setInterval(async () => {
      if (!data.jobId) return;
      const sres = await fetch(`/api/status/${data.jobId}`);
      const sdata = await sres.json();
      setStatus(sdata.status || '');
      if (sdata.status === 'completed' && sdata.slug) {
        clearInterval(interval);
        window.location.href = `/demo/${sdata.slug}`;
      }
      if (sdata.status === 'failed') {
        clearInterval(interval);
      }
    }, 2000);
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
    </form>
  );
}
