'use client';
import { useEffect, useRef, useState } from 'react';

function Canvas({ onPixels }: { onPixels: (pixels: number[]) => void }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  function start(e: React.MouseEvent) {
    setIsDrawing(true);
    draw(e);
  }
  function end() { setIsDrawing(false); }
  function draw(e: React.MouseEvent) {
    if (!isDrawing) return;
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(x, y, 12, 0, 2 * Math.PI);
    ctx.fill();
  }

  function export28x28() {
    const canvas = canvasRef.current!;
    const tmp = document.createElement('canvas');
    tmp.width = 28; tmp.height = 28;
    const tctx = tmp.getContext('2d')!;
    tctx.drawImage(canvas, 0, 0, 28, 28);
    const img = tctx.getImageData(0, 0, 28, 28);
    const pixels: number[] = [];
    for (let i = 0; i < img.data.length; i += 4) {
      const r = img.data[i], g = img.data[i+1], b = img.data[i+2];
      const v = (r + g + b) / 3; // 0-255
      pixels.push(v / 255.0);
    }
    onPixels(pixels);
  }

  return (
    <div>
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        style={{ border: '1px solid #ccc', touchAction: 'none' }}
        onMouseDown={start}
        onMouseUp={end}
        onMouseMove={draw}
        onMouseLeave={end}
      />
      <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <button onClick={() => { const c = canvasRef.current!; const ctx = c.getContext('2d')!; ctx.fillStyle = 'black'; ctx.fillRect(0,0,c.width,c.height); }}>Clear</button>
        <button onClick={export28x28}>Predict</button>
      </div>
    </div>
  );
}

export default function DemoPage({ params }: { params: { slug: string } }) {
  const { slug } = params;
  const [prediction, setPrediction] = useState<{ label: string; probs: number[] } | null>(null);

  async function onPixels(pixels: number[]) {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ slug, pixels })
    });
    const data = await res.json();
    setPrediction(data);
  }

  return (
    <div style={{ display: 'grid', gap: 12 }}>
      <h2>Demo: {slug}</h2>
      <Canvas onPixels={onPixels} />
      {prediction && (
        <div>
          <div><b>Prediction:</b> {prediction.label}</div>
        </div>
      )}
    </div>
  );
}
