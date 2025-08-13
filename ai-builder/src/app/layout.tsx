export const metadata = {
  title: 'AI Project Builder',
  description: 'Build, train, and demo ML projects from a prompt',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: 'Inter, system-ui, Arial, sans-serif', margin: 0 }}>
        <div style={{ maxWidth: 980, margin: '0 auto', padding: 24 }}>
          <h1>AI Project Builder</h1>
          {children}
        </div>
      </body>
    </html>
  );
}
