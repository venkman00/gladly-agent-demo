import { useEffect, useState } from "react";
import { api, type Ticket } from "./api";

const POLL_MS = 2000;

export default function App() {
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState<string | null>(null);

  const refresh = () => api.list().then(setTickets, (e: Error) => setError(e.message));

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_MS); // polling is the minimum effective dose; websockets later
    return () => clearInterval(id);
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!message.trim()) return;
    await api.create(message).catch((e: Error) => setError(e.message));
    setMessage("");
    refresh();
  }

  async function review(id: string, approved: boolean) {
    await api.review(id, approved).catch((e: Error) => setError(e.message));
    refresh();
  }

  return (
    <main>
      <h1>Support Copilot</h1>
      <form onSubmit={submit}>
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Paste a customer message, e.g. 'Where is my order #1001?'"
          rows={3}
        />
        <button type="submit">Draft reply</button>
      </form>
      {error && <p className="error">{error}</p>}
      {tickets.map((t) => (
        <TicketCard key={t.id} ticket={t} onReview={review} />
      ))}
    </main>
  );
}

function TicketCard({ ticket: t, onReview }: { ticket: Ticket; onReview: (id: string, ok: boolean) => void }) {
  const d = t.draft;
  return (
    <article className={`card ${t.status}`}>
      <header>
        <span className="badge">{t.status}</span>
        {d && <span className="badge">{d.intent}</span>}
        {t.latency_ms != null && <small>{(t.latency_ms / 1000).toFixed(1)}s</small>}
      </header>
      <blockquote>{t.message ?? "(fetching from Gladly…)"}</blockquote>
      {d && (
        <>
          <p className="reply">{d.reply || <em>No draft — needs a human.</em>}</p>
          <div className="meter" title={`confidence ${d.confidence}`}>
            <div style={{ width: `${d.confidence * 100}%` }} />
          </div>
          <small>
            {d.reason}
            {d.citations.length > 0 && <> · Sources: {d.citations.join(", ")}</>}
          </small>
          {(t.status === "drafted" || t.status === "escalated") && (
            <div className="actions">
              <button onClick={() => onReview(t.id, true)}>Approve</button>
              <button className="secondary" onClick={() => onReview(t.id, false)}>
                Reject
              </button>
            </div>
          )}
        </>
      )}
    </article>
  );
}
