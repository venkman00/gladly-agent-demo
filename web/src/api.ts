// Types mirror app/models.py. Keep them in sync, or generate them from /openapi.json at scale.

export type Intent = "order_status" | "return_refund" | "shipping" | "account" | "other";
export type TicketStatus = "queued" | "drafted" | "escalated" | "approved" | "rejected" | "ignored";

export interface Draft {
  intent: Intent;
  reply: string;
  confidence: number;
  citations: string[];
  escalate: boolean;
  reason: string;
}

export interface Ticket {
  id: string;
  conversation_id: string;
  customer_id: string;
  message: string | null;
  status: TicketStatus;
  draft: Draft | null;
  latency_ms: number | null;
  created_at: string;
}

const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const TOKEN = import.meta.env.VITE_API_TOKEN ?? "dev-token";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${TOKEN}`, ...init?.headers },
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  list: () => call<Ticket[]>("/api/tickets"),
  create: (message: string) => call<Ticket>("/api/tickets", { method: "POST", body: JSON.stringify({ message }) }),
  review: (id: string, approved: boolean) =>
    call<Ticket>(`/api/tickets/${id}/review`, { method: "POST", body: JSON.stringify({ approved }) }),
};
