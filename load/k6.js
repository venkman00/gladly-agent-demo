// Load test for the edge (API + enqueue), not the LLM: we want the webhook path to stay fast
// no matter how slow the model is. Run: k6 run -e BASE_URL=http://localhost:8000 -e TOKEN=dev-token load/k6.js
import http from "k6/http";
import { check } from "k6";

export const options = {
  scenarios: {
    ramp: {
      executor: "ramping-arrival-rate",
      startRate: 5,
      timeUnit: "1s",
      preAllocatedVUs: 50,
      stages: [
        { target: 50, duration: "30s" }, // ramp to 50 req/s
        { target: 50, duration: "1m" }, // hold
        { target: 0, duration: "10s" },
      ],
    },
  },
  thresholds: {
    http_req_failed: ["rate<0.01"], // <1% errors
    http_req_duration: ["p(95)<300"], // p95 under 300ms: enqueue, don't process inline
  },
};

const messages = [
  "Where is my order #1001?",
  "How long do refunds take?",
  "Can I return a final-sale item?",
  "I forgot my password",
];

export default function () {
  const res = http.post(
    `${__ENV.BASE_URL}/api/tickets`,
    JSON.stringify({ message: messages[Math.floor(Math.random() * messages.length)] }),
    { headers: { "Content-Type": "application/json", Authorization: `Bearer ${__ENV.TOKEN}` } },
  );
  check(res, { "202 accepted": (r) => r.status === 202 });
}
