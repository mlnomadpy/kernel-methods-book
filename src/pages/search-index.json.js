import { buildBook } from "../lib/book.js";

export function GET() {
  const { searchIndex } = buildBook();
  return new Response(JSON.stringify(searchIndex), {
    headers: { "Content-Type": "application/json" },
  });
}
