"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Source = {
  id: number;
  source: string;
  page: number | null;
  excerpt: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  streaming?: boolean;
  error?: string;
};

const SUGGESTED = [
  "What is the literacy rate of Bihar according to the census?",
  "Summarise the state's key agricultural exports.",
  "Which districts have the highest population density?",
  "What welfare schemes are highlighted in the reports?",
];

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

function prettySource(name: string) {
  return name.replace(/^.*[\\/]/, "");
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ready, setReady] = useState<boolean | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => setReady(!!d.ready))
      .catch(() => setReady(false));
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [input]);

  const send = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || busy) return;

      setError(null);
      const userMsg: Message = { id: uid(), role: "user", content: q };
      const assistantId = uid();
      const assistantMsg: Message = {
        id: assistantId,
        role: "assistant",
        content: "",
        streaming: true,
      };
      setMessages((m) => [...m, userMsg, assistantMsg]);
      setInput("");
      setBusy(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: q }),
          signal: controller.signal,
        });

        if (!res.ok || !res.body) {
          throw new Error(`Request failed (${res.status})`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const chunks = buffer.split("\n\n");
          buffer = chunks.pop() ?? "";

          for (const chunk of chunks) {
            const lines = chunk.split("\n");
            let event = "message";
            let data = "";
            for (const line of lines) {
              if (line.startsWith("event: ")) event = line.slice(7).trim();
              else if (line.startsWith("data: ")) data += line.slice(6);
            }
            if (!data) continue;
            let payload: Record<string, unknown> = {};
            try {
              payload = JSON.parse(data);
            } catch {
              continue;
            }

            if (event === "sources") {
              const sources = payload.sources as Source[];
              setMessages((m) =>
                m.map((msg) =>
                  msg.id === assistantId ? { ...msg, sources } : msg
                )
              );
            } else if (event === "token") {
              const text = String(payload.text ?? "");
              setMessages((m) =>
                m.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, content: msg.content + text }
                    : msg
                )
              );
            } else if (event === "error") {
              const message = String(payload.message ?? "Unknown error");
              setMessages((m) =>
                m.map((msg) =>
                  msg.id === assistantId
                    ? { ...msg, error: message, streaming: false }
                    : msg
                )
              );
            } else if (event === "done") {
              setMessages((m) =>
                m.map((msg) =>
                  msg.id === assistantId ? { ...msg, streaming: false } : msg
                )
              );
            }
          }
        }
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : "Request failed";
        setError(msg);
        setMessages((m) =>
          m.map((x) =>
            x.id === assistantId ? { ...x, streaming: false, error: msg } : x
          )
        );
      } finally {
        setBusy(false);
        abortRef.current = null;
      }
    },
    [busy]
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
    setBusy(false);
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setBusy(false);
    setError(null);
  }, []);

  const empty = messages.length === 0;
  const canSend = input.trim().length > 0 && !busy;

  const statusLabel = useMemo(() => {
    if (ready === null) return "Connecting…";
    if (ready) return "Ready";
    return "Backend offline";
  }, [ready]);
  const statusColor =
    ready === null
      ? "bg-text-subtle"
      : ready
      ? "bg-accent"
      : "bg-[color:var(--danger)]";

  return (
    <div className="grid h-svh grid-cols-1 gap-3 p-3 md:grid-cols-[280px_1fr] md:gap-4 md:p-5 lg:p-6">
      <Sidebar
        onNew={reset}
        onSuggest={(q) => {
          setInput(q);
          textareaRef.current?.focus();
        }}
        statusLabel={statusLabel}
        statusColor={statusColor}
      />

      <main className="relative flex h-full min-h-0 flex-col overflow-hidden rounded-[28px] border border-[color:var(--glass-border)] shadow-[var(--shadow-lg)]">
        <Header />

        <section
          ref={scrollRef}
          className="scroll-slim flex-1 overflow-y-auto"
          aria-live="polite"
        >
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-8 md:px-8">
            {empty ? (
              <EmptyState onPick={(q) => send(q)} />
            ) : (
              messages.map((m) => <Bubble key={m.id} message={m} />)
            )}
          </div>
        </section>

        <Composer
          value={input}
          onChange={setInput}
          onSend={() => send(input)}
          onStop={stop}
          busy={busy}
          canSend={canSend}
          textareaRef={textareaRef}
          error={error}
        />
      </main>
    </div>
  );
}

/* --------------------------------- Sidebar --------------------------------- */

function Sidebar({
  onNew,
  onSuggest,
  statusLabel,
  statusColor,
}: {
  onNew: () => void;
  onSuggest: (q: string) => void;
  statusLabel: string;
  statusColor: string;
}) {
  return (
    <aside className="glass hidden h-full flex-col gap-6 overflow-y-auto rounded-[28px] px-5 py-6 shadow-[var(--shadow-lg)] md:flex">
      <div className="flex items-center gap-2.5">
        <BrandMark />
        <div className="flex flex-col leading-tight">
          <span className="font-serif text-lg font-semibold tracking-tight">
            Bihar Insights
          </span>
          <span className="text-xs text-text-subtle">Grounded document RAG</span>
        </div>
      </div>

      <button
        type="button"
        onClick={onNew}
        className="group inline-flex cursor-pointer items-center justify-center gap-2 rounded-2xl border border-border-strong bg-bg-elev px-3.5 py-2.5 text-sm font-medium shadow-[var(--shadow-sm)] transition-colors duration-200 hover:border-primary hover:text-primary focus-visible:border-primary"
      >
        <PlusIcon className="h-4 w-4" />
        New conversation
      </button>

      <div className="flex flex-col gap-3">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-text-subtle">
          Try asking
        </span>
        <ul className="flex flex-col gap-1.5">
          {SUGGESTED.map((s) => (
            <li key={s}>
              <button
                type="button"
                onClick={() => onSuggest(s)}
                className="w-full cursor-pointer rounded-xl px-2.5 py-2 text-left text-sm leading-snug text-text-muted transition-colors duration-200 hover:bg-bg-elev hover:text-text"
              >
                {s}
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="mt-auto flex flex-col gap-3 rounded-2xl border border-border bg-bg-elev/70 p-3.5 text-xs text-text-muted">
        <div className="flex items-center gap-2">
          <span className={`inline-block h-2 w-2 rounded-full ${statusColor}`} />
          <span className="font-medium text-text">{statusLabel}</span>
        </div>
        <p className="leading-relaxed">
          Answers cite the source PDF and page. If a fact is absent, the model
          will say so instead of guessing.
        </p>
      </div>
    </aside>
  );
}

/* ---------------------------------- Header --------------------------------- */

function Header() {
  return (
    <header className="glass-strong sticky top-0 z-10 flex items-center justify-between gap-4 px-4 py-3 md:px-8">
      <div className="flex items-center gap-2 md:hidden">
        <BrandMark small />
        <span className="font-serif text-base font-semibold">Bihar Insights</span>
      </div>
      <div className="hidden md:flex items-center gap-2 text-sm text-text-muted">
        <SparkleIcon className="h-4 w-4 text-primary" />
        <span>Ask, cite, verify.</span>
      </div>
      <a
        href=" https://huggingface.co/"
        target="_blank"
        rel="noreferrer"
        className="rounded-full border border-border bg-bg-elev px-3 py-1 text-xs font-medium text-text-muted transition-colors duration-200 hover:border-accent hover:text-accent"
      >
        Powered by HuggingFace
      </a>
    </header>
  );
}

/* -------------------------------- Empty state ------------------------------ */

function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="reveal flex flex-col items-center gap-8 py-16 text-center">
      <div className="flex flex-col items-center gap-3">
        <BrandMark large />
        <h1 className="font-serif text-3xl font-semibold leading-tight md:text-4xl">
          Ask anything about Bihar
        </h1>
        <p className="max-w-lg text-base leading-relaxed text-text-muted">
          Answers are grounded in the indexed reports. Each response includes
          the source PDF and page number so you can verify quickly.
        </p>
      </div>

      <div className="grid w-full max-w-2xl grid-cols-1 gap-3 md:grid-cols-2">
        {SUGGESTED.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => onPick(s)}
            className="glass group cursor-pointer rounded-2xl p-4 text-left text-sm leading-snug shadow-[var(--shadow-sm)] transition-[border-color,transform] duration-200 hover:border-primary hover:scale-[1.01] focus-visible:border-primary"
          >
            <span className="mb-1 inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-primary">
              <SparkleIcon className="h-3 w-3" />
              Prompt
            </span>
            <span className="block text-text-muted transition-colors group-hover:text-text">
              {s}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

/* --------------------------------- Bubble ---------------------------------- */

function Bubble({ message }: { message: Message }) {
  const isUser = message.role === "user";
  return (
    <article
      className={`reveal flex gap-3 ${isUser ? "flex-row-reverse" : "flex-row"}`}
    >
      <Avatar role={message.role} />
      <div
        className={`flex min-w-0 flex-col gap-2 ${
          isUser ? "items-end" : "items-start"
        }`}
      >
        <div
          className={
            isUser
              ? "max-w-[85%] rounded-[22px] rounded-tr-lg bg-primary px-4 py-3 text-[15px] leading-relaxed text-white shadow-[var(--shadow-md)]"
              : "glass max-w-[85%] rounded-[22px] rounded-tl-lg px-4 py-3 text-[15px] leading-relaxed text-text shadow-[var(--shadow-sm)]"
          }
        >
          {message.error ? (
            <span className="text-[color:var(--danger)]">{message.error}</span>
          ) : message.content ? (
            <RenderedText text={message.content} streaming={message.streaming} />
          ) : message.streaming ? (
            <TypingDots />
          ) : null}
        </div>

        {!isUser && message.sources && message.sources.length > 0 && (
          <Sources sources={message.sources} />
        )}
      </div>
    </article>
  );
}

function RenderedText({
  text,
  streaming,
}: {
  text: string;
  streaming?: boolean;
}) {
  const paragraphs = text.split(/\n{2,}/);
  return (
    <div className="whitespace-pre-wrap">
      {paragraphs.map((p, i) => (
        <p key={i} className={i === 0 ? "" : "mt-3"}>
          {p}
          {streaming && i === paragraphs.length - 1 ? (
            <span className="caret" aria-hidden />
          ) : null}
        </p>
      ))}
    </div>
  );
}

function TypingDots() {
  return (
    <span
      className="inline-flex items-center gap-1 py-1"
      aria-label="Assistant is thinking"
    >
      <span className="dot" />
      <span className="dot" />
      <span className="dot" />
    </span>
  );
}

/* --------------------------------- Sources --------------------------------- */

function Sources({ sources }: { sources: Source[] }) {
  const [openId, setOpenId] = useState<number | null>(null);
  return (
    <div className="flex w-full max-w-[85%] flex-col gap-2">
      <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-wider text-text-subtle">
        <BookIcon className="h-3.5 w-3.5" />
        Sources
      </div>
      <div className="flex flex-wrap gap-1.5">
        {sources.map((s) => {
          const active = openId === s.id;
          const label = `${prettySource(s.source)}${
            s.page != null ? ` · p. ${s.page}` : ""
          }`;
          return (
            <button
              key={s.id}
              type="button"
              onClick={() => setOpenId(active ? null : s.id)}
              aria-expanded={active}
              className={`cursor-pointer rounded-full border px-2.5 py-1 text-xs font-medium backdrop-blur-md transition-colors duration-200 ${
                active
                  ? "border-accent bg-[color:color-mix(in_srgb,var(--accent)_15%,var(--glass))] text-accent"
                  : "border-[color:var(--glass-border)] bg-[color:var(--glass)] text-text-muted hover:border-primary hover:text-primary"
              }`}
            >
              [{s.id}] {label}
            </button>
          );
        })}
      </div>
      {openId != null &&
        (() => {
          const s = sources.find((x) => x.id === openId);
          if (!s) return null;
          return (
            <div className="reveal glass rounded-2xl px-3.5 py-3 text-sm leading-relaxed text-text-muted">
              <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-text-subtle">
                Excerpt · {prettySource(s.source)}
                {s.page != null ? ` · p. ${s.page}` : ""}
              </div>
              <p className="whitespace-pre-wrap">{s.excerpt}</p>
            </div>
          );
        })()}
    </div>
  );
}

/* --------------------------------- Avatar ---------------------------------- */

function Avatar({ role }: { role: "user" | "assistant" }) {
  if (role === "user") {
    return (
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-bg-elev text-text-muted shadow-sm ring-1 ring-border">
        <UserIcon className="h-4 w-4" />
      </div>
    );
  }
  return (
    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-white shadow-sm">
      <SparkleIcon className="h-4 w-4" />
    </div>
  );
}

/* -------------------------------- Composer --------------------------------- */

function Composer({
  value,
  onChange,
  onSend,
  onStop,
  busy,
  canSend,
  textareaRef,
  error,
}: {
  value: string;
  onChange: (v: string) => void;
  onSend: () => void;
  onStop: () => void;
  busy: boolean;
  canSend: boolean;
  textareaRef: React.RefObject<HTMLTextAreaElement | null>;
  error: string | null;
}) {
  return (
    <div className="glass-strong sticky bottom-0 z-10 px-4 py-4 md:px-8">
      <div className="mx-auto w-full max-w-3xl">
        {error && (
          <div className="mb-2 rounded-lg border border-[color:var(--danger)]/40 bg-[color:color-mix(in_srgb,var(--danger)_10%,var(--bg-elev))] px-3 py-2 text-sm text-[color:var(--danger)]">
            {error}
          </div>
        )}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (canSend) onSend();
          }}
          className="glass flex items-end gap-2 rounded-[20px] p-2 shadow-[var(--shadow-md)] transition-colors focus-within:border-primary"
        >
          <label htmlFor="composer" className="sr-only">
            Ask a question
          </label>
          <textarea
            id="composer"
            ref={textareaRef}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                if (canSend) onSend();
              }
            }}
            rows={1}
            placeholder="Ask about Bihar…"
            className="flex-1 resize-none bg-transparent px-3 py-2.5 text-[15px] leading-relaxed text-text placeholder:text-text-subtle focus:outline-none"
          />
          {busy ? (
            <button
              type="button"
              onClick={onStop}
              className="inline-flex h-11 w-11 cursor-pointer items-center justify-center rounded-xl border border-border bg-bg-sunken text-text-muted transition-colors duration-200 hover:border-primary hover:text-primary"
              aria-label="Stop"
            >
              <StopIcon className="h-4 w-4" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!canSend}
              className="inline-flex h-11 min-w-11 cursor-pointer items-center justify-center gap-2 rounded-xl bg-primary px-4 text-sm font-semibold text-white transition-colors duration-200 hover:bg-primary-hover disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-primary"
              aria-label="Send"
            >
              <SendIcon className="h-4 w-4" />
              <span className="hidden sm:inline">Send</span>
            </button>
          )}
        </form>
        <p className="mt-2 text-center text-[11px] text-text-subtle">
          Enter to send · Shift + Enter for a new line
        </p>
      </div>
    </div>
  );
}

/* ---------------------------------- Icons ---------------------------------- */

function BrandMark({ small, large }: { small?: boolean; large?: boolean }) {
  const size = large ? "h-14 w-14" : small ? "h-6 w-6" : "h-8 w-8";
  return (
    <div
      className={`${size} inline-flex items-center justify-center rounded-xl bg-gradient-to-br from-primary to-accent text-white shadow-md`}
    >
      <SparkleIcon className={large ? "h-6 w-6" : "h-4 w-4"} />
    </div>
  );
}

function SparkleIcon({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      className={className}
      aria-hidden
    >
      <path
        d="M12 3l1.8 4.6L18 9l-4.2 1.4L12 15l-1.8-4.6L6 9l4.2-1.4L12 3z"
        fill="currentColor"
      />
      <path
        d="M18.5 14l.9 2.3L21.5 17l-2.1.7L18.5 20l-.9-2.3L15.5 17l2.1-.7.9-2.3z"
        fill="currentColor"
        opacity=".7"
      />
    </svg>
  );
}

function SendIcon({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden>
      <path
        d="M4 12l16-8-6 18-2-8-8-2z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}

function StopIcon({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden>
      <rect
        x="6"
        y="6"
        width="12"
        height="12"
        rx="2"
        fill="currentColor"
      />
    </svg>
  );
}

function PlusIcon({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden>
      <path
        d="M12 5v14M5 12h14"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

function UserIcon({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden>
      <circle cx="12" cy="8" r="4" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M4 20c1.6-4 5-6 8-6s6.4 2 8 6"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
    </svg>
  );
}

function BookIcon({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" className={className} aria-hidden>
      <path
        d="M4 5a2 2 0 012-2h11v16H6a2 2 0 00-2 2V5z"
        stroke="currentColor"
        strokeWidth="1.6"
      />
      <path
        d="M8 7h6M8 11h6"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
    </svg>
  );
}
