import { useState, useEffect, useCallback, useRef } from "react"

// ─────────────────────────────────────────────────────────────
// DESIGN TOKENS
// Space Grotesk for display/data, Inter for body copy
// Violet accent, emerald/rose for sentiment polarity
// Sharp left-border cards instead of radius — IDE/terminal feel
// ─────────────────────────────────────────────────────────────
const T = {
  bg:        "#0D0D0F",
  surface:   "#141418",
  surfaceHi: "#1C1C22",
  border:    "#242428",
  borderHi:  "#383840",
  accent:    "#7C6EFA",
  accentDim: "#7C6EFA18",
  pos:       "#10B981",
  posDim:    "#10B98118",
  neg:       "#F43F5E",
  negDim:    "#F43F5E18",
  warn:      "#F59E0B",
  text:      "#EEECEA",
  sub:       "#A8A6A0",
  muted:     "#5A5858",
  font:      "'Space Grotesk', system-ui, sans-serif",
  body:      "'Inter', system-ui, sans-serif",
}

// ─────────────────────────────────────────────────────────────
// GLOBAL STYLES
// ─────────────────────────────────────────────────────────────
const GLOBAL_CSS = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { font-size: 16px; }
  body { background: ${T.bg}; color: ${T.text}; font-family: ${T.body}; -webkit-font-smoothing: antialiased; }
  input, textarea, select, button { font-family: inherit; }
  input:focus, textarea:focus, select:focus { outline: none; }
  ::-webkit-scrollbar { width: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: ${T.border}; border-radius: 3px; }
  @keyframes spin   { to { transform: rotate(360deg); } }
  @keyframes fadeUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
  @keyframes pulse  { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
  @keyframes arcIn  { from { stroke-dashoffset: 220; } }
`

// ─────────────────────────────────────────────────────────────
// API
// ─────────────────────────────────────────────────────────────
const BASE = "/api"

async function call(path, opts = {}) {
  const r = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  })
  const d = await r.json()
  if (!r.ok) throw new Error(d.error || `HTTP ${r.status}`)
  return d
}

const api = {
  health:          ()    => call("/health"),
  products:        ()    => call("/products"),
  product:         (id)  => call(`/products/${id}`),
  createProduct:   (b)   => call("/products",        { method:"POST", body:JSON.stringify(b) }),
  submitReview:    (b)   => call("/reviews",         { method:"POST", body:JSON.stringify(b) }),
  previewSentiment:(t)   => call("/reviews/preview", { method:"POST", body:JSON.stringify({ text:t }) }),
}

// ─────────────────────────────────────────────────────────────
// PRIMITIVE COMPONENTS
// ─────────────────────────────────────────────────────────────

function Spinner({ size = 28 }) {
  return (
    <div style={{
      width: size, height: size,
      border: `2.5px solid ${T.border}`,
      borderTopColor: T.accent,
      borderRadius: "50%",
      animation: "spin 0.65s linear infinite",
      flexShrink: 0,
    }} />
  )
}

function Tag({ color = T.accent, children }) {
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 3,
      background: color + "18", color,
      border: `1px solid ${color}30`,
      borderRadius: 4, padding: "2px 7px",
      fontSize: 11, fontWeight: 700,
      fontFamily: T.font, letterSpacing: "0.04em",
    }}>
      {children}
    </span>
  )
}

function Btn({ children, variant = "primary", disabled, onClick, style }) {
  const base = {
    display: "inline-flex", alignItems: "center", gap: 6,
    border: "none", borderRadius: 7, padding: "9px 18px",
    fontSize: 13, fontWeight: 600, cursor: disabled ? "not-allowed" : "pointer",
    fontFamily: T.font, transition: "opacity 0.15s, background 0.15s",
    opacity: disabled ? 0.5 : 1,
  }
  const variants = {
    primary: { background: T.accent, color: "#fff" },
    ghost:   { background: "transparent", color: T.sub, border: `1px solid ${T.border}` },
    danger:  { background: T.neg + "18", color: T.neg, border: `1px solid ${T.neg}30` },
  }
  return (
    <button style={{ ...base, ...variants[variant], ...style }} disabled={disabled} onClick={onClick}>
      {children}
    </button>
  )
}

function Field({ label, error, children }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      {label && (
        <label style={{ fontSize: 11, fontWeight: 700, color: T.muted,
          textTransform: "uppercase", letterSpacing: "0.06em", fontFamily: T.font }}>
          {label}
        </label>
      )}
      {children}
      {error && <span style={{ fontSize: 12, color: T.neg }}>{error}</span>}
    </div>
  )
}

const inputBase = {
  background: T.bg, border: `1px solid ${T.border}`,
  borderRadius: 7, color: T.text,
  padding: "9px 12px", fontSize: 13,
  width: "100%", transition: "border-color 0.15s",
}

function Input({ style, ...props }) {
  const [focus, setFocus] = useState(false)
  return (
    <input
      style={{ ...inputBase, borderColor: focus ? T.accent : T.border, ...style }}
      onFocus={() => setFocus(true)} onBlur={() => setFocus(false)}
      {...props}
    />
  )
}

function Textarea({ style, ...props }) {
  const [focus, setFocus] = useState(false)
  return (
    <textarea
      style={{ ...inputBase, resize: "vertical", minHeight: 110,
        fontFamily: T.body, lineHeight: 1.6,
        borderColor: focus ? T.accent : T.border, ...style }}
      onFocus={() => setFocus(true)} onBlur={() => setFocus(false)}
      {...props}
    />
  )
}

function Select({ children, style, ...props }) {
  const [focus, setFocus] = useState(false)
  return (
    <select
      style={{ ...inputBase, appearance: "none", cursor: "pointer",
        borderColor: focus ? T.accent : T.border, ...style }}
      onFocus={() => setFocus(true)} onBlur={() => setFocus(false)}
      {...props}
    >
      {children}
    </select>
  )
}

// ─────────────────────────────────────────────────────────────
// TOAST
// ─────────────────────────────────────────────────────────────

function Toast({ msg, type, onDone }) {
  useEffect(() => {
    const t = setTimeout(onDone, 3200)
    return () => clearTimeout(t)
  }, [onDone])
  const c = type === "error" ? T.neg : T.pos
  return (
    <div style={{
      position: "fixed", bottom: 24, right: 24, zIndex: 9999,
      background: T.surface, border: `1px solid ${c}40`,
      borderLeft: `3px solid ${c}`,
      borderRadius: 8, padding: "12px 18px",
      color: T.text, fontSize: 13, fontWeight: 500,
      boxShadow: "0 8px 40px rgba(0,0,0,0.5)",
      animation: "fadeUp 0.2s ease",
      maxWidth: 320,
    }}>
      {msg}
    </div>
  )
}

function useToast() {
  const [toast, setToast] = useState(null)
  const show = useCallback((msg, type = "success") => setToast({ msg, type }), [])
  const el = toast ? <Toast {...toast} onDone={() => setToast(null)} /> : null
  return [show, el]
}

// ─────────────────────────────────────────────────────────────
// STAR RATING
// ─────────────────────────────────────────────────────────────

function Stars({ value, onChange, size = 20 }) {
  const [hover, setHover] = useState(0)
  const interactive = !!onChange
  return (
    <div style={{ display: "flex", gap: 2 }}>
      {[1,2,3,4,5].map(n => (
        <span key={n}
          onClick={() => interactive && onChange(n)}
          onMouseEnter={() => interactive && setHover(n)}
          onMouseLeave={() => interactive && setHover(0)}
          style={{
            fontSize: size, lineHeight: 1,
            color: n <= (hover || value) ? T.warn : T.border,
            cursor: interactive ? "pointer" : "default",
            transition: "color 0.1s",
            userSelect: "none",
          }}
        >★</span>
      ))}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// AI SCORE RING
// Animates stroke-dashoffset on mount — the signature element
// ─────────────────────────────────────────────────────────────

function AIRing({ score, size = 72 }) {
  const [animated, setAnimated] = useState(false)
  useEffect(() => { const t = setTimeout(() => setAnimated(true), 80); return () => clearTimeout(t) }, [])

  if (score === undefined || score === null || score === 0) return null

  const r = size * 0.38
  const cx = size / 2, cy = size / 2
  const circ = 2 * Math.PI * r
  const offset = animated ? circ - (score / 100) * circ : circ
  const c = score >= 70 ? T.pos : score >= 45 ? T.warn : T.neg
  const label = score >= 70 ? "Strong" : score >= 45 ? "Mixed" : "Weak"
  const fs = size * 0.22

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, flexShrink: 0 }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={T.border} strokeWidth={size*0.08} />
        <circle cx={cx} cy={cy} r={r} fill="none"
          stroke={c} strokeWidth={size*0.08}
          strokeDasharray={circ}
          strokeDashoffset={offset}
          strokeLinecap="round"
          transform={`rotate(-90 ${cx} ${cy})`}
          style={{ transition: "stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1)" }}
        />
        <text x={cx} y={cy + fs*0.38} textAnchor="middle"
          fill={c} fontSize={fs} fontWeight="700"
          fontFamily={T.font}>
          {Math.round(score)}
        </text>
      </svg>
      <div>
        <div style={{ fontSize: 10, color: T.muted, textTransform: "uppercase",
          letterSpacing: "0.07em", fontFamily: T.font, fontWeight: 700, marginBottom: 2 }}>
          AI Score
        </div>
        <div style={{ color: c, fontWeight: 700, fontSize: 13, fontFamily: T.font }}>{label}</div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// MODEL COMPARISON BARS
// Shows FFC vs LSTM vs Ensemble — core educational feature
// ─────────────────────────────────────────────────────────────

function ModelBars({ sentiment }) {
  if (!sentiment) return null
  const models = [
    ["FFC",      sentiment.ffc],
    ["LSTM",     sentiment.lstm],
    ["Ensemble", sentiment.ensemble],
  ].filter(([, v]) => v)

  return (
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
      {models.map(([name, m]) => {
        const pos = m.scores?.POSITIVE ?? 0
        const c   = m.label === "POSITIVE" ? T.pos : T.neg
        return (
          <div key={name} style={{
            flex: "1 1 120px", background: T.bg,
            borderLeft: `2px solid ${c}`,
            borderRadius: "0 6px 6px 0",
            padding: "8px 10px",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
              <span style={{ fontSize: 10, fontFamily: T.font, fontWeight: 700,
                color: T.muted, textTransform: "uppercase", letterSpacing: "0.06em" }}>
                {name}
              </span>
              <span style={{ fontSize: 11, fontWeight: 700, color: c, fontFamily: T.font }}>
                {m.label === "POSITIVE" ? "+" : "−"}{Math.round(Math.max(pos, 1-pos) * 100)}%
              </span>
            </div>
            <div style={{ height: 3, background: T.border, borderRadius: 2 }}>
              <div style={{
                width: `${Math.round(pos * 100)}%`, height: "100%",
                background: c, borderRadius: 2,
                transition: "width 0.6s cubic-bezier(0.4,0,0.2,1)",
              }} />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", marginTop: 3 }}>
              <span style={{ fontSize: 10, color: T.muted }}>NEG</span>
              <span style={{ fontSize: 10, color: T.muted }}>POS</span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// LIVE SENTIMENT PREVIEW PANEL
// ─────────────────────────────────────────────────────────────

function SentimentPreview({ text }) {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(false)
  const timer = useRef(null)

  useEffect(() => {
    if (text.length < 20) { setData(null); return }
    clearTimeout(timer.current)
    setLoading(true)
    timer.current = setTimeout(async () => {
      try {
        const r = await api.previewSentiment(text)
        setData(r.sentiment)
      } catch { /* silent fail */ }
      setLoading(false)
    }, 550)
    return () => clearTimeout(timer.current)
  }, [text])

  if (text.length < 20) return null

  return (
    <div style={{
      background: T.bg, border: `1px solid ${T.border}`,
      borderRadius: 8, padding: "12px 14px",
      animation: "fadeUp 0.2s ease",
    }}>
      <div style={{ fontSize: 10, fontWeight: 700, color: T.muted,
        textTransform: "uppercase", letterSpacing: "0.06em",
        fontFamily: T.font, marginBottom: 8 }}>
        Live sentiment
      </div>
      {loading
        ? <div style={{ display: "flex", alignItems: "center", gap: 8, color: T.muted, fontSize: 12 }}>
            <Spinner size={14} /> Analysing...
          </div>
        : data
          ? <ModelBars sentiment={data} />
          : null
      }
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// REVIEW CARD
// ─────────────────────────────────────────────────────────────

function ReviewCard({ review }) {
  const [open, setOpen] = useState(false)
  const ens = review.sentiment?.ensemble
  const c   = ens?.label === "POSITIVE" ? T.pos : T.neg

  return (
    <div style={{
      background: T.surface,
      borderLeft: `2px solid ${c}`,
      borderRadius: "0 10px 10px 0",
      border: `1px solid ${T.border}`,
      borderLeftColor: c,
      padding: "16px 18px",
      animation: "fadeUp 0.2s ease",
    }}>
      {/* Header row */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, marginBottom: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 34, height: 34, borderRadius: "50%",
            background: T.accent + "22",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 14, fontWeight: 700, color: T.accent, fontFamily: T.font,
            flexShrink: 0,
          }}>
            {review.author?.[0]?.toUpperCase()}
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 2 }}>{review.author}</div>
            <Stars value={review.rating} size={13} />
          </div>
        </div>
        {ens && (
          <Tag color={c}>{ens.label === "POSITIVE" ? "▲" : "▼"} {ens.label}</Tag>
        )}
      </div>

      {/* Review text */}
      <p style={{ fontSize: 13, lineHeight: 1.7, color: "#CCC9C3", margin: "0 0 12px" }}>
        {review.text}
      </p>

      {/* Model breakdown toggle */}
      {review.sentiment && (
        <>
          <button
            onClick={() => setOpen(o => !o)}
            style={{
              background: "none", border: "none", cursor: "pointer",
              color: T.muted, fontSize: 11, fontFamily: T.font, fontWeight: 600,
              textTransform: "uppercase", letterSpacing: "0.05em",
              padding: "4px 0", display: "flex", alignItems: "center", gap: 4,
            }}
          >
            <span style={{ transition: "transform 0.2s", display: "inline-block",
              transform: open ? "rotate(90deg)" : "none" }}>▶</span>
            {open ? "Hide" : "Show"} FFC vs LSTM
          </button>
          {open && (
            <div style={{ marginTop: 10, animation: "fadeUp 0.15s ease" }}>
              <ModelBars sentiment={review.sentiment} />
            </div>
          )}
        </>
      )}

      <div style={{ fontSize: 11, color: T.muted, marginTop: 10 }}>
        {new Date(review.created_at).toLocaleDateString("en-GB", {
          day: "numeric", month: "short", year: "numeric",
        })}
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// REVIEW FORM
// ─────────────────────────────────────────────────────────────

function ReviewForm({ productId, onDone }) {
  const [author,  setAuthor]  = useState("")
  const [text,    setText]    = useState("")
  const [rating,  setRating]  = useState(0)
  const [busy,    setBusy]    = useState(false)
  const [errors,  setErrors]  = useState({})

  function validate() {
    const e = {}
    if (!author.trim())         e.author = "Enter your name"
    if (text.trim().length < 10) e.text  = "Write at least 10 characters"
    if (!rating)                 e.rating = "Choose a rating"
    setErrors(e)
    return Object.keys(e).length === 0
  }

  async function submit() {
    if (!validate()) return
    setBusy(true)
    try {
      const res = await api.submitReview({ product_id: productId, author, text, rating })
      onDone(res)
      setAuthor(""); setText(""); setRating(0); setErrors({})
    } catch (e) {
      setErrors({ form: e.message })
    }
    setBusy(false)
  }

  return (
    <div style={{
      background: T.surface, border: `1px solid ${T.border}`,
      borderRadius: 10, padding: "20px 22px",
    }}>
      <div style={{ fontFamily: T.font, fontWeight: 700, fontSize: 15, marginBottom: 18 }}>
        Write a review
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 14 }}>
        <Field label="Name" error={errors.author}>
          <Input value={author} onChange={e => setAuthor(e.target.value)} placeholder="Your name" />
        </Field>
        <Field label="Rating" error={errors.rating}>
          <div style={{ padding: "6px 0" }}>
            <Stars value={rating} onChange={setRating} size={22} />
          </div>
        </Field>
      </div>

      <Field label="Review" error={errors.text}>
        <Textarea
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="What did you think about this product?"
        />
        <div style={{ fontSize: 11, color: T.muted, marginTop: 4 }}>
          {text.length} chars · {text.length >= 20 ? "live preview active" : "keep typing for sentiment preview"}
        </div>
      </Field>

      <SentimentPreview text={text} />

      {errors.form && (
        <div style={{ color: T.neg, fontSize: 12, marginTop: 10 }}>{errors.form}</div>
      )}

      <div style={{ marginTop: 16 }}>
        <Btn onClick={submit} disabled={busy}>
          {busy ? <><Spinner size={13} /> Submitting…</> : "Submit review"}
        </Btn>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// PRODUCT STATS BAR
// ─────────────────────────────────────────────────────────────

function StatChip({ label, value, color }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <span style={{ fontSize: 10, color: T.muted, fontWeight: 700,
        textTransform: "uppercase", letterSpacing: "0.06em", fontFamily: T.font }}>
        {label}
      </span>
      <span style={{ fontSize: 16, fontWeight: 700, fontFamily: T.font, color: color || T.text }}>
        {value}
      </span>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// ADD PRODUCT MODAL
// ─────────────────────────────────────────────────────────────

const CATEGORIES = ["Electronics","Footwear","Kitchen","Books & Reading","Fashion","Sports","Home","Other"]

function AddProductModal({ onClose, onCreated }) {
  const [name, setName]       = useState("")
  const [desc, setDesc]       = useState("")
  const [cat,  setCat]        = useState("Electronics")
  const [busy, setBusy]       = useState(false)
  const [err,  setErr]        = useState("")

  async function create() {
    if (!name.trim() || !desc.trim()) return setErr("Name and description are required")
    setBusy(true)
    try {
      const p = await api.createProduct({ name: name.trim(), description: desc.trim(), category: cat })
      onCreated(p)
      onClose()
    } catch (e) { setErr(e.message) }
    setBusy(false)
  }

  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.75)",
      display: "flex", alignItems: "center", justifyContent: "center",
      zIndex: 500, padding: 16,
    }} onClick={e => e.target === e.currentTarget && onClose()}>
      <div style={{
        background: T.surface, border: `1px solid ${T.border}`,
        borderRadius: 12, padding: "24px 26px",
        width: "100%", maxWidth: 480,
        animation: "fadeUp 0.2s ease",
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
          <span style={{ fontFamily: T.font, fontWeight: 700, fontSize: 16 }}>Add product</span>
          <button onClick={onClose} style={{
            background: "none", border: "none", color: T.muted,
            fontSize: 18, cursor: "pointer", lineHeight: 1,
          }}>✕</button>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <Field label="Product name">
            <Input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. Sony WH-1000XM5" />
          </Field>
          <Field label="Description">
            <Textarea value={desc} onChange={e => setDesc(e.target.value)}
              placeholder="Brief product description" style={{ minHeight: 80 }} />
          </Field>
          <Field label="Category">
            <Select value={cat} onChange={e => setCat(e.target.value)}>
              {CATEGORIES.map(c => <option key={c}>{c}</option>)}
            </Select>
          </Field>
        </div>

        {err && <div style={{ color: T.neg, fontSize: 12, marginTop: 10 }}>{err}</div>}

        <div style={{ display: "flex", gap: 8, marginTop: 20, justifyContent: "flex-end" }}>
          <Btn variant="ghost" onClick={onClose}>Cancel</Btn>
          <Btn disabled={busy} onClick={create}>
            {busy ? <><Spinner size={13} /> Creating…</> : "Create product"}
          </Btn>
        </div>
      </div>
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// PRODUCT LIST CARD
// ─────────────────────────────────────────────────────────────

function ProductCard({ product, onClick }) {
  const [hov, setHov] = useState(false)
  const s = product.stats || {}
  const pp = s.sentiment_breakdown?.positive_pct
  const c  = pp >= 60 ? T.pos : pp >= 40 ? T.warn : pp > 0 ? T.neg : T.muted

  return (
    <div
      onClick={onClick}
      onMouseEnter={() => setHov(true)}
      onMouseLeave={() => setHov(false)}
      style={{
        background: T.surface,
        border: `1px solid ${hov ? T.borderHi : T.border}`,
        borderLeft: `2px solid ${hov ? T.accent : T.border}`,
        borderRadius: "0 10px 10px 0",
        padding: "16px 20px",
        cursor: "pointer",
        transition: "border-color 0.15s",
        display: "flex", justifyContent: "space-between",
        alignItems: "center", gap: 16,
      }}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
          <span style={{
            fontSize: 10, fontFamily: T.font, fontWeight: 700, color: T.muted,
            textTransform: "uppercase", letterSpacing: "0.05em",
            background: T.border, borderRadius: 3, padding: "2px 6px",
          }}>
            {product.category}
          </span>
          {s.review_count > 0 && (
            <Tag color={c}>{pp}% positive</Tag>
          )}
        </div>

        <div style={{ fontFamily: T.font, fontWeight: 700, fontSize: 15,
          color: hov ? T.accent : T.text, transition: "color 0.15s",
          marginBottom: 4, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {product.name}
        </div>
        <div style={{ fontSize: 12, color: T.sub, lineHeight: 1.5,
          display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
          {product.description}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 14, marginTop: 10 }}>
          {s.review_count > 0
            ? <>
                <Stars value={Math.round(s.avg_rating)} size={13} />
                <span style={{ fontSize: 11, color: T.muted }}>
                  {s.avg_rating?.toFixed(1)} · {s.review_count} review{s.review_count !== 1 ? "s" : ""}
                </span>
              </>
            : <span style={{ fontSize: 11, color: T.muted }}>No reviews yet</span>
          }
        </div>
      </div>

      <AIRing score={s.ai_score} size={64} />
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// PRODUCT DETAIL PAGE
// ─────────────────────────────────────────────────────────────

function ProductPage({ productId, onBack }) {
  const [data,    setData]    = useState(null)
  const [loading, setLoading] = useState(true)
  const [showToast, toastEl]  = useToast()

  const load = useCallback(async () => {
    setLoading(true)
    try { setData(await api.product(productId)) }
    catch (e) { showToast(e.message, "error") }
    setLoading(false)
  }, [productId])

  useEffect(() => { load() }, [load])

  if (loading) return (
    <div style={{ display: "flex", justifyContent: "center", paddingTop: 80 }}>
      <Spinner size={36} />
    </div>
  )
  if (!data) return null

  const { product, reviews: reviewList } = data
  const s = product.stats || {}

  return (
    <div style={{ maxWidth: 860, margin: "0 auto", padding: "28px 20px" }}>
      <Btn variant="ghost" onClick={onBack} style={{ marginBottom: 20, fontSize: 12 }}>
        ← Products
      </Btn>

      {/* Product header card */}
      <div style={{
        background: T.surface, border: `1px solid ${T.border}`,
        borderTop: `2px solid ${T.accent}`,
        borderRadius: "0 0 10px 10px",
        padding: "22px 24px", marginBottom: 20,
      }}>
        <div style={{ display: "flex", justifyContent: "space-between",
          alignItems: "flex-start", gap: 20, flexWrap: "wrap" }}>
          <div style={{ flex: 1 }}>
            <span style={{
              fontSize: 10, fontFamily: T.font, fontWeight: 700, color: T.muted,
              textTransform: "uppercase", letterSpacing: "0.05em",
              background: T.border, borderRadius: 3, padding: "2px 6px",
              display: "inline-block", marginBottom: 8,
            }}>{product.category}</span>
            <h1 style={{ fontFamily: T.font, fontSize: 22, fontWeight: 800,
              letterSpacing: "-0.02em", marginBottom: 8, color: T.text }}>
              {product.name}
            </h1>
            <p style={{ fontSize: 13, color: T.sub, lineHeight: 1.65 }}>
              {product.description}
            </p>
          </div>
          <AIRing score={s.ai_score} size={80} />
        </div>

        {/* Stats row */}
        {s.review_count > 0 && (
          <div style={{
            display: "flex", gap: 28, marginTop: 20,
            paddingTop: 18, borderTop: `1px solid ${T.border}`,
            flexWrap: "wrap",
          }}>
            <div>
              <div style={{ fontSize: 10, color: T.muted, fontWeight: 700,
                textTransform: "uppercase", letterSpacing: "0.06em",
                fontFamily: T.font, marginBottom: 4 }}>Rating</div>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <Stars value={Math.round(s.avg_rating)} size={14} />
                <span style={{ fontFamily: T.font, fontWeight: 700, fontSize: 15 }}>
                  {s.avg_rating?.toFixed(1)}
                </span>
              </div>
            </div>
            <StatChip label="Reviews"  value={s.review_count} />
            <StatChip label="Positive" value={`${s.sentiment_breakdown?.positive_pct}%`} color={T.pos} />
            <StatChip label="FFC pos"  value={`${((s.avg_ffc_positive  || 0)*100).toFixed(0)}%`} color={T.accent} />
            <StatChip label="LSTM pos" value={`${((s.avg_lstm_positive || 0)*100).toFixed(0)}%`} color={T.accent} />
          </div>
        )}
      </div>

      {/* Review form */}
      <div style={{ marginBottom: 20 }}>
        <ReviewForm productId={productId} onDone={() => { showToast("Review submitted"); load() }} />
      </div>

      {/* Reviews */}
      <div style={{ fontFamily: T.font, fontWeight: 700, fontSize: 13,
        color: T.muted, textTransform: "uppercase", letterSpacing: "0.06em",
        marginBottom: 12 }}>
        {reviewList.length} Review{reviewList.length !== 1 ? "s" : ""}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        {reviewList.length === 0
          ? <div style={{
              background: T.surface, border: `1px solid ${T.border}`,
              borderRadius: 10, padding: "40px 20px",
              textAlign: "center", color: T.muted, fontSize: 13,
            }}>
              No reviews yet — be the first.
            </div>
          : reviewList.map(r => <ReviewCard key={r._id} review={r} />)
        }
      </div>

      {toastEl}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// PRODUCTS LIST PAGE
// ─────────────────────────────────────────────────────────────

function ProductsPage({ onSelect }) {
  const [list,    setList]    = useState([])
  const [loading, setLoading] = useState(true)
  const [search,  setSearch]  = useState("")
  const [modal,   setModal]   = useState(false)
  const [showToast, toastEl]  = useToast()

  useEffect(() => {
    api.products()
      .then(setList)
      .catch(e => showToast(e.message, "error"))
      .finally(() => setLoading(false))
  }, [])

  const filtered = list.filter(p =>
    p.name.toLowerCase().includes(search.toLowerCase()) ||
    (p.category || "").toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div style={{ maxWidth: 860, margin: "0 auto", padding: "28px 20px" }}>
      {/* Page header */}
      <div style={{ display: "flex", justifyContent: "space-between",
        alignItems: "flex-start", marginBottom: 24, flexWrap: "wrap", gap: 14 }}>
        <div>
          <h1 style={{ fontFamily: T.font, fontSize: 22, fontWeight: 800,
            letterSpacing: "-0.02em", marginBottom: 4 }}>
            Products
          </h1>
          <p style={{ fontSize: 12, color: T.muted }}>
            {list.length} products · ranked by AI sentiment score
          </p>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <Input
            value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search…" style={{ width: 180 }}
          />
          <Btn onClick={() => setModal(true)}>+ Add</Btn>
        </div>
      </div>

      {/* List */}
      {loading
        ? <div style={{ display: "flex", justifyContent: "center", paddingTop: 60 }}>
            <Spinner size={32} />
          </div>
        : filtered.length === 0
          ? <div style={{
              background: T.surface, border: `1px solid ${T.border}`,
              borderRadius: 10, padding: "48px 20px",
              textAlign: "center", color: T.muted, fontSize: 13,
            }}>
              {search ? "No products match your search." : "No products yet — add one to get started."}
            </div>
          : <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {filtered.map(p => (
                <ProductCard key={p._id} product={p} onClick={() => onSelect(p._id)} />
              ))}
            </div>
      }

      {modal && (
        <AddProductModal
          onClose={() => setModal(false)}
          onCreated={p => { setList(prev => [p, ...prev]); showToast(`${p.name} added`) }}
        />
      )}

      {toastEl}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────
// NAV BAR
// ─────────────────────────────────────────────────────────────

function Nav({ onHome, health }) {
  return (
    <nav style={{
      background: T.surface, borderBottom: `1px solid ${T.border}`,
      height: 54, display: "flex", alignItems: "center",
      justifyContent: "space-between", padding: "0 24px",
      position: "sticky", top: 0, zIndex: 100,
    }}>
      <button onClick={onHome} style={{
        background: "none", border: "none", cursor: "pointer",
        fontFamily: T.font, fontWeight: 800, fontSize: 17,
        color: T.accent, letterSpacing: "-0.03em",
      }}>
        Rev<span style={{ color: T.text }}>X</span>
      </button>

      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {health && (
          <>
            {health.ffc_model  && <Tag color={T.pos}>FFC</Tag>}
            {health.lstm_model && <Tag color={T.pos}>LSTM</Tag>}
            {!health.ffc_model && !health.lstm_model &&
              <Tag color={T.neg}>No models loaded</Tag>}
            <span style={{ width: 1, height: 14, background: T.border }} />
            <span style={{ fontSize: 11, color: T.muted, fontFamily: T.font }}>
              {health.vocab_size?.toLocaleString()} vocab
            </span>
          </>
        )}
      </div>
    </nav>
  )
}

// ─────────────────────────────────────────────────────────────
// ROOT
// ─────────────────────────────────────────────────────────────

export default function App() {
  const [page,      setPage]      = useState("list")
  const [productId, setProductId] = useState(null)
  const [health,    setHealth]    = useState(null)

  useEffect(() => { api.health().then(setHealth).catch(() => {}) }, [])

  return (
    <>
      <style>{GLOBAL_CSS}</style>
      <div style={{ minHeight: "100vh", background: T.bg }}>
        <Nav
          onHome={() => { setPage("list"); setProductId(null) }}
          health={health}
        />
        {page === "list" && (
          <ProductsPage onSelect={id => { setProductId(id); setPage("detail") }} />
        )}
        {page === "detail" && productId && (
          <ProductPage
            productId={productId}
            onBack={() => { setPage("list"); setProductId(null) }}
          />
        )}
      </div>
    </>
  )
}
