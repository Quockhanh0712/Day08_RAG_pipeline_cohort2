import { useMemo, useState } from 'react'

const API_URL = import.meta.env.VITE_RAG_API_URL || 'http://127.0.0.1:8000'

const EXAMPLE_QUESTIONS = [
  'Hình phạt cho tội tàng trữ trái phép chất ma túy là gì?',
  'Luật phòng chống ma túy quy định gì về cai nghiện?',
  'Các bài báo nói gì về nghệ sĩ liên quan đến ma túy?',
]

function SourceCard({ source, index }) {
  const metadata = source.metadata || {}
  const title = metadata.source || metadata.path || `Nguồn ${index + 1}`

  return (
    <details className="source-card">
      <summary>
        <span>{title}</span>
        {typeof source.score === 'number' && <small>score {source.score.toFixed(3)}</small>}
      </summary>
      <p>{source.content}</p>
      <div className="metadata">
        {source.source && <span>retrieval: {source.source}</span>}
        {metadata.type && <span>type: {metadata.type}</span>}
        {metadata.path && <span>path: {metadata.path}</span>}
      </div>
    </details>
  )
}

function Message({ item }) {
  const isUser = item.role === 'user'
  return (
    <article className={`message ${isUser ? 'user' : 'assistant'}`}>
      <div className="bubble">
        <div className="role">{isUser ? 'Bạn' : 'RAG Chatbot'}</div>
        <div className="text">{item.content}</div>
        {!isUser && item.retrievalSource && (
          <div className="retrieval-source">
            <span className="retrieval-badge">retrieval: {item.retrievalSource}</span>
            {item.sources?.length > 0 && (
              <span className="source-count-badge">{item.sources.length} sources used</span>
            )}
          </div>
        )}
      </div>
      {!isUser && item.sources?.length > 0 && (
        <div className="sources">
          <h3>Nguồn đã dùng ({item.sources.length} tài liệu)</h3>
          {item.sources.map((source, index) => (
            <SourceCard key={`${source.metadata?.path || source.metadata?.source || index}-${index}`} source={source} index={index} />
          ))}
        </div>
      )}
    </article>
  )
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content:
        'Xin chào! Mình là chatbot RAG về pháp luật ma túy và các bài báo liên quan. Hãy đặt câu hỏi, mình sẽ trả lời kèm nguồn.',
      sources: [],
      retrievalSource: 'none',
    },
  ])
  const [input, setInput] = useState('')
  const [topK, setTopK] = useState(8)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading])

  async function sendMessage(question = input) {
    const message = question.trim()
    if (!message || loading) return

    const chatHistory = messages
      .filter((msg) => msg.role === 'user' || msg.role === 'assistant')
      .map((msg) => ({
        role: msg.role,
        content: msg.content,
      }))

    setError('')
    setInput('')
    setLoading(true)
    setMessages((prev) => [...prev, { role: 'user', content: message }])

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          history: chatHistory,
          top_k: Number(topK),
        }),
      })

      if (!response.ok) {
        const detail = await response.text()
        throw new Error(detail || `HTTP ${response.status}`)
      }

      const data = await response.json()
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer || 'Không có câu trả lời.',
          sources: data.sources || [],
          retrievalSource: data.retrieval_source || 'none',
        },
      ])
    } catch (err) {
      setError(`Không gọi được backend RAG tại ${API_URL}. Chi tiết: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  function handleSubmit(event) {
    event.preventDefault()
    sendMessage()
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <p className="eyebrow">Group Project</p>
        <h1>Drug Law RAG Chatbot</h1>
        <p>
          Hỏi đáp về pháp luật ma túy và tin tức liên quan, sử dụng pipeline retrieval + generation đã xây dựng trong Python.
        </p>
      </section>

      <section className="examples">
        {EXAMPLE_QUESTIONS.map((question) => (
          <button key={question} type="button" onClick={() => sendMessage(question)} disabled={loading}>
            {question}
          </button>
        ))}
      </section>

      <section className="chat-panel">
        <div className="messages">
          {messages.map((message, index) => (
            <Message key={index} item={message} />
          ))}
          {loading && <div className="loading">Đang truy xuất tài liệu và sinh câu trả lời...</div>}
        </div>

        {error && <div className="error">{error}</div>}

        <form className="composer" onSubmit={handleSubmit}>
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Nhập câu hỏi của bạn..."
            rows={3}
          />
          <div className="controls">
            <label>
              top_k
              <input
                type="number"
                min="1"
                max="10"
                value={topK}
                onChange={(event) => setTopK(event.target.value)}
              />
            </label>
            <button type="submit" disabled={!canSend}>
              {loading ? 'Đang gửi...' : 'Gửi câu hỏi'}
            </button>
          </div>
        </form>
      </section>
    </main>
  )
}
