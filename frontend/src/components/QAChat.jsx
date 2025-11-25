import { useState, useEffect, useRef } from 'react'
import { FiSend, FiX, FiMessageSquare } from 'react-icons/fi'
import ReactMarkdown from 'react-markdown'
import { toast } from 'react-toastify'
import '../styles/QAChat.css'

function QAChat({ videoContext, videoTitle, onClose }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [ws, setWs] = useState(null)
  const messagesEndRef = useRef(null)
  const clientId = useRef(`client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // Initialize WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${protocol}//${window.location.host}/ws/chat/${clientId.current}`
    
    const websocket = new WebSocket(wsUrl)

    websocket.onopen = () => {
      console.log('WebSocket connected')
      // Initialize chat with context
      websocket.send(JSON.stringify({
        type: 'init',
        context: videoContext
      }))
      
      setMessages([{
        type: 'system',
        content: `Connected! Ask me anything about "${videoTitle}"`
      }])
    }

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      
      if (data.type === 'answer') {
        setMessages(prev => [...prev, {
          type: 'assistant',
          content: data.answer,
          confidence: data.confidence
        }])
        setLoading(false)
      } else if (data.type === 'error') {
        toast.error(data.message)
        setLoading(false)
      } else if (data.type === 'status') {
        console.log('Status:', data.message)
      }
    }

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error)
      toast.error('Connection error. Please try again.')
      setLoading(false)
    }

    websocket.onclose = () => {
      console.log('WebSocket disconnected')
    }

    setWs(websocket)

    return () => {
      if (websocket.readyState === WebSocket.OPEN) {
        websocket.close()
      }
    }
  }, [videoContext, videoTitle])

  const sendMessage = async () => {
    if (!input.trim() || !ws || loading) return

    const userMessage = input.trim()
    setInput('')
    setLoading(true)

    // Add user message to chat
    setMessages(prev => [...prev, {
      type: 'user',
      content: userMessage
    }])

    // Send via WebSocket
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'message',
        message: userMessage,
        history: messages.filter(m => m.type === 'user' || m.type === 'assistant').slice(-5)
      }))
    } else {
      toast.error('Connection lost. Please close and reopen chat.')
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="qa-chat-overlay">
      <div className="qa-chat">
        <div className="chat-header">
          <div className="chat-title">
            <FiMessageSquare />
            <span>Q&A Assistant</span>
          </div>
          <button className="close-btn" onClick={onClose}>
            <FiX />
          </button>
        </div>

        <div className="chat-messages">
          {messages.map((message, idx) => (
            <div key={idx} className={`message message-${message.type}`}>
              {message.type === 'user' && (
                <div className="message-content">
                  <strong>You:</strong>
                  <p>{message.content}</p>
                </div>
              )}
              {message.type === 'assistant' && (
                <div className="message-content">
                  <strong>Assistant:</strong>
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                  {message.confidence && (
                    <span className="confidence">
                      Confidence: {Math.round(message.confidence * 100)}%
                    </span>
                  )}
                </div>
              )}
              {message.type === 'system' && (
                <div className="message-content system">
                  {message.content}
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="message message-assistant">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-container">
          <textarea
            className="chat-input"
            placeholder="Ask a question about the video..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            rows={2}
            disabled={loading}
          />
          <button
            className="send-btn"
            onClick={sendMessage}
            disabled={!input.trim() || loading}
          >
            <FiSend />
          </button>
        </div>
      </div>
    </div>
  )
}

export default QAChat

