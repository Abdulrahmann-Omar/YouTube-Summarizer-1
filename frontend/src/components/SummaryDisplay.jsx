import { FiDownload, FiRefreshCw, FiMessageCircle, FiClock, FiFileText } from 'react-icons/fi'
import ReactMarkdown from 'react-markdown'
import { toast } from 'react-toastify'
import '../styles/SummaryDisplay.css'

function SummaryDisplay({ summary, onReset, onStartChat }) {
  const downloadSummary = (format) => {
    const content = format === 'txt' ? summary.summary : JSON.stringify(summary, null, 2)
    const blob = new Blob([content], { type: format === 'txt' ? 'text/plain' : 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${summary.video_info.title.substring(0, 50)}_summary.${format}`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
    toast.success(`Summary downloaded as ${format.toUpperCase()}`)
  }

  const copyToClipboard = () => {
    navigator.clipboard.writeText(summary.summary)
    toast.success('Summary copied to clipboard!')
  }

  return (
    <div className="summary-display">
      <div className="summary-header">
        <div>
          <h2>{summary.video_info.title}</h2>
          <div className="summary-meta">
            <span className="meta-item">
              <FiClock size={14} />
              {summary.processing_time.toFixed(2)}s
            </span>
            <span className="meta-item">
              <FiFileText size={14} />
              {summary.word_count} words, {summary.sentence_count} sentences
            </span>
            <span className="meta-badge">{summary.method_used}</span>
          </div>
        </div>
        
        <div className="header-actions">
          <button className="btn btn-secondary" onClick={onReset}>
            <FiRefreshCw /> New Summary
          </button>
        </div>
      </div>

      <div className="summary-content">
        <div className="summary-text">
          <ReactMarkdown>{summary.summary}</ReactMarkdown>
        </div>

        {summary.entities && summary.entities.length > 0 && (
          <div className="summary-section">
            <h3>Key Entities</h3>
            <div className="tags">
              {summary.entities.map((entity, idx) => (
                <span key={idx} className="tag">{entity}</span>
              ))}
            </div>
          </div>
        )}

        {summary.key_topics && summary.key_topics.length > 0 && (
          <div className="summary-section">
            <h3>Key Topics</h3>
            <div className="tags">
              {summary.key_topics.map((topic, idx) => (
                <span key={idx} className="tag tag-topic">{topic}</span>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="summary-actions">
        <button className="btn btn-primary" onClick={onStartChat}>
          <FiMessageCircle /> Ask Questions
        </button>
        <button className="btn btn-secondary" onClick={copyToClipboard}>
          Copy to Clipboard
        </button>
        <button className="btn btn-secondary" onClick={() => downloadSummary('txt')}>
          <FiDownload /> Download TXT
        </button>
        <button className="btn btn-secondary" onClick={() => downloadSummary('json')}>
          <FiDownload /> Download JSON
        </button>
      </div>
    </div>
  )
}

export default SummaryDisplay

