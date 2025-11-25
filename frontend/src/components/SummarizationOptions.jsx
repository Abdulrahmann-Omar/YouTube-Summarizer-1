import '../styles/SummarizationOptions.css'

function SummarizationOptions({ method, setMethod, fraction, setFraction }) {
  const methods = [
    { value: 'gemini', label: 'Gemini AI', description: 'Best quality, abstractive summarization' },
    { value: 'tfidf', label: 'TF-IDF', description: 'Fast, keyword-focused extraction' },
    { value: 'frequency', label: 'Frequency-Based', description: 'Position-aware sentence selection' },
    { value: 'gensim', label: 'Gensim TextRank', description: 'Graph-based summarization' },
  ]

  return (
    <div className="summarization-options">
      <div className="option-group">
        <h3>Summarization Method</h3>
        <div className="method-grid">
          {methods.map((m) => (
            <div
              key={m.value}
              className={`method-card ${method === m.value ? 'selected' : ''}`}
              onClick={() => setMethod(m.value)}
            >
              <div className="method-header">
                <input
                  type="radio"
                  name="method"
                  value={m.value}
                  checked={method === m.value}
                  onChange={(e) => setMethod(e.target.value)}
                />
                <span className="method-label">{m.label}</span>
              </div>
              <p className="method-description">{m.description}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="option-group">
        <h3>Summary Length</h3>
        <div className="slider-container">
          <input
            type="range"
            min="0.1"
            max="0.8"
            step="0.1"
            value={fraction}
            onChange={(e) => setFraction(parseFloat(e.target.value))}
            className="slider"
          />
          <div className="slider-labels">
            <span>Brief</span>
            <span className="slider-value">{Math.round(fraction * 100)}%</span>
            <span>Detailed</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default SummarizationOptions

