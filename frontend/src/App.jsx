import { useState, useEffect } from 'react'
import { ToastContainer, toast } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css'

import VideoInput from './components/VideoInput'
import SummarizationOptions from './components/SummarizationOptions'
import SummaryDisplay from './components/SummaryDisplay'
import QAChat from './components/QAChat'
import Header from './components/Header'

import './styles/App.css'

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('darkMode')
    return saved ? JSON.parse(saved) : true
  })
  
  const [videoUrl, setVideoUrl] = useState('')
  const [videoInfo, setVideoInfo] = useState(null)
  const [summarizationMethod, setSummarizationMethod] = useState('gemini')
  const [fraction, setFraction] = useState(0.3)
  const [summary, setSummary] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showChat, setShowChat] = useState(false)
  const [videoContext, setVideoContext] = useState('')

  // Apply dark mode
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', darkMode ? 'dark' : 'light')
    localStorage.setItem('darkMode', JSON.stringify(darkMode))
  }, [darkMode])

  const toggleDarkMode = () => {
    setDarkMode(!darkMode)
  }

  const handleSummarize = async (url, method, frac) => {
    setLoading(true)
    setError(null)
    setSummary(null)
    setShowChat(false)
    
    try {
      const response = await fetch('/api/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          url: url,
          method: method,
          fraction: frac,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to summarize video')
      }

      const data = await response.json()
      setSummary(data)
      setVideoInfo(data.video_info)
      setVideoContext(data.summary) // Use summary as context for Q&A
      toast.success('Summary generated successfully!')
      
    } catch (err) {
      setError(err.message)
      toast.error(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setVideoUrl('')
    setVideoInfo(null)
    setSummary(null)
    setError(null)
    setShowChat(false)
    setVideoContext('')
  }

  return (
    <div className="app">
      <Header darkMode={darkMode} toggleDarkMode={toggleDarkMode} />
      
      <main className="container">
        <div className="content-wrapper">
          {!summary ? (
            <div className="input-section">
              <VideoInput
                videoUrl={videoUrl}
                setVideoUrl={setVideoUrl}
                videoInfo={videoInfo}
                setVideoInfo={setVideoInfo}
              />
              
              <SummarizationOptions
                method={summarizationMethod}
                setMethod={setSummarizationMethod}
                fraction={fraction}
                setFraction={setFraction}
              />
              
              <button
                className="btn btn-primary btn-large"
                onClick={() => handleSummarize(videoUrl, summarizationMethod, fraction)}
                disabled={!videoUrl || loading}
              >
                {loading ? 'Processing...' : 'Generate Summary'}
              </button>
              
              {error && (
                <div className="error-message">
                  <p>{error}</p>
                </div>
              )}
            </div>
          ) : (
            <div className="results-section">
              <SummaryDisplay
                summary={summary}
                onReset={handleReset}
                onStartChat={() => setShowChat(true)}
              />
              
              {showChat && (
                <QAChat
                  videoContext={videoContext}
                  videoTitle={videoInfo?.title}
                  onClose={() => setShowChat(false)}
                />
              )}
            </div>
          )}
        </div>
      </main>
      
      <ToastContainer
        position="bottom-right"
        autoClose={3000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme={darkMode ? 'dark' : 'light'}
      />
    </div>
  )
}

export default App

