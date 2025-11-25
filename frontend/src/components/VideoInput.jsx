import { useState, useEffect } from 'react'
import { FiYoutube, FiCheck, FiX } from 'react-icons/fi'
import '../styles/VideoInput.css'

function VideoInput({ videoUrl, setVideoUrl, videoInfo, setVideoInfo }) {
  const [isValidating, setIsValidating] = useState(false)
  const [isValid, setIsValid] = useState(null)

  useEffect(() => {
    if (!videoUrl) {
      setIsValid(null)
      setVideoInfo(null)
      return
    }

    const timer = setTimeout(() => {
      validateAndFetchInfo(videoUrl)
    }, 500)

    return () => clearTimeout(timer)
  }, [videoUrl])

  const validateAndFetchInfo = async (url) => {
    const youtubePatterns = [
      /youtube\.com\/watch/,
      /youtu\.be\//,
      /youtube\.com\/embed\//,
      /youtube\.com\/v\//
    ]

    const isValidUrl = youtubePatterns.some(pattern => pattern.test(url))
    
    if (!isValidUrl) {
      setIsValid(false)
      setVideoInfo(null)
      return
    }

    setIsValidating(true)
    
    try {
      const response = await fetch(`/api/video-info?url=${encodeURIComponent(url)}`)
      
      if (response.ok) {
        const info = await response.json()
        setVideoInfo(info)
        setIsValid(true)
      } else {
        setIsValid(false)
        setVideoInfo(null)
      }
    } catch (error) {
      console.error('Error fetching video info:', error)
      setIsValid(false)
      setVideoInfo(null)
    } finally {
      setIsValidating(false)
    }
  }

  return (
    <div className="video-input-section">
      <h2>Enter YouTube URL</h2>
      
      <div className="input-wrapper">
        <FiYoutube className="input-icon" />
        <input
          type="text"
          className={`video-input ${isValid === true ? 'valid' : ''} ${isValid === false ? 'invalid' : ''}`}
          placeholder="https://www.youtube.com/watch?v=..."
          value={videoUrl}
          onChange={(e) => setVideoUrl(e.target.value)}
        />
        {isValidating && <div className="spinner-small"></div>}
        {!isValidating && isValid === true && <FiCheck className="validation-icon valid" />}
        {!isValidating && isValid === false && <FiX className="validation-icon invalid" />}
      </div>

      {videoInfo && (
        <div className="video-preview">
          <h3>{videoInfo.title}</h3>
          {videoInfo.channel && <p className="channel">Channel: {videoInfo.channel}</p>}
          {videoInfo.duration && (
            <p className="duration">
              Duration: {Math.floor(videoInfo.duration / 60)}:{String(videoInfo.duration % 60).padStart(2, '0')}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

export default VideoInput

