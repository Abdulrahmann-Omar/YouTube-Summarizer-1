/**
 * API service for communicating with the backend.
 * Uses native fetch for HTTP requests and WebSocket for real-time chat.
 */
// Get API base URL from env or use relative path for local dev
const API_BASE_URL = import.meta.env.VITE_API_URL || ''

// Debug: log the API URL being used (remove after debugging)
console.log('[API] Base URL:', API_BASE_URL || '(using relative path)')

// Robust response parser: handles empty bodies and non-JSON responses
async function parseResponse(response) {
  // Read as text first to avoid JSON parse errors on empty responses
  const text = await response.text()

  if (!text) {
    // No content
    return null
  }

  // Try to parse JSON, otherwise return raw text
  try {
    return JSON.parse(text)
  } catch (e) {
    return text
  }
}

/**
 * Fetch captions from YouTube via Vercel serverless function
 * @param {string} url - YouTube video URL
 * @returns {Promise<Object>} Captions response
 */
export async function getCaptions(url) {
  console.log('[API] Fetching captions from Vercel...')
  const response = await fetch('/api/get-captions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ url }),
  })

  if (!response.ok) {
    const error = await parseResponse(response)
    const detail = error && (error.error || error.detail || 'Failed to fetch captions')
    throw new Error(detail)
  }

  const data = await parseResponse(response)
  console.log(`[API] Captions fetched: ${data.word_count} words`)
  return data
}

/**
 * Summarize text directly (bypasses YouTube fetching)
 * @param {string} text - Text to summarize
 * @param {string} method - Summarization method
 * @param {number} fraction - Summary fraction
 * @param {string} videoTitle - Optional video title
 * @returns {Promise<Object>} Summary response
 */
export async function summarizeText(text, method, fraction, videoTitle = 'Video') {
  console.log('[API] Sending text to HF for summarization...')
  const response = await fetch(`${API_BASE_URL}/api/summarize-text`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text,
      method,
      fraction,
      video_title: videoTitle
    }),
  })

  if (!response.ok) {
    const error = await parseResponse(response)
    const detail = error && (error.detail || error.error || 'Failed to summarize text')
    throw new Error(detail)
  }

  return parseResponse(response)
}

/**
 * Summarize a YouTube video using split architecture:
 * 1. Fetch captions from Vercel (has YouTube access)
 * 2. Send text to HF Spaces for summarization (has compute power)
 * 
 * @param {string} url - YouTube video URL
 * @param {string} method - Summarization method
 * @param {number} fraction - Summary fraction (0.1-0.8)
 * @returns {Promise<Object>} Summary response
 */
export async function summarizeVideo(url, method, fraction) {
  // Step 1: Get captions from Vercel serverless function
  const captionsData = await getCaptions(url)

  if (!captionsData.transcript || captionsData.transcript.length < 100) {
    throw new Error('Video captions are too short or unavailable')
  }

  // Step 2: Send text to HF Spaces for summarization
  const summaryData = await summarizeText(
    captionsData.transcript,
    method,
    fraction,
    `Video ${captionsData.video_id}`
  )

  return summaryData
}

/**
 * Get video information without processing
 * @param {string} url - YouTube video URL
 * @returns {Promise<Object>} Video info
 */
export async function getVideoInfo(url) {
  const response = await fetch(`${API_BASE_URL}/api/video-info?url=${encodeURIComponent(url)}`)

  if (!response.ok) {
    const error = await parseResponse(response)
    const detail = error && (error.detail || (error.error && error.error) || (error.message && error.message))
    throw new Error(detail || 'Failed to fetch video info')
  }

  return parseResponse(response)
}

/**
 * Ask a question about video content
 * @param {string} question - User question
 * @param {string} videoContext - Video transcript/summary
 * @param {Array} conversationHistory - Previous Q&A pairs
 * @returns {Promise<Object>} Answer response
 */
export async function askQuestion(question, videoContext, conversationHistory = []) {
  const response = await fetch(`${API_BASE_URL}/api/qa`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      question,
      video_context: videoContext,
      conversation_history: conversationHistory,
    }),
  })

  if (!response.ok) {
    const error = await parseResponse(response)
    const detail = error && (error.detail || (error.error && error.error) || (error.message && error.message))
    throw new Error(detail || 'Failed to get answer')
  }

  return parseResponse(response)
}

/**
 * Check API health
 * @returns {Promise<Object>} Health status
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`)

  if (!response.ok) {
    throw new Error('Health check failed')
  }

  return parseResponse(response)
}

/**
 * Create WebSocket connection for chat
 * @param {string} clientId - Unique client identifier
 * @returns {WebSocket} WebSocket instance
 */
export function createChatWebSocket(clientId) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const wsUrl = `${protocol}//${window.location.host}/ws/chat/${clientId}`

  return new WebSocket(wsUrl)
}

export default {
  getCaptions,
  summarizeText,
  summarizeVideo,
  getVideoInfo,
  askQuestion,
  checkHealth,
  createChatWebSocket,
}

