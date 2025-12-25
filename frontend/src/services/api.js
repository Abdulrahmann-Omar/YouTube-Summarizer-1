/**
 * API service for communicating with the backend.
 * Uses native fetch for HTTP requests and WebSocket for real-time chat.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api'

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
 * Summarize a YouTube video
 * @param {string} url - YouTube video URL
 * @param {string} method - Summarization method
 * @param {number} fraction - Summary fraction (0.1-0.8)
 * @returns {Promise<Object>} Summary response
 */
export async function summarizeVideo(url, method, fraction) {
  const response = await fetch(`${API_BASE_URL}/api/summarize`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ url, method, fraction }),
  })

  if (!response.ok) {
    const error = await parseResponse(response)
    // Try to extract common error shapes
    const detail = error && (error.detail || (error.error && error.error) || (error.message && error.message))
    throw new Error(detail || 'Failed to summarize video')
  }

  return parseResponse(response)
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
  summarizeVideo,
  getVideoInfo,
  askQuestion,
  checkHealth,
  createChatWebSocket,
}

