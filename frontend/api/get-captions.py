"""
Vercel serverless function to fetch YouTube captions.
This bypasses HF Spaces network restrictions by running on Vercel.
"""
from http.server import BaseHTTPRequestHandler
import json
import re
import sys
import traceback


def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    if not url:
        return None
    patterns = [
        r'(?:v=|/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed/|v/|youtu\.be/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_json(400, {"success": False, "error": "Empty request body"})
                return
                
            body = self.rfile.read(content_length)
            
            try:
                data = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError as e:
                self._send_json(400, {"success": False, "error": f"Invalid JSON: {str(e)}"})
                return
            
            url = data.get('url', '')
            if not url:
                self._send_json(400, {"success": False, "error": "Missing 'url' parameter"})
                return
            
            # Extract video ID
            video_id = extract_video_id(url)
            if not video_id:
                self._send_json(400, {"success": False, "error": f"Could not extract video ID from URL: {url}"})
                return
            
            # Import youtube_transcript_api
            try:
                from youtube_transcript_api import YouTubeTranscriptApi
                from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
            except ImportError as e:
                self._send_json(500, {"success": False, "error": f"Failed to import youtube_transcript_api: {str(e)}"})
                return
            
            # Try to get transcript
            transcript_text = None
            last_error = None
            
            # Try multiple language codes
            languages_to_try = [
                ['en'],
                ['en-US'],
                ['en-GB'],
                ['en', 'en-US', 'en-GB'],
            ]
            
            for langs in languages_to_try:
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
                    transcript_text = " ".join([entry['text'] for entry in transcript_list])
                    if transcript_text and len(transcript_text.strip()) > 0:
                        break
                except (NoTranscriptFound, TranscriptsDisabled) as e:
                    last_error = str(e)
                    continue
                except VideoUnavailable as e:
                    self._send_json(400, {"success": False, "error": f"Video unavailable: {str(e)}"})
                    return
                except Exception as e:
                    last_error = str(e)
                    continue
            
            # If no transcript found with specific languages, try without language filter
            if not transcript_text:
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    transcript_text = " ".join([entry['text'] for entry in transcript_list])
                except Exception as e:
                    last_error = str(e)
            
            if not transcript_text:
                error_msg = f"Could not fetch captions for video {video_id}"
                if last_error:
                    error_msg += f": {last_error}"
                self._send_json(400, {"success": False, "error": error_msg})
                return
            
            # Success!
            self._send_json(200, {
                "success": True,
                "video_id": video_id,
                "transcript": transcript_text,
                "word_count": len(transcript_text.split())
            })
            
        except Exception as e:
            error_details = traceback.format_exc()
            self._send_json(500, {
                "success": False, 
                "error": f"Server error: {str(e)}",
                "details": error_details
            })
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Content-Length', '0')
        self.end_headers()
    
    def _send_json(self, status_code: int, data: dict):
        """Helper to send JSON response with CORS headers."""
        response_body = json.dumps(data).encode('utf-8')
        
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)
    
    def log_message(self, format, *args):
        """Suppress default logging."""
        pass
