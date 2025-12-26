from http.server import BaseHTTPRequestHandler
import json
import re


def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
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
            body = self.rfile.read(content_length)
            data = json.loads(body) if body else {}
            
            url = data.get('url', '')
            
            # Extract video ID
            video_id = extract_video_id(url)
            if not video_id:
                self.send_error_response(400, "Invalid YouTube URL")
                return
            
            # Import here to avoid cold start issues
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
            
            # Try multiple languages
            languages = ['en', 'en-US', 'en-GB', 'a.en']
            transcript_text = None
            
            for lang in languages:
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(
                        video_id, 
                        languages=[lang]
                    )
                    transcript_text = " ".join([t['text'] for t in transcript_list])
                    break
                except (NoTranscriptFound, TranscriptsDisabled):
                    continue
                except Exception:
                    continue
            
            if not transcript_text:
                # Try without language filter
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                    transcript_text = " ".join([t['text'] for t in transcript_list])
                except Exception as e:
                    self.send_error_response(400, f"Could not fetch captions: {str(e)}")
                    return
            
            # Success response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            
            response = {
                "success": True,
                "video_id": video_id,
                "transcript": transcript_text,
                "word_count": len(transcript_text.split())
            }
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_error_response(500, f"Server error: {str(e)}")
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def send_error_response(self, code: int, message: str):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = {"success": False, "error": message}
        self.wfile.write(json.dumps(response).encode())
