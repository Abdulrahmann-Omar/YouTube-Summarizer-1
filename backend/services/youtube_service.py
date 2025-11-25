"""
YouTube caption extraction service using yt-dlp.
Async implementation for better performance.
"""

import asyncio
import os
import tempfile
import logging
from typing import Dict, Optional, Tuple
import yt_dlp
import webvtt
from pathlib import Path

from models.schemas import VideoInfo

logger = logging.getLogger(__name__)


class YouTubeService:
    """
    Service for extracting captions and metadata from YouTube videos.
    """
    
    def __init__(self):
        """Initialize YouTube service."""
        self.temp_dir = tempfile.gettempdir()
    
    async def extract_video_info(self, url: str) -> VideoInfo:
        """
        Extract video metadata without downloading.
        
        Args:
            url: YouTube video URL
            
        Returns:
            VideoInfo object with metadata
        """
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None,
                lambda: self._extract_info_sync(url, ydl_opts)
            )
            
            return VideoInfo(
                title=info.get('title', 'Unknown'),
                duration=info.get('duration'),
                channel=info.get('uploader'),
                description=info.get('description', '')[:500]  # Limit description length
            )
        except Exception as e:
            logger.error(f"Error extracting video info: {e}")
            raise ValueError(f"Could not extract video information: {str(e)}")
    
    def _extract_info_sync(self, url: str, opts: Dict) -> Dict:
        """Synchronous helper for extracting video info."""
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)
    
    async def get_captions(self, url: str) -> Tuple[str, VideoInfo]:
        """
        Extract captions from YouTube video.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Tuple of (caption_text, video_info)
        """
        # Generate unique temporary filename
        temp_id = os.urandom(8).hex()
        output_template = os.path.join(self.temp_dir, f'yt_captions_{temp_id}.%(ext)s')
        
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-GB'],  # Try multiple English variants
            'subtitlesformat': 'vtt',
            'outtmpl': output_template,
            'nooverwrites': False,
            'quiet': True,
            'no_warnings': True
        }
        
        try:
            # Extract captions in thread pool
            loop = asyncio.get_event_loop()
            info_dict = await loop.run_in_executor(
                None,
                lambda: self._download_captions_sync(url, ydl_opts)
            )
            
            # Extract video info
            video_info = VideoInfo(
                title=info_dict.get('title', 'Unknown'),
                duration=info_dict.get('duration'),
                channel=info_dict.get('uploader'),
                description=info_dict.get('description', '')[:500]
            )
            
            # Read caption file - try multiple possible filenames
            caption_file = None
            possible_files = [
                f'yt_captions_{temp_id}.en.vtt',
                f'yt_captions_{temp_id}.en-US.vtt',
                f'yt_captions_{temp_id}.en-GB.vtt',
                f'yt_captions_{temp_id}.vtt',
                f'yt_captions_{temp_id}.en.srt',
                f'yt_captions_{temp_id}.srt'
            ]
            
            for filename in possible_files:
                test_path = os.path.join(self.temp_dir, filename)
                if os.path.exists(test_path):
                    caption_file = test_path
                    logger.info(f"Found caption file: {filename}")
                    break
            
            if not caption_file:
                raise FileNotFoundError(
                    "Caption file not found. Video may not have English captions available. "
                    "Try a video with captions enabled (look for the CC button on YouTube)."
                )
            
            # Parse captions
            captions = await loop.run_in_executor(
                None,
                lambda: self._parse_captions(caption_file)
            )
            
            # Cleanup
            try:
                os.remove(caption_file)
            except Exception as cleanup_error:
                logger.warning(f"Could not remove temporary file: {cleanup_error}")
            
            return captions, video_info
            
        except Exception as e:
            logger.error(f"Error extracting captions: {e}")
            raise ValueError(f"Could not extract captions: {str(e)}")
    
    def _download_captions_sync(self, url: str, opts: Dict) -> Dict:
        """Synchronous helper for downloading captions."""
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
            return ydl.extract_info(url, download=False)
    
    def _parse_captions(self, caption_file: str) -> str:
        """
        Parse caption file and extract text.
        
        Args:
            caption_file: Path to caption file
            
        Returns:
            Concatenated caption text
        """
        try:
            corpus = []
            
            # Try parsing as WebVTT
            if caption_file.endswith('.vtt'):
                for caption in webvtt.read(caption_file):
                    corpus.append(caption.text)
            else:
                # Fallback: read as plain text
                with open(caption_file, 'r', encoding='utf-8') as f:
                    corpus.append(f.read())
            
            # Join and clean
            text = " ".join(corpus)
            text = text.replace('\n', ' ')
            
            return text
            
        except Exception as e:
            logger.error(f"Error parsing captions: {e}")
            raise ValueError(f"Could not parse caption file: {str(e)}")
    
    async def validate_url(self, url: str) -> bool:
        """
        Validate if URL is a valid YouTube URL.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        youtube_patterns = [
            'youtube.com/watch',
            'youtu.be/',
            'youtube.com/embed/',
            'youtube.com/v/'
        ]
        return any(pattern in url for pattern in youtube_patterns)


# Global instance
_youtube_service: Optional[YouTubeService] = None


def get_youtube_service() -> YouTubeService:
    """
    Get or create global YouTube service instance.
    
    Returns:
        YouTubeService instance
    """
    global _youtube_service
    if _youtube_service is None:
        _youtube_service = YouTubeService()
    return _youtube_service

