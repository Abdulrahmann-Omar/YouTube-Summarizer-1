"""
YouTube caption extraction service using yt-dlp.
Async implementation for better performance.
"""

import asyncio
import os
import tempfile
import logging
import re
from typing import Dict, Optional, Tuple
import yt_dlp
import webvtt
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

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
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video ID or None if not found
        """
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/|v\/|youtu\.be\/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    async def _get_captions_via_transcript_api(self, video_id: str) -> str:
        """
        Primary method: Extract captions using youtube-transcript-api.
        More reliable and bypasses many download restrictions.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Caption text
        """
        loop = asyncio.get_event_loop()
        
        # Try multiple language codes
        language_codes = ['en', 'en-US', 'en-GB', 'a.en']  # a.en = auto-generated
        
        for lang_code in language_codes:
            try:
                logger.info(f"Trying youtube-transcript-api with language: {lang_code}")
                
                # Run in thread pool
                transcript_list = await loop.run_in_executor(
                    None,
                    lambda: YouTubeTranscriptApi.get_transcript(
                        video_id,
                        languages=[lang_code]
                    )
                )
                
                # Concatenate all caption entries
                caption_text = " ".join([entry['text'] for entry in transcript_list])
                
                if caption_text and len(caption_text.strip()) > 0:
                    logger.info(f"Successfully extracted captions via transcript API ({lang_code})")
                    return caption_text
                    
            except (NoTranscriptFound, TranscriptsDisabled) as e:
                logger.debug(f"Language {lang_code} not available: {e}")
                continue
            except Exception as e:
                logger.warning(f"Transcript API failed for {lang_code}: {e}")
                continue
        
        raise ValueError("No transcripts found via transcript API")
    
    async def get_captions(self, url: str) -> Tuple[str, VideoInfo]:
        """
        Extract captions from YouTube video with dual-method approach.
        
        Primary: youtube-transcript-api (more reliable)
        Fallback: yt-dlp (works for some edge cases)
        
        Args:
            url: YouTube video URL
            
        Returns:
            Tuple of (caption_text, video_info)
        """
        # Extract video ID
        video_id = self._extract_video_id(url)
        if not video_id:
            raise ValueError("Could not extract video ID from URL")
        
        loop = asyncio.get_event_loop()
        
        # First, always fetch video info
        try:
            info_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True
            }
            info_dict = await loop.run_in_executor(
                None,
                lambda: self._extract_info_sync(url, info_opts)
            )
            
            video_info = VideoInfo(
                title=info_dict.get('title', 'Unknown'),
                duration=info_dict.get('duration'),
                channel=info_dict.get('uploader'),
                description=info_dict.get('description', '')[:500]
            )
        except Exception as e:
            logger.warning(f"Could not fetch video info: {e}")
            video_info = VideoInfo(title="Unknown", duration=None, channel=None, description="")
        
        # METHOD 1: Try youtube-transcript-api (primary - more reliable)
        try:
            logger.info("Attempting caption extraction via youtube-transcript-api...")
            captions = await self._get_captions_via_transcript_api(video_id)
            logger.info("✅ Successfully extracted captions via youtube-transcript-api")
            return captions, video_info
        except Exception as transcript_api_error:
            logger.warning(f"youtube-transcript-api failed: {transcript_api_error}")
        
        # METHOD 2: Fallback to yt-dlp
        logger.info("Falling back to yt-dlp method...")
        
        # Generate unique temporary filename
        temp_id = os.urandom(8).hex()
        output_template = os.path.join(self.temp_dir, f'yt_captions_{temp_id}.%(ext)s')
        
        # Try different subtitle language/format combinations with retries
        subtitle_configs = [
            # First try: English variants with vtt
            {
                'subtitleslangs': ['en', 'en-US', 'en-GB'],
                'subtitlesformat': 'vtt',
                'writesubtitles': True,
                'writeautomaticsub': True,
            },
            # Second try: any language with vtt (auto-translated)
            {
                'subtitleslangs': ['en'],
                'subtitlesformat': 'vtt',
                'writesubtitles': True,
                'writeautomaticsub': True,
                'skip_unavailable_fragments': True,
            },
            # Third try: srt format
            {
                'subtitleslangs': ['en'],
                'subtitlesformat': 'srt',
                'writesubtitles': True,
                'writeautomaticsub': True,
            },
            # Last try: all available subtitles (no language filter)
            {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'skip_unavailable_fragments': True,
            },
        ]
        
        captions = None
        last_error = None
        
        try:
            # Try each subtitle configuration
            for attempt, config in enumerate(subtitle_configs, 1):
                try:
                    logger.info(f"Attempt {attempt}: trying subtitle download with config {config}")
                    
                    ydl_opts = {
                        'skip_download': True,
                        'outtmpl': output_template,
                        'nooverwrites': False,
                        'quiet': True,
                        'no_warnings': True,
                    }
                    ydl_opts.update(config)
                    
                    # Try to download captions
                    await loop.run_in_executor(
                        None,
                        lambda: self._download_captions_sync(url, ydl_opts)
                    )
                    
                    # Look for caption file
                    caption_file = self._find_caption_file(temp_id)
                    
                    if caption_file:
                        logger.info(f"Successfully found caption file: {caption_file}")
                        captions = await loop.run_in_executor(
                            None,
                            lambda: self._parse_captions(caption_file)
                        )
                        
                        if captions and len(captions.strip()) > 0:
                            # Success! Clean up and return
                            try:
                                os.remove(caption_file)
                            except Exception as cleanup_error:
                                logger.warning(f"Could not remove temporary file: {cleanup_error}")
                            
                            return captions, video_info
                        else:
                            logger.warning(f"Caption file was empty, trying next method")
                            try:
                                os.remove(caption_file)
                            except:
                                pass
                    
                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Attempt {attempt} failed: {e}")
                    continue
            
            # If all attempts failed, raise informative error
            if not captions:
                error_msg = (
                    "Could not extract captions from this video. "
                    "This usually means: (1) the video has no captions, "
                    "(2) captions are disabled, or (3) the video is blocked from automated access. "
                    f"Last error: {last_error}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            return captions, video_info
            
        except Exception as e:
            logger.error(f"Error extracting captions: {e}")
            raise ValueError(f"Could not extract captions: {str(e)}")
    
    def _download_captions_sync(self, url: str, opts: Dict) -> Dict:
        """Synchronous helper for downloading captions."""
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
            return ydl.extract_info(url, download=False)
    
    def _find_caption_file(self, temp_id: str) -> Optional[str]:
        """
        Search for caption file with multiple possible extensions and language variants.
        
        Args:
            temp_id: Temporary file ID prefix
            
        Returns:
            Path to caption file if found, None otherwise
        """
        # Try multiple possible filenames
        possible_files = [
            f'yt_captions_{temp_id}.en.vtt',
            f'yt_captions_{temp_id}.en-US.vtt',
            f'yt_captions_{temp_id}.en-GB.vtt',
            f'yt_captions_{temp_id}.vtt',
            f'yt_captions_{temp_id}.en.srt',
            f'yt_captions_{temp_id}.srt',
            # Also try without language code (auto-downloaded)
            f'yt_captions_{temp_id}.fr.vtt',
            f'yt_captions_{temp_id}.de.vtt',
            f'yt_captions_{temp_id}.es.vtt',
        ]
        
        # Add any other vtt/srt files with this temp_id prefix
        try:
            temp_files = [f for f in os.listdir(self.temp_dir) if temp_id in f]
            possible_files.extend(temp_files)
        except Exception as e:
            logger.warning(f"Could not list temp dir: {e}")
        
        for filename in possible_files:
            test_path = os.path.join(self.temp_dir, filename)
            if os.path.exists(test_path):
                logger.info(f"Found caption file: {filename}")
                return test_path
        
        return None
    
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

