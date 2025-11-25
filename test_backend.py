#!/usr/bin/env python3
"""
Simple test script to verify the YouTube Summarizer backend is working.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is healthy!")
            print(f"   Status: {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Gemini Available: {data['gemini_available']}")
            print(f"   spaCy Model Loaded: {data['spacy_model_loaded']}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend. Is it running on port 8000?")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_video_info():
    """Test video info endpoint with a sample video."""
    print("\n📹 Testing video info endpoint...")
    
    # Using a popular, stable video (Rick Astley - Never Gonna Give You Up)
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/video-info",
            params={"url": test_url},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Video info retrieved successfully!")
            print(f"   Title: {data.get('title', 'N/A')}")
            print(f"   Channel: {data.get('channel', 'N/A')}")
            if data.get('duration'):
                minutes = data['duration'] // 60
                seconds = data['duration'] % 60
                print(f"   Duration: {minutes}m {seconds}s")
            return True
        else:
            print(f"❌ Video info failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_summarization():
    """Test summarization with TF-IDF method (doesn't require Gemini API)."""
    print("\n📝 Testing summarization endpoint...")
    print("   (This may take 30-60 seconds...)")
    
    # Using a short educational video
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    payload = {
        "url": test_url,
        "method": "tfidf",  # TF-IDF doesn't require Gemini API
        "fraction": 0.3
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/summarize",
            json=payload,
            timeout=120  # 2 minutes timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Summarization successful!")
            print(f"   Video: {data['video_info']['title']}")
            print(f"   Method: {data['method_used']}")
            print(f"   Processing Time: {data['processing_time']:.2f}s")
            print(f"   Word Count: {data['word_count']}")
            print(f"   Sentence Count: {data['sentence_count']}")
            print(f"\n   Summary Preview:")
            preview = data['summary'][:200] + "..." if len(data['summary']) > 200 else data['summary']
            print(f"   {preview}")
            
            if data.get('entities'):
                print(f"\n   Key Entities: {', '.join(data['entities'][:5])}")
            
            if data.get('key_topics'):
                print(f"   Key Topics: {', '.join(data['key_topics'][:5])}")
            
            return True
        else:
            print(f"❌ Summarization failed with status {response.status_code}")
            error_data = response.json()
            print(f"   Error: {error_data.get('detail', 'Unknown error')}")
            return False
    except requests.exceptions.Timeout:
        print("❌ Request timed out. Video processing may take longer.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("YouTube Summarizer Backend Test Suite")
    print("=" * 60)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ Backend is not running. Please start it with:")
        print("   cd backend && source venv/bin/activate && python -m uvicorn main:app --reload")
        return
    
    # Test 2: Video info
    test_video_info()
    
    # Test 3: Summarization (optional - takes longer)
    print("\n" + "=" * 60)
    choice = input("Run summarization test? (takes 30-60 seconds) [y/N]: ").lower()
    if choice == 'y':
        test_summarization()
    else:
        print("⏭️  Skipping summarization test")
    
    print("\n" + "=" * 60)
    print("✨ Testing complete!")
    print("\n📚 Next steps:")
    print("   1. Access API docs: http://localhost:8000/docs")
    print("   2. Install Node.js to run the web frontend")
    print("   3. Add GEMINI_API_KEY to .env for AI-powered summaries")
    print("=" * 60)

if __name__ == "__main__":
    main()

