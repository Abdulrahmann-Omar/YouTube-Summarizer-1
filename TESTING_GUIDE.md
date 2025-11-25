# 🧪 Testing Guide for YouTube Summarizer

## ⚠️ Common Issue: "Did not get any data blocks"

This error means the video **doesn't have English captions** (subtitles) available.

### ✅ How to Find Videos with Captions:

1. **Look for the "CC" button** on YouTube videos
2. Click on a video and check if there's a closed caption icon
3. Use educational channels that always have captions

## 🎬 **Videos That WILL Work** (Have Captions):

### Educational Videos with Captions:

1. **TED Talks** (always have captions):
   ```
   https://www.youtube.com/watch?v=8S0FDjFBj8o
   ```
   (Simon Sinek - Start With Why)

2. **Khan Academy** (educational, has captions):
   ```
   https://www.youtube.com/watch?v=uhxtUt_-GyM
   ```

3. **Crash Course** (popular educational channel):
   ```
   https://www.youtube.com/watch?v=YIQguao8Gfw
   ```

4. **MIT OpenCourseWare**:
   ```
   https://www.youtube.com/watch?v=0fKBhvDjuy0
   ```

5. **Google Developers** (tech talks):
   ```
   https://www.youtube.com/watch?v=cKxRvEZd3Mw
   ```

## 🧪 **Test with This Video:**

Try this one - it definitely has English captions:

**URL**: `https://www.youtube.com/watch?v=8S0FDjFBj8o`

**Settings**:
- **Method**: `tfidf`
- **Fraction**: `0.3`

## 📝 **How to Test in API Docs:**

1. Go to: http://localhost:8000/docs
2. Click on **POST /api/summarize**
3. Click **"Try it out"**
4. Enter the request:

```json
{
  "url": "https://www.youtube.com/watch?v=8S0FDjFBj8o",
  "method": "tfidf",
  "fraction": 0.3
}
```

5. Click **"Execute"**
6. Wait 20-30 seconds for processing

## ⚡ **Quick Test via Command Line:**

```bash
curl -X POST "http://localhost:8000/api/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=8S0FDjFBj8o",
    "method": "tfidf",
    "fraction": 0.3
  }'
```

## 🔍 **How to Check if a Video Has Captions:**

### Method 1: Check on YouTube
1. Play the video
2. Look for the **"CC"** button in the video player
3. If present, captions are available

### Method 2: Test Video Info Endpoint
```bash
curl "http://localhost:8000/api/video-info?url=YOUR_VIDEO_URL"
```

This will tell you if captions are available before trying to summarize.

## 🎯 **Best Video Types for Testing:**

✅ **Will Work:**
- TED Talks
- Khan Academy
- Crash Course
- MIT OpenCourseWare
- Google Developers talks
- Microsoft Developer videos
- News channels (CNN, BBC, etc.)
- Most popular educational content

❌ **Won't Work:**
- Music videos (usually no captions)
- Vlogs without captions
- Old videos without subtitles
- Non-English videos (unless they have English captions)

## 💡 **Tips:**

1. **Longer processing time = longer video**
   - Short videos (5-10 min): 10-20 seconds
   - Medium videos (20-30 min): 30-60 seconds
   - Long videos (60+ min): 1-2 minutes

2. **Best Methods for Testing:**
   - **tfidf**: Fastest, good quality
   - **frequency**: Fast, balanced
   - **gemini**: Best quality (needs API key)

3. **Optimal Fraction:**
   - `0.2` - Very brief summary
   - `0.3` - Balanced (recommended)
   - `0.5` - Detailed summary

## 🚀 **Quick Success Test:**

Copy this exact request in the API docs:

```json
{
  "url": "https://www.youtube.com/watch?v=8S0FDjFBj8o",
  "method": "tfidf",
  "fraction": 0.3
}
```

This will definitely work! 🎉

## 📊 **Expected Response:**

You should get something like:

```json
{
  "success": true,
  "video_info": {
    "title": "How great leaders inspire action | Simon Sinek",
    "duration": 1086,
    "channel": "TED"
  },
  "summary": "...",
  "method_used": "tfidf",
  "processing_time": 15.2,
  "word_count": 245,
  "sentence_count": 12,
  "entities": ["Simon Sinek", "Apple", ...],
  "key_topics": ["leadership", "inspire", ...]
}
```

## ❓ **Still Getting Errors?**

Check:
1. Is the backend running? (ps aux | grep uvicorn)
2. Is the URL correct and accessible?
3. Does the video have English captions?
4. Try a shorter video first (< 10 minutes)

---

**Ready to test?** Go to http://localhost:8000/docs and try the TED Talk video! 🚀

