# Audiblez Fork Memory Bank - Changes from Original

## Overview
Fork by runlevel6 2025 of audiblez by Claudio Santini. This documents all modifications made to create the enhanced version.

## File Structure Changes

### Original Structure
```
audiblez/
  ├── __init__.py
  ├── core.py
  ├── cli.py
  └── ui.py
  └── voices.py
```

### Forked Structure
```
audiblez/ (flat, not package)
  ├── __init__.py (modified)
  ├── core.py (significantly enhanced)
  ├── cli.py (minor changes)
  ├── ui.py (enhanced with settings persistence)
  ├── voices.py (unchanged)
  └── config.json (NEW FILE)
```

## Detailed Changes by File

### 1. core.py (Major Overhaul)

**New Features Added:**
- `config.json` support with settings persistence (`load_settings()`, `save_settings()`)
- `DEFAULT_VOICE = 'af_heart'` constant (line 394)
- `lang_code_from_voice()` helper function for proper Kokoro language code extraction (fix #5)
- Cached spaCy NLP instance (`_nlp` global + `get_nlp()`) - performance improvement (fix #4)
- Comprehensive text cleaning pipeline (`clean_text()`)
- Abbreviation expansion system (`ABBREVIATIONS` list + `_ABBREV_PATTERNS`)
- Roman numeral conversion (`expand_roman_numerals()`, `_ROMAN_*`, `_roman_to_int()`, `_int_to_words()`)

**Bug Fixes:**
- Fix #1: Street abbreviation pattern placed before Saint pattern to handle "St. James" correctly
- Fix #2: Only append period if text doesn't already end with terminal punctuation (., ?, !)
- Fix #3: Changed `max_sentences` check from `>` to `>=` to respect limit properly
- Fix #4: spaCy loaded once and cached globally instead of per-chapter
- Fix #5: `lang_code_from_voice()` extracts correct single-char lang code from voice name
- Fix #8: Preview threads run as daemon threads without blocking UI join
- Fix #9: Settings imported from core for single source of truth
- Fix #10: CONFIG_FILE path resolved relative to `__file__`, not cwd

**Audio Processing Improvements:**
- torch tensor to numpy conversion in `gen_audio_segments()`
- 5ms fade-in/fade-out per segment to eliminate clicks at boundaries
- Real-time ffmpeg progress parsing with accurate ETA in `create_m4b()`
- `_popen_run()` - interruptible subprocess with pipe-draining threads

**M4B Encoding Improvements:**
- Single-pass ffmpeg concat demuxer (removed intermediate .tmp.mp4 file)
- 64k mono bitrate (was 192k, faster encoding)
- Unique temp list filenames using UUID to avoid collisions
- Cleanup of temp files (chapters.txt, cover_temp_image.jpg, list file) in finally block

**Removed Functions:**
- `concat_wavs_with_ffmpeg()` - replaced by direct creation in `create_m4b()`
- `unmark_element()`, `unmark()` - markdown functions removed

**Other Changes:**
- Updated header attribution (claudio.uk → GitHub URL, "Fork by" instead of "Enhanced fork by")
- Improved shell escaping in `create_m4b()`:
  - Added `safe_stem = Path(filename).stem.replace("'", "")` to sanitize filenames
  - Changed string escaping from `"\'"` to `"'\\''"` for proper shell-safe quoting

---

### 2. cli.py (Minor Changes)

**Changes:**
- Default voice changed from `'af_sky'` to `'af_heart'`
- Passes speed to `main()` function (was missing)
- Imports `voices` and `available_voices_str` from `audiblez.voices` (unchanged)

---

### 3. ui.py (Enhanced)

**Major Additions:**
- `stop_event` threading support throughout
- Settings persistence via `load_settings`/`save_settings` from core
- Cancel button (`⛔ Cancel Synthesis`) with proper stop handling
- Progress bar label shows percentage
- ETA label visibility toggle
- `on_cancel()` method for user-initiated stop
- Preview threads marked as daemon, no blocking join on main thread
- Thread pruning for completed preview threads

**Settings Integration:**
- Imports `load_settings`, `save_settings`, `DEFAULT_VOICE` from `audiblez.core`
- Settings load on startup
- Voice/speed/output folder saved on change via `save_current_settings()`
- Settings saved on exit and completion

**Audio Preview Enhancements:**
- Error handling with try/except and user-facing message boxes
- Proper button re-enable in finally block via `wx.CallAfter()`
- Cleanup of temp file after playback

**Minor UI Changes:**
- About dialog mentions Vlad Reshetov fork
- Uses `core.lang_code_from_voice(voice)` for preview

---

### 4. config.json (NEW FILE)

Persistent settings storage for:
- `output_folder`: Last used output directory
- `voice`: Last selected voice
- `speed`: Last used speed setting (0.5-2.0)

---

### 5. voices.py

**No changes** - identical to original

---

### 6. __init__.py

**No changes** - identical to original

---

## Summary Table

| Category | Original | Forked |
|----------|----------|--------|
| Structure | Package (`audiblez/` dir) | Flat layout with imports |
| Settings | None | JSON file with persistence |
| Text Processing | Minimal | Full cleaning + abbreviations + roman numerals |
| Performance | spaCy loaded every time | Cached spaCy NLP |
| Audio | Direct concat, no fades | Fade-in/out, real-time progress |
| Cancellation | None | Full stop_event support |
| M4B Encoding | Two-pass (concat then m4b) | Single-pass via concat demuxer |
| Error Handling | Basic | Enhanced with thread-safe UI updates |