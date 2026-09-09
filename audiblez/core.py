#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M model for high-quality text-to-speech synthesis.
# Originally by Claudio Santini 2025 - https://claudio.uk
# Fork by Vlad Reshetov 2025
import os
import traceback
import uuid
from glob import glob
import json

import torch.cuda
import spacy
import ebooklib
import soundfile
import numpy as np
import time
import shutil
import subprocess
import platform
import re
import threading
from types import SimpleNamespace
from tabulate import tabulate
from pathlib import Path
from string import Formatter
from bs4 import BeautifulSoup
from kokoro import KPipeline
from ebooklib import epub
from pick import pick
from google import genai

sample_rate = 24000

_SPOKEN_CHARS_PER_SEC = 12.5  # ~150 wpm * 5 chars / 60 s
_AAC_ENCODE_RT_FACTOR = 50    # ffmpeg aac runs ~50x realtime

# Fix #10: use path relative to this file so it resolves consistently
# regardless of where the process is launched from.
CONFIG_FILE = Path(__file__).parent / 'config.json'


# ---------------------------------------------------------------------------
# Settings  (fix #9: single source of truth — UI imports from here)
# ---------------------------------------------------------------------------

def load_settings():
    """Loads settings from the JSON configuration file."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            settings = json.load(f)
            if 'output_folder' in settings:
                settings['output_folder'] = str(Path(settings['output_folder']))
            settings.setdefault('gemini_api_key', '')
            settings.setdefault('gemini_model', 'gemini-3.1-flash-lite')
            settings.setdefault('gemini_enabled', False)
            settings.setdefault('last_open_dir', str(Path.home()))
            return settings
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(output_folder, voice, speed=1.0, gemini_api_key='', gemini_model='gemini-3.1-flash-lite', gemini_enabled=False, last_open_dir=''):
    """
    Saves settings to the JSON configuration file.
    Accepts an optional speed parameter so both core and UI write a
    consistent schema to the same file.
    """
    settings = {
        'output_folder': str(output_folder),
        'voice': voice,
        'speed': float(speed),
        'gemini_api_key': gemini_api_key,
        'gemini_model': gemini_model,
        'gemini_enabled': gemini_enabled,
        'last_open_dir': last_open_dir,
    }
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        print(f"Settings saved to {CONFIG_FILE}.")
    except IOError as e:
        print(f"Error saving settings to {CONFIG_FILE}: {e}")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Fix #1: Street pattern MUST come before the generic Saint pattern so that
# "St. James" → "Street James" does not accidentally fire first.
ABBREVIATIONS = [
    # Titles
    (r'\bMrs\.', 'Missus'),
    (r'\bMr\.', 'Mister'),
    (r'\bMs\.', 'Miz'),
    (r'\bMiss\.', 'Miss'),
    (r'\bDr\.', 'Doctor'),
    (r'\bProf\.', 'Professor'),
    (r'\bRev\.', 'Reverend'),
    (r'\bHon\.', 'Honorable'),
    # Street before Saint so the lookahead fires first
    (r'\bSt\.(?=\s+[A-Z])', 'Street'),
    (r'\bSt\.', 'Saint'),
    # Military / professional ranks
    (r'\bGen\.', 'General'),
    (r'\bCol\.', 'Colonel'),
    (r'\bMaj\.', 'Major'),
    (r'\bCapt\.', 'Captain'),
    (r'\bLt\.', 'Lieutenant'),
    (r'\bSgt\.', 'Sergeant'),
    (r'\bCpl\.', 'Corporal'),
    (r'\bPvt\.', 'Private'),
    (r'\bAdm\.', 'Admiral'),
    # Name suffixes
    (r'\bJr\.', 'Junior'),
    (r'\bSr\.', 'Senior'),
    (r'\bEsq\.', 'Esquire'),
    # Common Latin / general abbreviations
    (r'\betc\.', 'et cetera'),
    (r'\bvs\.', 'versus'),
    (r'\bVs\.', 'Versus'),
    (r'\bapprox\.', 'approximately'),
    (r'\bcf\.', 'compare'),
    (r'\be\.g\.', 'for example'),
    (r'\bi\.e\.', 'that is'),
    (r'\bviz\.', 'namely'),
    (r'\bib\.', 'in the same place'),
    (r'\bibid\.', 'in the same place'),
    (r'\bop\. cit\.', 'in the work cited'),
    (r'\bno\.?\s*(?=\d)', 'number '),
    (r'\bNo\.?\s*(?=\d)', 'Number '),
    (r'\bvol\.', 'volume'),
    (r'\bVol\.', 'Volume'),
    (r'\bch\.', 'chapter'),
    (r'\bCh\.', 'Chapter'),
    (r'\bfig\.', 'figure'),
    (r'\bFig\.', 'Figure'),
    (r'\bp\.', 'page'),
    (r'\bpp\.', 'pages'),
    # Addresses
    (r'\bAve\.', 'Avenue'),
    (r'\bBlvd\.', 'Boulevard'),
    (r'\bRd\.', 'Road'),
    (r'\bDept\.', 'Department'),
    (r'\bGovt\.', 'Government'),
]

# Compile once for efficiency
_ABBREV_PATTERNS = [(re.compile(pat), repl) for pat, repl in ABBREVIATIONS]


# ---------------------------------------------------------------------------
# Roman numeral conversion
# ---------------------------------------------------------------------------

# Matches a valid Roman numeral token (1–3999).  The alternation structure
# ensures only well-formed sequences match — e.g. "IIII" won't match because
# there is no four-I combination in standard notation.
_ROMAN_RE_SRC = (
    r'M{0,4}'           # thousands: 0–4000
    r'(?:CM|CD|D?C{0,3})'   # hundreds: 900, 400, 0–300, 500–800
    r'(?:XC|XL|L?X{0,3})'   # tens: 90, 40, 0–30, 50–80
    r'(?:IX|IV|V?I{0,3})'   # ones: 9, 4, 0–3, 5–8
)

# Case 1 — standalone heading: a line that is *only* a Roman numeral,
# optionally followed by a period, colon, or dash (and nothing else).
# Examples:  "III."   "XIV"   "IV:"   "  ii.  "
_ROMAN_HEADING_RE = re.compile(
    r'^(' + _ROMAN_RE_SRC + r')([.:\-]?)$',
    re.IGNORECASE | re.MULTILINE,
)

# Case 2 — inline after a structural keyword.
# Examples:  "Chapter III"   "Part IV:"   "Book II,"   "Act V, Scene i"
_ROMAN_KEYWORD_RE = re.compile(
    r'\b(Chapter|Part|Section|Book|Volume|Vol|Act|Scene|Article|Appendix|Canto)'
    r'(\s+)(' + _ROMAN_RE_SRC + r')\b',
    re.IGNORECASE,
)


def _roman_to_int(s: str) -> int:
    """Convert a Roman numeral string to an integer.  Returns 0 for empty/invalid."""
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50,
              'C': 100, 'D': 500, 'M': 1000}
    s = s.upper().strip()
    if not s:
        return 0
    total, prev = 0, 0
    for ch in reversed(s):
        v = values.get(ch, 0)
        if v == 0:
            return 0   # invalid character — bail out
        total += v if v >= prev else -v
        prev = v
    return total


def _int_to_words(n: int) -> str:
    """Convert a positive integer (1–3999) to English words."""
    if n <= 0 or n > 3999:
        return str(n)
    ones = [
        '', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
        'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
        'sixteen', 'seventeen', 'eighteen', 'nineteen',
    ]
    tens_words = [
        '', '', 'twenty', 'thirty', 'forty', 'fifty',
        'sixty', 'seventy', 'eighty', 'ninety',
    ]
    parts = []
    if n >= 1000:
        parts.append(ones[n // 1000] + ' thousand')
        n %= 1000
    if n >= 100:
        parts.append(ones[n // 100] + ' hundred')
        n %= 100
    if n >= 20:
        t = tens_words[n // 10]
        o = ones[n % 10]
        parts.append(t + ('-' + o if o else ''))
    elif n > 0:
        parts.append(ones[n])
    return ' '.join(parts)


def _roman_match_to_words(roman_str: str) -> str:
    """
    Convert a Roman numeral string to its English word equivalent (always
    in lowercase base form). Returns the original string unchanged if it
    doesn't parse as a valid Roman numeral (guards against false positives
    like a lone 'C' or 'D').

    Casing is intentionally NOT decided here — callers apply whatever
    capitalisation fits their context (heading vs. inline), so there is a
    single, unambiguous place that decides the final case.
    """
    n = _roman_to_int(roman_str)
    if n == 0:
        return roman_str
    return _int_to_words(n)


def expand_roman_numerals(text: str) -> str:
    """
    Replace Roman numerals in two contexts:

    1. Standalone headings — a line whose entire content is a Roman numeral
       (optionally followed by . : or -).
       e.g.  "III."  →  "Three."

    2. After structural keywords — Chapter, Part, Section, Book, Volume,
       Act, Scene, Article, Appendix, Canto.
       e.g.  "Chapter XIV"  →  "Chapter Fourteen"
             "Act V, Scene i"  →  "Act Five, Scene i"  (second pass)
    """
    # Pass 1 — standalone headings. Headings are always rendered with a
    # leading capital regardless of source casing.
    def _replace_heading(m: re.Match) -> str:
        numeral, punctuation = m.group(1), m.group(2)
        words = _roman_match_to_words(numeral)
        if words == numeral:          # failed to parse — leave untouched
            return m.group(0)
        return words.capitalize() + punctuation

    text = _ROMAN_HEADING_RE.sub(_replace_heading, text)

    # Pass 2 — inline after keyword. Mirror the capitalisation style of the
    # source numeral so the result blends naturally with the surrounding text.
    def _replace_inline(m: re.Match) -> str:
        keyword, space, numeral = m.group(1), m.group(2), m.group(3)
        words = _roman_match_to_words(numeral)
        if words == numeral:
            return m.group(0)
        if numeral.isupper():
            words = words.title()
        elif numeral[:1].isupper():
            words = words.capitalize()
        return keyword + space + words

    text = _ROMAN_KEYWORD_RE.sub(_replace_inline, text)

    return text


def clean_text(text: str) -> str:
    """
    Clean raw text extracted from an EPUB chapter before TTS synthesis.

    Steps applied in order:
    1. Remove soft hyphens (U+00AD).
    2. Expand common abbreviations ending in '.' to full words.
    3. Expand Roman numerals (standalone headings and after structural keywords).
    4. Normalise typographic / Unicode punctuation to plain ASCII equivalents.
    5. Collapse runs of whitespace (spaces, tabs) to a single space.
    6. Trim leading/trailing whitespace on each line.
    7. Collapse runs of 3+ newlines to a single blank line.
    8. Strip repeated punctuation (e.g. "!!!" → "!"), preserving ellipsis.
    """
    # 1. Remove soft hyphens
    text = text.replace('\u00ad', '')

    # 2. Expand abbreviations
    for pattern, replacement in _ABBREV_PATTERNS:
        text = pattern.sub(replacement, text)

    # 3. Expand Roman numerals
    text = expand_roman_numerals(text)

    # 4. Normalise Unicode punctuation
    unicode_replacements = {
        '\u2018': "'",    # left single quotation mark
        '\u2019': "'",    # right single quotation mark
        '\u201c': '"',    # left double quotation mark
        '\u201d': '"',    # right double quotation mark
        '\u2013': '-',    # en dash
        '\u2014': ' - ',  # em dash (spaces so TTS pauses naturally)
        '\u2026': '...',  # horizontal ellipsis
        '\u00b7': '.',    # middle dot
        '\u2022': '',     # bullet — drop it
        '\xa0': ' ',      # non-breaking space
    }
    for orig, repl in unicode_replacements.items():
        text = text.replace(orig, repl)

    # 5. Collapse runs of spaces/tabs within a line
    text = re.sub(r'[ \t]+', ' ', text)

    # 6. Trim each line
    text = '\n'.join(line.strip() for line in text.splitlines())

    # 7. Collapse 3+ consecutive newlines to two
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 8. Collapse repeated punctuation; preserve ellipsis and --
    text = re.sub(r'([!?])\1+', r'\1', text)
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r'\.{4,}', '...', text)

    return text


# ---------------------------------------------------------------------------
# spaCy — load once, reuse everywhere  (fix #4)
# ---------------------------------------------------------------------------

_nlp = None


def get_nlp():
    """Return a cached spaCy nlp instance, loading it on first call."""
    global _nlp
    if _nlp is None:
        load_spacy()          # downloads model if absent
        _nlp = spacy.load('xx_ent_wiki_sm')
        _nlp.add_pipe('sentencizer')
    return _nlp


def load_spacy():
    if not spacy.util.is_package("xx_ent_wiki_sm"):
        print("Downloading Spacy model xx_ent_wiki_sm...")
        spacy.cli.download("xx_ent_wiki_sm")


def set_espeak_library():
    """Find and register the espeak-ng library path."""
    try:
        if os.environ.get('ESPEAK_LIBRARY'):
            library = os.environ['ESPEAK_LIBRARY']
        elif platform.system() == 'Darwin':
            from subprocess import check_output
            try:
                cellar = Path(check_output(["brew", "--cellar"], text=True).strip())
                pattern = cellar / "espeak-ng" / "*" / "lib" / "*.dylib"
                if not (library := next(iter(glob(str(pattern))), None)):
                    raise RuntimeError("No espeak-ng library found; please set the path manually")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise RuntimeError("Cannot locate Homebrew Cellar. Is 'brew' installed and in PATH?") from e
        elif platform.system() == 'Linux':
            library = glob('/usr/lib/*/libespeak-ng*')[0]
        elif platform.system() == 'Windows':
            library = 'C:\\Program Files*\\eSpeak NG\\libespeak-ng.dll'
        else:
            print('Unsupported OS, please set the espeak library path manually')
            return
        print('Using espeak library:', library)
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        EspeakWrapper.set_library(library)
    except Exception:
        traceback.print_exc()
        print("Error finding espeak-ng library:")
        print("Probably you haven't installed espeak-ng.")
        print("On Mac: brew install espeak-ng")
        print("On Linux: sudo apt install espeak-ng")


# ---------------------------------------------------------------------------
# Lang-code helper  (fix #5)
# ---------------------------------------------------------------------------

# Kokoro lang codes are a single character prefix of the voice name,
# e.g. "af_heart" → "a", "bf_emma" → "b".  The old settings default of
# "en_US" would have produced "e" which is invalid.
DEFAULT_VOICE = 'af_heart'


def lang_code_from_voice(voice: str) -> str:
    """
    Extract the single-character Kokoro language code from a voice name.
    e.g. "af_heart" → "a",  "bf_emma" → "b".
    Falls back to 'a' (American English) if the voice string is unexpected.
    """
    if voice and len(voice) >= 1 and voice[1:2] in ('f', 'm', '_'):
        return voice[0]
    # Unrecognised format — default to American English
    print(f"Warning: could not determine lang code from voice '{voice}', defaulting to 'a'.")
    return 'a'


# ---------------------------------------------------------------------------
# Gemini API usage guidance (AFC)
# ---------------------------------------------------------------------------
# Automatic Function Calling (AFC) is how the google-genai SDK executes the
# function calls the model returns. Whenever Gemini is invoked with tools /
# function declarations, drive AFC through a Chat session:
#
#   chat = client.chats.create(model=model, config=config_with_tools)
#   chat.send_message(contents)          # non-streaming  → recommended
#   chat.send_message_stream(contents)   # streaming      → recommended
#
# Direct use of AFC through Models.generate_content / Models.generate_content_stream
# is NOT recommended: those raw entry points bypass chat-managed context, and the
# SDK logs the warning "Direct use of automatic function calling (AFC) in
# Models.generate_content_stream is not recommended. Instead, we recommend to use
# AFC in Chat.send_message_stream." Use Chat.send_message (or
# Chat.send_message_stream) instead. The plain no-tools calls below are the only
# place the code may still hit client.models.generate_content directly.


# ---------------------------------------------------------------------------
# Gemini retry helper
# ---------------------------------------------------------------------------

# Backoff schedule requested: retry after 1 minute, then 2, then 3, then 4
# (each pause is the previous pause + 1 minute). That's 4 retries (5 total
# attempts) before giving up.
_GEMINI_RETRY_DELAYS_SECS = [60, 120, 180, 240]

# Errors in this set are permanent (bad model name, bad key, no permission)
# and will never succeed no matter how many times we retry, so we bail out
# immediately instead of making the caller wait up to 10 minutes for nothing.
_NON_RETRYABLE_ERROR_MARKERS = ('NOT_FOUND', 'PERMISSION_DENIED', 'INVALID_ARGUMENT', 'UNAUTHENTICATED')


def _is_retryable_gemini_error(exc) -> bool:
    msg = str(exc)
    return not any(marker in msg for marker in _NON_RETRYABLE_ERROR_MARKERS)


def _sleep_with_stop_event(seconds, stop_event=None):
    """Sleep in small increments so a stop_event can interrupt promptly
    instead of blocking for the full backoff duration."""
    elapsed = 0.0
    step = 1.0
    while elapsed < seconds:
        if stop_event and stop_event.is_set():
            return
        time.sleep(min(step, seconds - elapsed))
        elapsed += step


def _call_gemini_with_retry(func, *args, stop_event=None, post_event=None, **kwargs):
    """
    Call func(*args, **kwargs), retrying on failure with an increasing
    backoff of 1, 2, 3, then 4 minutes (4 retries / 5 attempts total).
    Permanent-looking errors (invalid model, bad key, permission denied)
    are not retried. If stop_event fires during a backoff wait, the last
    exception is raised immediately.

    Returns func's return value on success, or raises the last exception
    once attempts are exhausted. If post_event is provided, it is called
    with event name 'CORE_AI_RETRY_EXHAUSTED' and the error message when
    all retries are exhausted due to transient errors.
    """
    last_exc = None
    for attempt in range(len(_GEMINI_RETRY_DELAYS_SECS) + 1):
        if stop_event and stop_event.is_set():
            if last_exc:
                raise last_exc
            raise RuntimeError('Stopped by user.')
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exc = e
            more_attempts_left = attempt < len(_GEMINI_RETRY_DELAYS_SECS)
            if more_attempts_left and _is_retryable_gemini_error(e):
                delay = _GEMINI_RETRY_DELAYS_SECS[attempt]
                print(f'\033[93mGemini call failed ({e}). Retrying in {delay // 60} '
                      f'minute(s)... (attempt {attempt + 2}/{len(_GEMINI_RETRY_DELAYS_SECS) + 1})\033[0m')
                _sleep_with_stop_event(delay, stop_event)
            else:
                break
    if post_event and last_exc and _is_retryable_gemini_error(last_exc):
        post_event('CORE_AI_RETRY_EXHAUSTED', message=str(last_exc))
    raise last_exc


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


# _m4b_progress_reporter removed: progress is now driven by real-time parsing
# of ffmpeg's stderr inside create_m4b, using the actual speed=Nx value.


def main(file_path, voice=None, pick_manually=False, speed=1, output_folder='.',
         max_chapters=None, max_sentences=None, selected_chapters=None, post_event=None, stop_event=None):
    if post_event:
        post_event('CORE_STARTED')

    # Ensure spaCy is ready before we do anything else
    get_nlp()

    settings = load_settings()

    output_folder = Path(output_folder) if output_folder != '.' else Path(settings.get('output_folder', '.'))
    voice = voice if voice else settings.get('voice', DEFAULT_VOICE)

    ai_enabled = bool(settings.get('gemini_enabled', False))
    ai_api_key = settings.get('gemini_api_key', '') or ''
    ai_model = settings.get('gemini_model', 'gemini-3.1-flash-lite') or 'gemini-3.1-flash-lite'
    if ai_enabled:
        if not _sanitize_api_key(ai_api_key).startswith('AIza'):
            print('\033[93m' + 'AI Phonetic Check is enabled but the API key is missing or invalid. '
                  'Falling back to original text for the whole book.' + '\033[0m')
            ai_enabled = False
        else:
            print(f'AI Phonetic Check enabled (model={ai_model}). Chapters will be '
                  f'rewritten in chunks of ~{_AI_REWRITE_MAX_TOKENS:,} tokens before TTS.')

    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output folder set to: {output_folder}")
    print(f"Voice selected: {voice}")

    filename = Path(file_path).name
    extension = '.epub'
    try:
        book = epub.read_epub(file_path)
    except Exception as e:
        print(f'\033[91mFailed to read EPUB file "{file_path}": {e}\033[0m')
        if post_event:
            post_event('CORE_ERROR', message=f'Failed to read EPUB file: {e}')
        raise

    meta_title = book.get_metadata('DC', 'title')
    title = meta_title[0][0] if meta_title else ''
    meta_creator = book.get_metadata('DC', 'creator')
    creator = meta_creator[0][0] if meta_creator else ''

    cover_maybe = find_cover(book)
    cover_image = cover_maybe.get_content() if cover_maybe else b""
    if cover_maybe:
        print(f'Found cover image {cover_maybe.file_name} in {cover_maybe.media_type} format')

    document_chapters = find_document_chapters_and_extract_texts(book, ai_enabled=ai_enabled)

    if not selected_chapters:
        if pick_manually is True:
            selected_chapters = pick_chapters(document_chapters)
        else:
            selected_chapters = find_good_chapters(document_chapters)
    print_selected_chapters(document_chapters, selected_chapters)
    texts = [c.extracted_text for c in selected_chapters]

    has_ffmpeg = shutil.which('ffmpeg') is not None
    if not has_ffmpeg:
        print('\033[91m' + 'ffmpeg not found. Please install ffmpeg to create mp3 and m4b audiobook files.' + '\033[0m')

    stats = SimpleNamespace(
        total_chars=sum(map(len, texts)),
        processed_chars=0,
        chars_per_sec=500 if torch.cuda.is_available() else 50)
    _est_audio_secs = stats.total_chars / _SPOKEN_CHARS_PER_SEC
    _est_encode_secs = _est_audio_secs / _AAC_ENCODE_RT_FACTOR
    _est_tts_secs = stats.total_chars / stats.chars_per_sec
    # Fix: guard against ZeroDivisionError when there's nothing to process
    # (e.g. no chapters selected, or every chapter extracted to empty text).
    _est_total_secs = _est_tts_secs + _est_encode_secs
    stats.tts_progress_share = (_est_tts_secs / _est_total_secs) if _est_total_secs > 0 else 1.0
    stats.estimated_encode_secs = _est_encode_secs
    print('Started at:', time.strftime('%H:%M:%S'))
    print(f'Total characters: {stats.total_chars:,}')
    print('Total words:', len(' '.join(texts).split()))
    eta = strfdelta((stats.total_chars - stats.processed_chars) / stats.chars_per_sec)
    print(f'Estimated time remaining (assuming {stats.chars_per_sec} chars/sec): {eta}')

    set_espeak_library()
    try:
        pipeline = KPipeline(lang_code=lang_code_from_voice(voice))  # fix #5
    except Exception as e:
        print(f'\033[91mFailed to initialize the Kokoro TTS pipeline: {e}\033[0m')
        if post_event:
            post_event('CORE_ERROR', message=f'Failed to initialize TTS pipeline: {e}')
        raise

    chapter_wav_files = []
    for i, chapter in enumerate(selected_chapters, start=1):
        if stop_event and stop_event.is_set():
            print('Synthesis stopped by user.')
            break
        if max_chapters is not None and i > max_chapters:
            break
        text = chapter.extracted_text
        xhtml_file_name = chapter.get_name().replace(' ', '_').replace('/', '_').replace('\\', '_')
        # Fix: include `speed` in the cache-key filename. Previously only
        # `voice` was encoded, so re-running with a different speed would
        # silently reuse WAVs generated at the old speed.
        speed_tag = str(speed).replace('.', 'p')
        chapter_wav_path = Path(output_folder) / filename.replace(
            extension, f'_chapter_{i}_{voice}_{speed_tag}_{xhtml_file_name}.wav')
        chapter_wav_files.append(chapter_wav_path)

        if Path(chapter_wav_path).exists():
            print(f'File for chapter {i} already exists. Skipping')
            stats.processed_chars += len(text)
            if post_event:
                post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
            continue
        if len(text.strip()) < 10:
            print(f'Skipping empty chapter {i}')
            chapter_wav_files.remove(chapter_wav_path)
            # Fix: still count these characters as processed so progress/ETA
            # tracking doesn't permanently under-count the total.
            stats.processed_chars += len(text)
            continue
        if ai_enabled:
            if post_event:
                post_event('CORE_AI_REWRITE', chapter_index=chapter.chapter_index,
                           chapter_total=len(selected_chapters),
                           chunk_index=0, chunk_total=0)
            print(f'AI rewrite: chapter {i} ({len(text):,} chars)')
            text = correct_phonetics_ai(
                text, ai_api_key, model=ai_model,
                stop_event=stop_event, post_event=post_event,
                chapter_index=chapter.chapter_index,
                chapter_total=len(selected_chapters),
            )
            if stop_event and stop_event.is_set():
                print('Synthesis stopped by user during AI rewrite.')
                break
        if i == 1:
            text = f'{title} – {creator}.\n\n' + text

        start_time = time.time()
        if post_event:
            post_event('CORE_CHAPTER_STARTED', chapter_index=chapter.chapter_index)

        audio_segments = gen_audio_segments(
            pipeline, text, voice, speed, stats,
            post_event=post_event, max_sentences=max_sentences, stop_event=stop_event)

        if audio_segments:
            final_audio = np.concatenate(audio_segments)
            peak = np.abs(final_audio).max()
            if peak > 0:
                final_audio = final_audio * (0.708 / peak)
            # Fix: write to a temp file and rename atomically, so a run that
            # is killed mid-write never leaves behind a partial WAV that a
            # later "already exists" resume check would mistake for done.
            tmp_wav_path = chapter_wav_path.with_suffix('.wav.tmp')
            soundfile.write(tmp_wav_path, final_audio, sample_rate, format='WAV', subtype='PCM_16')
            tmp_wav_path.replace(chapter_wav_path)
            end_time = time.time()
            delta_seconds = end_time - start_time
            chars_per_sec = len(text) / delta_seconds
            print('Chapter written to', chapter_wav_path)
            if post_event:
                post_event('CORE_CHAPTER_FINISHED', chapter_index=chapter.chapter_index)
            print(f'Chapter {i} read in {delta_seconds:.2f} seconds ({chars_per_sec:.0f} characters per second)')
        else:
            print(f'Warning: No audio generated for chapter {i}')
            chapter_wav_files.remove(chapter_wav_path)

    if has_ffmpeg and not (stop_event and stop_event.is_set()):
        # Fix: guard against an empty chapter_wav_files list (e.g. every
        # chapter got skipped/empty) instead of feeding ffmpeg an empty
        # concat list and failing with a confusing error.
        if not chapter_wav_files:
            print('\033[93mNo audio was generated for any chapter — skipping M4B creation.\033[0m')
            if post_event:
                post_event('CORE_ERROR', message='No audio was generated for any chapter.')
        else:
            total_audio_secs, chapters_txt_path = create_index_file(title, creator, chapter_wav_files, output_folder)
            create_m4b(chapter_wav_files, filename, cover_image, output_folder,
                       chapters_txt_path=chapters_txt_path,
                       total_audio_secs=total_audio_secs, stats=stats,
                       post_event=post_event, stop_event=stop_event)
            delete_wav_files(chapter_wav_files)
            if post_event:
                stats.progress = 100
                stats.eta = strfdelta(0)
                post_event('CORE_PROGRESS', stats=stats)
                post_event('CORE_FINISHED')

    settings = load_settings()
    save_settings(
        output_folder, voice, speed,
        gemini_api_key=settings.get('gemini_api_key', ''),
        gemini_model=settings.get('gemini_model', 'gemini-3.1-flash-lite'),
        gemini_enabled=settings.get('gemini_enabled', False),
        last_open_dir=settings.get('last_open_dir', ''),
    )


def find_cover(book):
    def is_image(item):
        return item is not None and item.media_type.startswith('image/')

    for item in book.get_items_of_type(ebooklib.ITEM_COVER):
        if is_image(item):
            return item

    for meta in book.get_metadata('OPF', 'cover'):
        if is_image(item := book.get_item_with_id(meta[1]['content'])):
            return item

    if is_image(item := book.get_item_with_id('cover')):
        return item

    for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
        if 'cover' in item.get_name().lower() and is_image(item):
            return item

    return None


def print_selected_chapters(document_chapters, chapters):
    ok = 'X' if platform.system() == 'Windows' else '✅'
    print(tabulate([
        [i, c.get_name(), len(c.extracted_text), ok if c in chapters else '', chapter_beginning_one_liner(c)]
        for i, c in enumerate(document_chapters, start=1)
    ], headers=['#', 'Chapter', 'Text Length', 'Selected', 'First words']))


def gen_audio_segments(pipeline, text, voice, speed, stats=None, max_sentences=None, post_event=None, stop_event=None):
    nlp = get_nlp()  # fix #4: use cached instance
    audio_segments = []
    doc = nlp(text)
    sentences = list(doc.sents)
    for i, sent in enumerate(sentences):
        if stop_event and stop_event.is_set():
            print('Synthesis stopped by user.')
            break
        if max_sentences is not None and i >= max_sentences:  # fix #3: >= not >, and treat 0 correctly
            break
        for gs, ps, audio in pipeline(sent.text, voice=voice, speed=speed, split_pattern=r'\n\n\n'):
            # Fix: Kokoro returns a torch tensor; convert to numpy array so
            # that numpy operations (linspace dtype, arithmetic) work correctly.
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()

            # Apply a short 5 ms fade-in / fade-out to each segment to
            # prevent clicks caused by DC-offset discontinuities at
            # sentence boundaries.
            fade_samples = min(int(sample_rate * 0.005), len(audio) // 4)
            if fade_samples > 0:
                ramp = np.linspace(0.0, 1.0, fade_samples, dtype=audio.dtype)
                audio = audio.copy()
                audio[:fade_samples] *= ramp
                audio[-fade_samples:] *= ramp[::-1]
            audio_segments.append(audio)
        if stats:
            stats.processed_chars += len(sent.text)
            tts_share = getattr(stats, 'tts_progress_share', 1.0)
            stats.progress = int(
                stats.processed_chars / stats.total_chars * tts_share * 100)
            remaining_tts = (stats.total_chars - stats.processed_chars) / stats.chars_per_sec
            remaining_encode = getattr(stats, 'estimated_encode_secs', 0)
            stats.eta = strfdelta(remaining_tts + remaining_encode)
            if post_event:
                post_event('CORE_PROGRESS', stats=stats)
            print(f'Estimated time remaining: {stats.eta}')
            print('Progress:', f'{stats.progress}%\n')
    return audio_segments


def gen_text(text, voice=DEFAULT_VOICE, output_file='text.wav', speed=1, play=False):
    pipeline = KPipeline(lang_code=lang_code_from_voice(voice))
    audio_segments = gen_audio_segments(pipeline, text, voice=voice, speed=speed)
    final_audio = np.concatenate(audio_segments)
    soundfile.write(output_file, final_audio, sample_rate)
    if play:
        subprocess.run(['ffplay', '-autoexit', '-nodisp', output_file])


def _sanitize_api_key(api_key):
    if not api_key:
        return ''
    cleaned = api_key.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').strip()
    if cleaned != api_key or any(c.isspace() for c in api_key):
        cleaned = ' '.join(cleaned.split())
    return cleaned



# ---------------------------------------------------------------------------
# Shared phonetic rewrite rules — single source of truth used by BOTH the
# silent rewrite path (_ai_rewrite_single_chunk) and the human-readable
# analysis path (check_phonetic_transcription_ai), so the two can never
# describe/apply different rules for what counts as "needs fixing".
#
# Note: espeak-ng's [[...]] phoneme-override syntax is intentionally NOT
# offered here. That syntax expects espeak's own Kirshenbaum-based ASCII
# mnemonics (e.g. "w3:ld"), not standard Unicode IPA — Gemini reliably
# produces the latter but not the former, so asking it to emit [[...]]
# phonemes silently produces garbled/unrecognized input to espeak-ng.
# Kokoro's own inline IPA override (plain Unicode IPA, no brackets) is the
# only phonetic-override path offered, since Gemini can produce valid IPA.
# ---------------------------------------------------------------------------
_PHONETIC_RULES = (
    "Only flag/change words or formatting that affect pronunciation, such as:\n"
    "- Expand abbreviations (Dr. -> Doctor, St. -> Saint only when it is a name, "
    "  NASA -> N A S A, FBI -> F B I, etc.)\n"
    "- Spell out numbers and dates in words (2024 -> twenty twenty four, 3rd -> third)\n"
    "- Fix homophones, silent letters, or unusual stress by re-spelling the word\n\n"
    "FOREIGN PROPER NOUNS (mandatory, not optional):\n"
    "Any proper noun using non-English orthography or spelling conventions — accented "
    "letters (é, è, ç, œ, ü, etc.), unusual consonant clusters, silent letters, or "
    "foreign name endings — MUST be rewritten in some form. This applies to place names, "
    "personal names, military unit names, and any other proper noun. Do not leave a "
    "French, German, or other foreign-language name unchanged, even if you are uncertain "
    "of the exact native pronunciation — an approximate English rendering is always better "
    "than leaving the raw spelling for the default G2P to butcher.\n"
    "  Example: 'Amiens' -> 'Amyen' or ˈɑːmiˌæn\n"
    "  Example: 'Péronne' -> 'Peyron' or peɪˈrɒn\n"
    "  Example: 'Flixécourt' -> 'Flixaycoor'\n"
    "  Example: 'GUDERIAN' -> 'Guderian' (normalize casing)\n\n"
    "You MAY use the Kokoro inline phonetic override: replace a word directly with its "
    "standard Unicode IPA phonetic spelling (no brackets needed), using stress marks ˈ "
    "(primary) and ˌ (secondary). Example: Kokoro or Paris rewritten as kˈOkəɹO or ˈpæɹɪs. "
    "Do not use espeak-style double-bracket phoneme syntax (e.g. [[...]]) — it is not "
    "supported here and will not be pronounced correctly.\n\n"
    "Prosody: existing punctuation already controls intonation — "
    "; : , . ! ? — … \" ( ) \u201c \u201d all shape phrasing and pitch. "
    "Do not remove or alter punctuation; it is meaningful for Kokoro.\n\n"
    "- Prefer the simplest fix: only use inline IPA replacements for words that "
    "  the default G2P would misread (names, loanwords, acronyms). Plain English "
    "  respelling is preferred when it's simpler and equally accurate."
)

def check_phonetic_transcription_ai(text, api_key, model='gemini-3.1-flash-lite', stop_event=None):
    """
    Use Google Gemini AI to analyze text for potential TTS pronunciation issues
    and provide phonetic transcription guidance.

    Returns a string with the AI's explanation/suggestions, followed by the
    actual rewritten text — produced by the same correct_phonetics_ai() /
    _ai_rewrite_single_chunk() path used during real synthesis — so what's
    shown here is exactly what would be sent to the TTS engine, not just a
    description of the changes.

    The analysis prompt below shares _PHONETIC_RULES with
    _ai_rewrite_single_chunk so the explanation and the actual rewrite can
    never disagree about what counts as an issue worth fixing.
    """
    if not text.strip():
        return "Error: text is empty."

    api_key = _sanitize_api_key(api_key)
    if not api_key:
        return "Error: API key is missing. Please paste it in the AI Phonetic Check section."
    if not api_key.startswith('AIza'):
        return "Error: API key looks invalid (Gemini keys typically start with 'AIza'). Please check the value you pasted."

    def _do_call():
        client = genai.Client(api_key=api_key)
        return client.models.generate_content(
            model=model,
            contents=(
                "You are a phonetic transcription expert for a Kokoro text-to-speech system. "
                "Analyze the following text from an audiobook and identify words or phrases "
                "that might be mispronounced by the TTS engine, using EXACTLY the same rules "
                "that will be used to actually rewrite this text (listed below), so your analysis "
                "matches what the rewrite step will do.\n\n"
                f"{_PHONETIC_RULES}\n\n"
                "For each issue found, provide:\n"
                "1. The problematic word/phrase\n"
                "2. Why it might be mispronounced\n"
                "3. The correct phonetic transcription (IPA)\n"
                "4. The suggested rewrite, respelling, or [[espeak_phoneme]]/IPA override that "
                "   will actually be applied\n\n"
                "Remember: foreign proper nouns are a MANDATORY category — do not skip any of "
                "them in your analysis, even if you're only approximating the pronunciation.\n\n"
                "If the text looks clean with no obvious issues, say so and provide a brief confirmation.\n\n"
                f"Text to analyze:\n\"\"\"\n{text}\n\"\"\""
            )
        )

    try:
        response = _call_gemini_with_retry(_do_call, stop_event=stop_event)
        analysis = response.text.strip()
    except Exception as e:
        msg = str(e)
        if '404' in msg and 'NOT_FOUND' in msg:
            return (f"Error: the model '{model}' is not available for your API key/account. "
                    f"Please pick a current model in the GUI (e.g. gemini-3.1-flash-lite, "
                    f"gemini-3.5-flash, or gemini-flash-lite-latest) and try again.\n\nDetails: {msg}")
        return f"Error during AI analysis: {msg}"

    # Produce the actual TTS-bound text using the exact same rewrite path
    # (including chunking for long text) that main() uses before synthesis,
    # so the preview matches reality rather than just describing changes.
    rewritten_text = correct_phonetics_ai(text, api_key, model=model, stop_event=stop_event)

    return (
        f"{analysis}\n\n"
        f"{'=' * 60}\n"
        f"REWRITTEN TEXT (this is what will be sent to the TTS engine)\n"
        f"{'=' * 60}\n\n"
        f"{rewritten_text}"
    )

# Rough char-to-token ratio used to size Gemini requests for the
# phonetic rewriter. ~4 chars/token is a defensible average for English
# prose, so 300K tokens ~= 1.2M characters of payload per request.
_AI_REWRITE_MAX_TOKENS = 300_000
_AI_REWRITE_CHARS_PER_TOKEN = 4
_AI_REWRITE_MAX_CHARS = _AI_REWRITE_MAX_TOKENS * _AI_REWRITE_CHARS_PER_TOKEN


def _ai_split_paragraphs(text, max_chars):
    """Split text into chunks of <= max_chars on whitespace, joining whole
    paragraphs back together greedily. Falls back to hard word-splitting
    when a single paragraph exceeds max_chars (rare for cleaned EPUBs)."""
    if len(text) <= max_chars:
        return [text]
    paragraphs = [p for p in text.split('\n\n') if p]
    chunks, current = [], ''
    for para in paragraphs:
        if not current:
            current = para
            continue
        if len(current) + 2 + len(para) <= max_chars:
            current = current + '\n\n' + para
        else:
            chunks.append(current)
            current = para
    if current:
        chunks.append(current)

    if any(len(c) > max_chars for c in chunks):
        refined = []
        for c in chunks:
            if len(c) <= max_chars:
                refined.append(c)
                continue
            for i in range(0, len(c), max_chars):
                refined.append(c[i:i + max_chars])
        chunks = refined
    return chunks


def correct_phonetics_ai(text, api_key, model='gemini-3.1-flash-lite',
                         stop_event=None, post_event=None,
                         chapter_index=None, chapter_total=None):
    """
    Use Google Gemini AI to silently rewrite text for TTS-friendly pronunciation.

    For long inputs the text is split into chunks of at most ~300K tokens
    (≈ 1.2M chars) and each chunk is rewritten in its own request. The
    chunks are joined with blank lines to mirror the paragraph structure
    of the input.

    Returns the corrected text string suitable for direct use as TTS input.
    If AI cannot be reached or returns invalid output, the original text is
    returned unchanged so the caller can still proceed with TTS.
    """
    if not text.strip():
        return text

    api_key = _sanitize_api_key(api_key)
    if not api_key or not api_key.startswith('AIza'):
        return text

    chunks = _ai_split_paragraphs(text, _AI_REWRITE_MAX_CHARS)
    if len(chunks) == 1:
        return _ai_rewrite_single_chunk(chunks[0], api_key, model, stop_event=stop_event, post_event=post_event)

    rewritten = []
    for idx, chunk in enumerate(chunks, start=1):
        if stop_event and stop_event.is_set():
            return text
        if post_event:
            post_event('CORE_AI_REWRITE', chapter_index=chapter_index,
                       chapter_total=chapter_total,
                       chunk_index=idx, chunk_total=len(chunks))
        out = _ai_rewrite_single_chunk(chunk, api_key, model, stop_event=stop_event, post_event=post_event)
        if out == chunk:
            rewritten.append(chunk)
        else:
            rewritten.append(out)
    return '\n\n'.join(rewritten)

def _ai_rewrite_single_chunk(text, api_key, model, stop_event=None, post_event=None):
    """Single-chunk Gemini call used by correct_phonetics_ai. Falls back to
    the original text on any failure or invalid output (after retries).

    Uses a two-section output format (NAMES_FOUND then REWRITTEN_TEXT) to
    force the model to explicitly enumerate foreign/unusual proper nouns
    before it writes the rewrite. Shares _PHONETIC_RULES with
    check_phonetic_transcription_ai so both prompts apply identical rules.
    """
    def _do_call():
        client = genai.Client(api_key=api_key)
        return client.models.generate_content(
            model=model,
            contents=(
                "You are a phonetic preprocessing step for the Kokoro TTS engine. "
                "You will do this in two steps, and your response MUST contain both "
                "sections below, in order, with the exact headers shown.\n\n"
                "STEP 1 — Find every proper noun in the text that does not use standard "
                "English spelling or pronunciation: place names, personal names, military "
                "unit/formation names, or any other proper noun with accented letters "
                "(é, è, ç, œ, ü, etc.), unusual consonant clusters, silent letters, or "
                "foreign-language endings. List each one exactly as it appears in the text, "
                "one per line. If there are none, write 'None found.'\n\n"
                "STEP 2 — Rewrite the full text so the TTS engine reads it correctly. Keep "
                "the meaning, punctuation, and sentence structure exactly the same. Apply "
                "these rules:\n\n"
                f"{_PHONETIC_RULES}\n\n"
                "Every proper noun you listed in Step 1 MUST be changed in some way in the "
                "Step 2 rewrite.\n\n"
                "FORMAT YOUR RESPONSE EXACTLY LIKE THIS (including the headers, nothing before or after):\n"
                "===NAMES_FOUND===\n"
                "<one name per line, or 'None found.'>\n"
                "===REWRITTEN_TEXT===\n"
                "<the full rewritten text, nothing else — no commentary, no quotes, no labels>\n\n"
                f"Text:\n\"\"\"\n{text}\n\"\"\""
            )
        )

    try:
        response = _call_gemini_with_retry(_do_call, stop_event=stop_event, post_event=post_event)
    except Exception as e:
        print(f'\033[91mGemini rewrite failed, keeping original text for this chunk: {e}\033[0m')
        return text

    raw = (response.text or '').strip()
    if not raw or raw.startswith('Error'):
        return text

    corrected = _extract_rewritten_section(raw)
    if corrected is None:
        print('\033[93mAI response missing REWRITTEN_TEXT header; using raw response as-is.\033[0m')
        corrected = raw

    if not corrected or len(corrected) > len(text) * 2:
        return text
    return corrected


def _extract_rewritten_section(raw: str):
    """Pull the REWRITTEN_TEXT section out of the two-section AI response.
    Returns None if the expected header isn't present, so the caller can
    fall back gracefully instead of silently shipping the names list (or
    other junk) to the TTS engine."""
    marker = '===REWRITTEN_TEXT==='
    idx = raw.find(marker)
    if idx == -1:
        return None

    names_section = raw[:idx].replace('===NAMES_FOUND===', '').strip()
    if names_section and names_section.lower() != 'none found.':
        found = [line.strip() for line in names_section.splitlines() if line.strip()]
        print(f'AI flagged {len(found)} foreign/unusual proper noun(s): {", ".join(found)}')

    return raw[idx + len(marker):].strip()

def find_document_chapters_and_extract_texts(book, ai_enabled=False):
    """Returns every chapter that is an ITEM_DOCUMENT and enriches each
    chapter with extracted text.

    When ai_enabled is True the raw extracted text is left untouched —
    clean_text() is skipped — because the AI phonetic rewrite step handles
    all text normalization before TTS.

    Iterates book.spine rather than book.get_items() so that:
      - Only items in the actual reading order are processed (no orphaned
        manifest resources that are never shown to the reader).
      - Duplicate manifest entries for the same content are naturally
        avoided, since the spine lists each idref at most once.
    """
    document_chapters = []
    for idref, _linear in book.spine:
        chapter = book.get_item_with_id(idref)
        if chapter is None or chapter.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        xml = chapter.get_body_content()
        soup = BeautifulSoup(xml, features='lxml')
        chapter.extracted_text = ''
        html_content_tags = ['title', 'p', 'h1', 'h2', 'h3', 'h4', 'li']
        for text in [c.text.strip() for c in soup.find_all(html_content_tags) if c.text]:
            # fix #2: only append a period when the sentence doesn't already
            # end with terminal punctuation (., ?, !)
            if text and text[-1] not in '.?!':
                text += '.'
            chapter.extracted_text += text + '\n'

        # Apply automated text cleaning unless AI pronunciation rewriting is
        # enabled — in that case the AI step handles normalization instead.
        if not ai_enabled:
            chapter.extracted_text = clean_text(chapter.extracted_text)

        document_chapters.append(chapter)
    for i, c in enumerate(document_chapters):
        c.chapter_index = i
    return document_chapters


def is_chapter(c):
    name = c.get_name().lower()
    has_min_len = len(c.extracted_text) > 100
    title_looks_like_chapter = bool(
        'chapter' in name
        or re.search(r'part_?\d{1,3}', name)
        or re.search(r'split_?\d{1,3}', name)
        or re.search(r'ch_?\d{1,3}', name)
        or re.search(r'chap_?\d{1,3}', name)
    )
    return has_min_len and title_looks_like_chapter


def chapter_beginning_one_liner(c, chars=20):
    s = c.extracted_text[:chars].strip().replace('\n', ' ').replace('\r', ' ')
    return s + '…' if len(s) > 0 else ''


def find_good_chapters(document_chapters):
    chapters = [c for c in document_chapters if c.get_type() == ebooklib.ITEM_DOCUMENT and is_chapter(c)]
    if len(chapters) == 0:
        print('Not easy to recognize the chapters, defaulting to all non-empty documents.')
        chapters = [c for c in document_chapters if c.get_type() == ebooklib.ITEM_DOCUMENT and len(c.extracted_text) > 10]
    return chapters


def pick_chapters(chapters):
    chapters_by_names = {
        f'{c.get_name()}\t({len(c.extracted_text)} chars)\t[{chapter_beginning_one_liner(c, 50)}]': c
        for c in chapters}
    title = 'Select which chapters to read in the audiobook'
    ret = pick(list(chapters_by_names.keys()), title, multiselect=True, min_selection_count=1)
    selected_chapters_out_of_order = [chapters_by_names[r[0]] for r in ret]
    selected_chapters = [c for c in chapters if c in selected_chapters_out_of_order]
    return selected_chapters


def strfdelta(tdelta, fmt='{D:02}d {H:02}h {M:02}m {S:02}s'):
    remainder = int(tdelta)
    f = Formatter()
    desired_fields = [field_tuple[1] for field_tuple in f.parse(fmt)]
    possible_fields = ('W', 'D', 'H', 'M', 'S')
    constants = {'W': 604800, 'D': 86400, 'H': 3600, 'M': 60, 'S': 1}
    values = {}
    for field in possible_fields:
        if field in desired_fields and field in constants:
            values[field], remainder = divmod(remainder, constants[field])
    return f.format(fmt, **values)


def delete_wav_files(wav_files):
    """Deletes a list of WAV files."""
    print("Deleting temporary WAV files...")
    for wav_file in wav_files:
        try:
            Path(wav_file).unlink()
            print(f"Deleted: {wav_file}")
        except OSError as e:
            print(f"Error deleting {wav_file}: {e}")


def _popen_run(args, stop_event=None, on_stderr_line=None, **kwargs):
    """
    Drop-in replacement for subprocess.run() that can be interrupted.
    Polls every 0.5 s; if stop_event is set, terminates the child process
    and raises RuntimeError so callers can abort cleanly.

    Pipe buffer fix: if stdout/stderr are piped, drain them in background
    threads so the OS pipe buffer (typically 64 KB on Linux) never fills up
    and blocks the child process — which would cause an unrecoverable deadlock
    on long ffmpeg encodes.  Captured output is stored on proc._stdout_data
    and proc._stderr_data so callers can inspect it after the process exits.

    If on_stderr_line is given, it is called with each stderr line/fragment
    (split on \\r or \\n, matching how ffmpeg writes progress) as it arrives,
    so callers can do real-time progress parsing without duplicating the
    stream-draining logic themselves.
    """
    proc = subprocess.Popen(args, **kwargs)

    stdout_lines, stderr_lines = [], []

    def _drain_stdout(stream, buf):
        for line in stream:
            buf.append(line)

    def _drain_stderr(stream, buf, callback):
        chunk_buf = ''
        while True:
            chunk = stream.read(256)
            if not chunk:
                break
            chunk_buf += chunk
            parts = re.split(r'[\r\n]', chunk_buf)
            chunk_buf = parts[-1]        # keep the incomplete trailing fragment
            for part in parts[:-1]:
                buf.append(part)
                if callback:
                    callback(part)
        if chunk_buf:                    # flush any final fragment
            buf.append(chunk_buf)
            if callback:
                callback(chunk_buf)

    drain_threads = []
    if proc.stdout:
        t = threading.Thread(target=_drain_stdout, args=(proc.stdout, stdout_lines), daemon=True)
        t.start()
        drain_threads.append(t)
    if proc.stderr:
        t = threading.Thread(target=_drain_stderr, args=(proc.stderr, stderr_lines, on_stderr_line), daemon=True)
        t.start()
        drain_threads.append(t)

    while True:
        try:
            proc.wait(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if stop_event and stop_event.is_set():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                for t in drain_threads:
                    t.join()
                raise RuntimeError('Stopped by user.')

    for t in drain_threads:
        t.join()

    proc._stdout_data = ''.join(stdout_lines)
    proc._stderr_data = ''.join(stderr_lines)

    return proc


def create_m4b(chapter_files, filename, cover_image, output_folder, chapters_txt_path,
               total_audio_secs=0, stats=None, post_event=None, stop_event=None):
    """Encode chapter WAVs directly to M4B in a single ffmpeg pass.

    Uses the concat demuxer as the audio source, eliminating the intermediate
    .tmp.mp4 file and halving total disk I/O compared to the two-step approach.
    64 k mono is transparent quality for speech and encodes ~2x faster than 128 k.

    When total_audio_secs > 0, parses ffmpeg's stderr in real time to derive
    accurate progress and ETA from the actual encode speed (speed=Nx lines).

    chapters_txt_path is produced by create_index_file() and passed in here
    (rather than each function independently hardcoding the same filename)
    so the two functions can't drift out of sync, and so the path can be
    made unique per-run (see create_index_file).
    """
    m4b_name = filename.replace('.epub', '.m4b')
    m4b_dir = Path(output_folder) / Path(m4b_name).stem
    m4b_dir.mkdir(parents=True, exist_ok=True)
    final_filename = m4b_dir / m4b_name
    safe_stem = Path(filename).stem.replace("'", "")
    list_file_path = Path(output_folder) / f"{safe_stem}_wav_list_{uuid.uuid4().hex[:8]}.txt"

    with open(list_file_path, 'w') as f:
        for chapter_file in chapter_files:
            escaped = str(Path(chapter_file).resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    print(f"WAV list ({len(chapter_files)} files): {list_file_path}")

    print('Creating M4B file...')
    ffmpeg_command = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0', '-i', str(list_file_path),
        '-i', str(chapters_txt_path),
    ]

    # Fix: give the cover temp file a unique name (like list_file_path
    # already had) so two concurrent conversions writing to the same
    # output_folder don't stomp on each other's temp files.
    cover_file_path = None
    if cover_image:
        cover_file_path = Path(output_folder) / f'cover_temp_{uuid.uuid4().hex[:8]}.jpg'
        cover_file_path.write_bytes(cover_image)
        ffmpeg_command.extend(['-i', str(cover_file_path)])
        ffmpeg_command.extend([
            '-map', '2:v',
            '-c:v', 'copy',
            '-disposition:v', 'attached_pic',
            '-metadata:s:v', 'title=Album cover',
            '-metadata:s:v', 'comment=Cover (front)',
        ])

    ffmpeg_command.extend([
        '-map', '0:a:0',
        '-map_metadata', '1',
        '-c:a', 'aac',
        '-ac', '1',
        '-b:a', '64k',
        '-f', 'mp4',
        str(final_filename),
    ])

    print("FFmpeg command:", " ".join(ffmpeg_command))

    tts_share = getattr(stats, 'tts_progress_share', 0.9) if stats else 0.9
    encode_share = 1.0 - tts_share

    def _process_ffmpeg_line(line):
        """Parse one ffmpeg progress line; post CORE_PROGRESS if we have enough info."""
        if not (stats and post_event and total_audio_secs > 0):
            return
        time_m = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', line)
        if not time_m:
            return
        encoded_secs = (int(time_m.group(1)) * 3600
                        + int(time_m.group(2)) * 60
                        + float(time_m.group(3)))
        fraction = min(encoded_secs / total_audio_secs, 1.0)
        stats.progress = int((tts_share + fraction * encode_share) * 100)
        speed_m = re.search(r'speed=\s*([\d.]+)x', line)
        if speed_m:
            speed = float(speed_m.group(1))
            remaining_secs = max(total_audio_secs - encoded_secs, 0) / speed if speed > 0 else 0
            stats.eta = strfdelta(remaining_secs)
            print(f'Encoding: {fraction*100:.1f}% | speed={speed}x | ETA {stats.eta}')
        post_event('CORE_PROGRESS', stats=stats)

    try:
        # Fix: reuse the shared _popen_run helper (stop_event handling +
        # non-blocking pipe draining) instead of duplicating that logic
        # inline, so there's a single implementation to maintain.
        proc = _popen_run(
            ffmpeg_command,
            stop_event=stop_event,
            on_stderr_line=_process_ffmpeg_line,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode == 0:
            print(f'{final_filename} created. Enjoy your audiobook.')
        else:
            print(f"Error creating M4B file. FFmpeg returned code: {proc.returncode}")
            print("FFmpeg stdout:\n", proc._stdout_data)
            print("FFmpeg stderr:\n", proc._stderr_data)
    except RuntimeError as e:
        print(str(e))
    finally:
        list_file_path.unlink(missing_ok=True)
        if cover_file_path and cover_file_path.exists():
            cover_file_path.unlink()
        if Path(chapters_txt_path).exists():
            Path(chapters_txt_path).unlink()


def probe_duration(file_name):
    """Return the duration (seconds) of a local WAV file.

    Uses soundfile (already a dependency, since we write these WAVs
    ourselves) instead of shelling out to ffprobe per chapter. This avoids
    an extra subprocess per chapter and a previously-unhandled exception
    path: if a WAV can't be read, we log a warning and fall back to 0s
    instead of crashing after all the TTS work for the book is done.
    """
    try:
        info = soundfile.info(str(file_name))
        return info.frames / float(info.samplerate)
    except Exception as e:
        print(f'\033[93mWarning: could not read duration of {file_name} ({e}); assuming 0s.\033[0m')
        return 0.0


def create_index_file(title, creator, chapter_mp3_files, output_folder):
    """Write ffmpeg chapter metadata and return (total_audio_secs, chapters_txt_path).

    The chapters.txt path is given a unique suffix (like the WAV concat
    list already was) so two concurrent conversions in the same
    output_folder can't clobber each other's metadata file, and the path
    is returned so create_m4b() doesn't have to independently guess it.
    """
    chapters_txt_path = Path(output_folder) / f"chapters_{uuid.uuid4().hex[:8]}.txt"
    total_secs = 0.0
    with open(chapters_txt_path, "w", encoding="utf-8") as f:
        f.write(f";FFMETADATA1\ntitle={title}\nartist={creator}\n\n")
        start = 0
        for i, c in enumerate(chapter_mp3_files):
            duration = probe_duration(c)
            total_secs += duration
            end = start + int(duration * 1000)
            f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle=Chapter {i}\n\n")
            start = end
    return total_secs, chapters_txt_path