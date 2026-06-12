#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# audiblez - A program to convert e-books into audiobooks using
# Kokoro-82M model for high-quality text-to-speech synthesis.
# Originally by Claudio Santini 2025 - https://claudio.uk
# Fork by runlevel6 2025

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
            return settings
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_settings(output_folder, voice, speed=1.0):
    """
    Saves settings to the JSON configuration file.
    Accepts an optional speed parameter so both core and UI write a
    consistent schema to the same file.
    """
    settings = {
        'output_folder': str(output_folder),
        'voice': voice,
        'speed': float(speed),
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
    Convert a Roman numeral string to its English word equivalent.
    Returns the original string unchanged if it doesn't parse as a valid
    Roman numeral (guards against false positives like a lone 'C' or 'D').
    """
    n = _roman_to_int(roman_str)
    if n == 0:
        return roman_str
    words = _int_to_words(n)
    # Mirror the capitalisation of the source token so the output blends
    # naturally with surrounding text: ALL-CAPS → Title Case, else lowercase.
    if roman_str.isupper():
        return words.title()
    return words


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
    # Pass 1 — standalone headings
    def _replace_heading(m: re.Match) -> str:
        numeral, punctuation = m.group(1), m.group(2)
        words = _roman_match_to_words(numeral)
        if words == numeral:          # failed to parse — leave untouched
            return m.group(0)
        return words.capitalize() + punctuation

    text = _ROMAN_HEADING_RE.sub(_replace_heading, text)

    # Pass 2 — inline after keyword
    def _replace_inline(m: re.Match) -> str:
        keyword, space, numeral = m.group(1), m.group(2), m.group(3)
        words = _roman_match_to_words(numeral)
        if words == numeral:
            return m.group(0)
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

    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output folder set to: {output_folder}")
    print(f"Voice selected: {voice}")

    filename = Path(file_path).name
    extension = '.epub'
    book = epub.read_epub(file_path)

    meta_title = book.get_metadata('DC', 'title')
    title = meta_title[0][0] if meta_title else ''
    meta_creator = book.get_metadata('DC', 'creator')
    creator = meta_creator[0][0] if meta_creator else ''

    cover_maybe = find_cover(book)
    cover_image = cover_maybe.get_content() if cover_maybe else b""
    if cover_maybe:
        print(f'Found cover image {cover_maybe.file_name} in {cover_maybe.media_type} format')

    document_chapters = find_document_chapters_and_extract_texts(book)

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
    stats.tts_progress_share = _est_tts_secs / (_est_tts_secs + _est_encode_secs)
    stats.estimated_encode_secs = _est_encode_secs
    print('Started at:', time.strftime('%H:%M:%S'))
    print(f'Total characters: {stats.total_chars:,}')
    print('Total words:', len(' '.join(texts).split()))
    eta = strfdelta((stats.total_chars - stats.processed_chars) / stats.chars_per_sec)
    print(f'Estimated time remaining (assuming {stats.chars_per_sec} chars/sec): {eta}')

    set_espeak_library()
    pipeline = KPipeline(lang_code=lang_code_from_voice(voice))  # fix #5

    chapter_wav_files = []
    for i, chapter in enumerate(selected_chapters, start=1):
        if stop_event and stop_event.is_set():
            print('Synthesis stopped by user.')
            break
        if max_chapters and i > max_chapters:
            break
        text = chapter.extracted_text
        xhtml_file_name = chapter.get_name().replace(' ', '_').replace('/', '_').replace('\\', '_')
        chapter_wav_path = Path(output_folder) / filename.replace(extension, f'_chapter_{i}_{voice}_{xhtml_file_name}.wav')
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
            continue
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
            soundfile.write(chapter_wav_path, final_audio, sample_rate)
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
        total_audio_secs = create_index_file(title, creator, chapter_wav_files, output_folder)
        create_m4b(chapter_wav_files, filename, cover_image, output_folder,
                   total_audio_secs=total_audio_secs, stats=stats,
                   post_event=post_event, stop_event=stop_event)
        delete_wav_files(chapter_wav_files)
        if post_event:
            stats.progress = 100
            stats.eta = strfdelta(0)
            post_event('CORE_PROGRESS', stats=stats)
            post_event('CORE_FINISHED')

    save_settings(output_folder, voice, speed)


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
        if max_sentences and i >= max_sentences:  # fix #3: >= not >
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


def find_document_chapters_and_extract_texts(book):
    """Returns every chapter that is an ITEM_DOCUMENT and enriches each
    chapter with cleaned extracted_text.

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

        # Apply automated text cleaning before anything else sees this text
        chapter.extracted_text = clean_text(chapter.extracted_text)

        document_chapters.append(chapter)
    for i, c in enumerate(document_chapters):
        c.chapter_index = i
    return document_chapters


def is_chapter(c):
    name = c.get_name().lower()
    has_min_len = len(c.extracted_text) > 100
    title_looks_like_chapter = bool(
        'chapter' in name.lower()
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


def _popen_run(args, stop_event=None, **kwargs):
    """
    Drop-in replacement for subprocess.run() that can be interrupted.
    Polls every 0.5 s; if stop_event is set, terminates the child process
    and raises RuntimeError so callers can abort cleanly.

    Pipe buffer fix: if stdout/stderr are piped, drain them in background
    threads so the OS pipe buffer (typically 64 KB on Linux) never fills up
    and blocks the child process — which would cause an unrecoverable deadlock
    on long ffmpeg encodes.  Captured output is stored on proc._stdout_data
    and proc._stderr_data so callers can inspect it after the process exits.
    """
    proc = subprocess.Popen(args, **kwargs)

    stdout_lines, stderr_lines = [], []

    def _drain(stream, buf):
        for line in stream:
            buf.append(line)

    drain_threads = []
    if proc.stdout:
        t = threading.Thread(target=_drain, args=(proc.stdout, stdout_lines), daemon=True)
        t.start()
        drain_threads.append(t)
    if proc.stderr:
        t = threading.Thread(target=_drain, args=(proc.stderr, stderr_lines), daemon=True)
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
                raise RuntimeError('Stopped by user.')

    for t in drain_threads:
        t.join()

    proc._stdout_data = ''.join(stdout_lines)
    proc._stderr_data = ''.join(stderr_lines)

    return proc


def create_m4b(chapter_files, filename, cover_image, output_folder,
               total_audio_secs=0, stats=None, post_event=None, stop_event=None):
    """Encode chapter WAVs directly to M4B in a single ffmpeg pass.

    Uses the concat demuxer as the audio source, eliminating the intermediate
    .tmp.mp4 file and halving total disk I/O compared to the two-step approach.
    64 k mono is transparent quality for speech and encodes ~2x faster than 128 k.

    When total_audio_secs > 0, parses ffmpeg's stderr in real time to derive
    accurate progress and ETA from the actual encode speed (speed=Nx lines).
    ffmpeg writes progress with \\r rather than \\n when stderr is piped, so we
    read in 256-byte chunks and split on both characters.
    """
    m4b_name = filename.replace('.epub', '.m4b')
    m4b_dir = Path(output_folder) / Path(m4b_name).stem
    m4b_dir.mkdir(parents=True, exist_ok=True)
    final_filename = m4b_dir / m4b_name
    chapters_txt_path = Path(output_folder) / "chapters.txt"
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

    cover_file_path = None
    if cover_image:
        cover_file_path = Path(output_folder) / 'cover_temp_image.jpg'
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

    proc = subprocess.Popen(ffmpeg_command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)

    stdout_lines, stderr_lines = [], []
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

    def drain_stdout():
        for line in proc.stdout:
            stdout_lines.append(line)

    def drain_stderr():
        buf = ''
        while True:
            chunk = proc.stderr.read(256)
            if not chunk:
                break
            buf += chunk
            parts = re.split(r'[\r\n]', buf)
            buf = parts[-1]          # keep the incomplete trailing fragment
            for part in parts[:-1]:
                stderr_lines.append(part)
                _process_ffmpeg_line(part)
        if buf:                      # flush any final fragment
            stderr_lines.append(buf)
            _process_ffmpeg_line(buf)

    stdout_thread = threading.Thread(target=drain_stdout, daemon=True)
    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stdout_thread.start()
    stderr_thread.start()

    try:
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
                    raise RuntimeError('Stopped by user.')

        stdout_thread.join()
        stderr_thread.join()

        if proc.returncode == 0:
            print(f'{final_filename} created. Enjoy your audiobook.')
        else:
            print(f"Error creating M4B file. FFmpeg returned code: {proc.returncode}")
            print("FFmpeg stdout:\n", ''.join(stdout_lines))
            print("FFmpeg stderr:\n", ''.join(stderr_lines))
    finally:
        list_file_path.unlink(missing_ok=True)
        if cover_file_path and cover_file_path.exists():
            cover_file_path.unlink()
        if chapters_txt_path.exists():
            chapters_txt_path.unlink()


def probe_duration(file_name):
    args = ['ffprobe', '-i', file_name, '-show_entries', 'format=duration',
            '-v', 'quiet', '-of', 'default=noprint_wrappers=1:nokey=1']
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    return float(proc.stdout.strip())


def create_index_file(title, creator, chapter_mp3_files, output_folder):
    """Write ffmpeg chapter metadata and return total audio duration in seconds."""
    total_secs = 0.0
    with open(Path(output_folder) / "chapters.txt", "w", encoding="utf-8") as f:
        f.write(f";FFMETADATA1\ntitle={title}\nartist={creator}\n\n")
        start = 0
        for i, c in enumerate(chapter_mp3_files):
            duration = probe_duration(c)
            total_secs += duration
            end = start + int(duration * 1000)
            f.write(f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle=Chapter {i}\n\n")
            start = end
    return total_secs
