#!/usr/bin/env python3
"""
scripts/record_and_query.py

End-to-end terminal test for the Medical RAG Assistant.

Phase 1 test:
  1. Records your voice from the microphone
  2. Transcribes it with Whisper (local, free)
  3. Prints the transcribed text

Run from project root:
    python scripts/record_and_query.py

Later phases will be added here as they are built:
  - Phase 3: PubMed retrieval
  - Phase 5: LLM answer
  - Phase 7: TTS playback
"""

import sys
import os
import argparse

# Make sure Python can find the backend package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.config import get_settings
from backend.core.logging import setup_logging, logger
from backend.utils.audio import record_from_mic, record_until_silence, validate_audio_file
from backend.services.stt.whisper_service import WhisperSTTService


def parse_args():
    parser = argparse.ArgumentParser(description="Medical RAG Assistant — terminal test")
    parser.add_argument(
        "--mode",
        choices=["fixed", "silence"],
        default="fixed",
        help="fixed = record for N seconds | silence = stop automatically on silence",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Duration in seconds for fixed mode (default: from .env AUDIO_RECORD_SECONDS)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Skip recording and transcribe an existing audio file instead",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override Whisper model (tiny|base|small|medium|large)",
    )
    return parser.parse_args()


def main():
    setup_logging()
    settings = get_settings()
    args = parse_args()

    # Override model if specified
    if args.model:
        os.environ["WHISPER_MODEL"] = args.model

    print("=" * 60)
    print("  🏥  Medical RAG Assistant — Terminal Test  ")
    print("=" * 60)
    print(f"  Whisper model : {settings.whisper_model}")
    print(f"  Sample rate   : {settings.audio_sample_rate} Hz")
    print("=" * 60)

    # ── Step 1: Get audio ─────────────────────────────────────
    if args.file:
        # Transcribe an existing file
        audio_path = args.file
        print(f"\n📂  Using file: {audio_path}")
        meta = validate_audio_file(audio_path)
        print(f"    Duration : {meta['duration_seconds']}s")
        print(f"    Format   : {meta['format']} @ {meta['sample_rate']}Hz")
    else:
        # Record from mic
        print("\nPRESS ENTER to start recording...")
        input()

        if args.mode == "silence":
            audio_path = record_until_silence(max_duration=30)
        else:
            audio_path = record_from_mic(duration_seconds=args.duration)

    # ── Step 2: Transcribe ────────────────────────────────────
    print("🔄  Transcribing with Whisper...")
    stt = WhisperSTTService()

    import time
    start = time.time()
    result = stt.transcribe_file(audio_path)
    elapsed = time.time() - start

    # ── Step 3: Print result ──────────────────────────────────
    print("\n" + "─" * 60)
    print("📝  TRANSCRIPTION RESULT")
    print("─" * 60)
    print(f"  Text     : {result.text}")
    print(f"  Language : {result.language}")
    print(f"  Segments : {len(result.segments)}")
    print(f"  Time     : {elapsed:.2f}s")
    print("─" * 60)

    # ── Phase 3+ placeholder ──────────────────────────────────
    if result.text:
        print("\n🔜  Next: Phase 3 will send this query to PubMed + ChromaDB")
        print(f"   Query ready: '{result.text}'")

    print("\n✅  Phase 1 complete.\n")
    return result.text


if __name__ == "__main__":
    main()