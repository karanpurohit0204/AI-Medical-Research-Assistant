"""
Phase 4 — TTS Service
Uses gTTS (fully free, no API key) with ElevenLabs (SDK v2.x) as optional upgrade.
"""

import time
from pathlib import Path
from backend.core.config import get_settings
from backend.core.logging import logger


class TTSService:

    def __init__(self):
        self.settings = get_settings()
        self.output_dir = Path(self.settings.temp_audio_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.settings.elevenlabs_api_key:
            self.engine = "elevenlabs"
        else:
            self.engine = "gtts"

        logger.info(f"TTS service ready | engine={self.engine}")

    def speak(self, text: str) -> str:
        timestamp = int(time.time())
        try:
            if self.engine == "elevenlabs":
                return self._elevenlabs(text, timestamp)
        except Exception as e:
            logger.warning(f"ElevenLabs failed ({e}), falling back to gTTS")
        return self._gtts(text, timestamp)

    def _gtts(self, text: str, timestamp: int) -> str:
        from gtts import gTTS
        output_path = str(self.output_dir / f"answer_{timestamp}.mp3")
        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(output_path)
        logger.info(f"gTTS audio saved → {output_path}")
        return output_path

    def _elevenlabs(self, text: str, timestamp: int) -> str:
        """Compatible with elevenlabs SDK v2.x"""
        from elevenlabs import ElevenLabs
        output_path = str(self.output_dir / f"answer_{timestamp}.mp3")

        client = ElevenLabs(api_key=self.settings.elevenlabs_api_key)
        voice_id = self.settings.elevenlabs_voice_id or "21m3hNDoDP8UNvVxNbwu"
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_turbo_v2",
            output_format="mp3_44100_128",
        )
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        logger.info(f"ElevenLabs audio saved → {output_path}")
        return output_path