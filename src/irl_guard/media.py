"""Finalisation vidéo avec le binaire ffmpeg fourni par imageio-ffmpeg."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import imageio_ffmpeg

log = logging.getLogger(__name__)


def ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def a_une_piste_audio(chemin: str | Path) -> bool:
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(chemin)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return "Audio:" in r.stderr


def intervalles_pause(drapeaux: list[bool], fps: float) -> list[tuple[float, float]]:
    intervalles, debut = [], None
    for i, pause in enumerate([*drapeaux, False]):
        if pause and debut is None:
            debut = i
        elif not pause and debut is not None:
            intervalles.append((round(debut / fps, 3), round(i / fps, 3)))
            debut = None
    return intervalles


def finaliser(video_muette: str | Path, source: str | Path, sortie: str | Path,
              intervalles: list[tuple[float, float]], couper_audio: bool) -> None:
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(video_muette)]
    if a_une_piste_audio(source):
        cmd += ["-i", str(source), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest"]
        if couper_audio and intervalles:
            expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in intervalles)
            cmd += ["-af", f"volume=enable='{expr}':volume=0"]
    else:
        log.warning("Vidéo source sans piste audio : sortie sans audio (%s)", source)
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", str(sortie)]
    subprocess.run(cmd, check=True)
