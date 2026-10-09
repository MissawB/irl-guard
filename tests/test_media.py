import logging
import subprocess

import cv2

from irl_guard.media import a_une_piste_audio, ffmpeg_exe, finaliser, intervalles_pause


def _nb_images(chemin):
    cap = cv2.VideoCapture(str(chemin))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def _clip_avec_audio(chemin):
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc=duration=2:size=160x120:rate=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(chemin)],
                   check=True)
    return chemin


def test_intervalles_pause():
    assert intervalles_pause([False, True, True, False, True], fps=10) == [(0.1, 0.3), (0.4, 0.5)]
    assert intervalles_pause([False, False], fps=10) == []


def test_finaliser_avec_audio(make_video, tmp_path):
    muette = make_video(n=20, fps=10.0, size=(160, 120))
    source = _clip_avec_audio(tmp_path / "source.mp4")
    sortie = tmp_path / "out.mp4"
    finaliser(muette, source, sortie, [(0.5, 1.0)], couper_audio=True)
    assert a_une_piste_audio(sortie) is True
    assert a_une_piste_audio(muette) is False
    assert _nb_images(sortie) == 20


def test_finaliser_sans_audio(make_video, tmp_path, caplog):
    muette = make_video(n=20, fps=10.0, size=(160, 120))
    sortie = tmp_path / "out.mp4"
    with caplog.at_level(logging.WARNING):
        finaliser(muette, muette, sortie, [], couper_audio=True)
    assert sortie.exists() and a_une_piste_audio(sortie) is False
    assert _nb_images(sortie) == 20
    assert "sans piste audio" in caplog.text
