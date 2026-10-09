import pytest

from irl_guard.sources import VideoSource


def test_proprietes_et_iteration(make_video):
    src = VideoSource(make_video(n=12, fps=10.0, size=(320, 240)))
    assert (src.width, src.height) == (320, 240)
    assert src.fps == pytest.approx(10.0)
    frames = list(src)
    assert [f.index for f in frames] == list(range(12))
    assert frames[5].t == pytest.approx(0.5)
    assert frames[0].image.shape == (240, 320, 3)
    assert len(list(src)) == 12  # relisible


def test_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoSource(tmp_path / "absent.mp4")
