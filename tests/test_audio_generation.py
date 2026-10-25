"""Pipeline regression tests: real tensors/WAV I/O, without gated model weights.

Only the model, authentication, and Gradio are stubbed. These tests do not
measure generated audio quality. Run: python -m unittest discover -s tests -v
"""
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock
import torch
import torchaudio


ROOT = Path(__file__).resolve().parents[1]


class AudioGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        generation = types.ModuleType("stable_audio_tools.inference.generation")
        self.upstream = Mock()
        generation.generate_diffusion_cond = self.upstream
        inference = types.ModuleType("stable_audio_tools.inference")
        inference.generation = generation
        stable_audio = types.ModuleType("stable_audio_tools")
        stable_audio.inference = inference
        stable_audio.get_pretrained_model = Mock()
        gradio = types.ModuleType("gradio")
        gradio.Progress = Mock(return_value=Mock())
        auth = types.ModuleType("auth")
        for name in ("login_with_token", "login_with_env_token",
                     "is_authenticated", "check_model_access"):
            setattr(auth, name, Mock(return_value=True))
        # Import the real app and real seed wrapper against a model-call spy.
        modules = {
            "stable_audio_tools": stable_audio,
            "stable_audio_tools.inference": inference,
            "stable_audio_tools.inference.generation": generation,
            "gradio": gradio,
            "auth": auth,
        }
        # Restore only our replacements, not Torch's lazily imported modules.
        missing = object()
        previous = {name: sys.modules.get(name, missing)
                    for name in (*modules, "seed_patch")}
        def restore_modules():
            for name, value in previous.items():
                if value is missing:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = value
        self.addCleanup(restore_modules)
        sys.modules.update(modules)
        sys.modules.pop("seed_patch", None)
        spec = importlib.util.spec_from_file_location("audio_app_under_test", ROOT / "app.py")
        self.app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.app)
        self.app.OUTPUT_DIR = self.temp.name
        self.app.model = Mock()
        self.app.device = "cpu"
        self.app.sample_rate = 44100
        self.app.sample_size = 524288

    def generate(self, waveform, **kwargs):
        self.upstream.return_value = waveform.clone()
        path, seed = self.app.generate_audio("birds chirping", **kwargs)
        audio, rate = torchaudio.load(path)
        self.assertEqual(rate, 44100)
        self.assertEqual(audio.shape[0], 2)
        return audio, seed

    def test_documented_sampler_and_guidance_reach_upstream(self):
        self.generate(torch.ones(1, 2, 524288), seed=123)
        kwargs = self.upstream.call_args.kwargs
        self.assertEqual(kwargs.get("sampler_type"), "pingpong")
        self.assertEqual(kwargs["cfg_scale"], 1.0)
        self.assertEqual(kwargs["steps"], 8)
        self.assertEqual(kwargs["sample_size"], 524288)
        self.assertEqual(kwargs["conditioning"], [
            {"prompt": "birds chirping", "seconds_total": 5.0}
        ])

    def test_reported_seed_is_the_sampler_seed(self):
        for seed in (123, -1, 2**32):
            with self.subTest(seed=seed):
                _, used = self.generate(torch.ones(1, 2, 524288), seed=seed)
                self.assertEqual(self.upstream.call_args.kwargs["seed"], used)
                self.assertTrue(0 <= used < 2**31 - 1)

    def test_duration_crop_removes_tail_before_peak_normalization(self):
        for duration in (1.0, 5.0, 8.5, 11.0):
            with self.subTest(duration=duration):
                frames = int(duration * 44100)
                waveform = torch.full((1, 2, 524288), 100.0)
                waveform[:, 0, :frames] = 0.25
                waveform[:, 1, :frames] = -0.5
                audio, _ = self.generate(waveform, duration=duration, seed=123)
                self.assertEqual(audio.shape, (2, frames))
                # An out-of-duration peak must not change retained audio gain.
                torch.testing.assert_close(audio[0], torch.full((frames,), 0.5))
                torch.testing.assert_close(audio[1], torch.full((frames,), -1.0))

    def test_silence_stays_finite(self):
        audio, _ = self.generate(torch.zeros(1, 2, 524288), seed=123)
        self.assertTrue(torch.isfinite(audio).all())
        self.assertEqual(torch.count_nonzero(audio).item(), 0)

    def test_custom_generation_controls_still_forwarded(self):
        self.generate(torch.ones(1, 2, 524288), steps=12, cfg_scale=2.0, seed=123)
        self.assertEqual(self.upstream.call_args.kwargs["steps"], 12)
        self.assertEqual(self.upstream.call_args.kwargs["cfg_scale"], 2.0)

    def test_loaded_model_is_put_in_evaluation_mode(self):
        loaded_model = Mock()
        loaded_model.to.return_value = loaded_model
        loaded_model.eval.return_value = loaded_model
        loaded_model.parameters.return_value = iter(())
        self.app.get_pretrained_model.return_value = (
            loaded_model, {"sample_rate": 44100, "sample_size": 524288}
        )
        result = self.app.load_model()
        loaded_model.eval.assert_called_once_with()
        self.assertIs(result[0], loaded_model)

    def test_seed_patch_preserves_sampler_options_and_safe_seed(self):
        import seed_patch
        seed_patch.patched_generate_diffusion_cond(
            self.app.model, seed=2**32, sampler_type="pingpong"
        )
        self.assertEqual(self.upstream.call_args.kwargs["sampler_type"], "pingpong")
        self.assertEqual(self.upstream.call_args.kwargs["seed"], (2**32) % (2**31 - 1))


if __name__ == "__main__":
    unittest.main()
