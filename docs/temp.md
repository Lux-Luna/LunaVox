.\build\qwen3-tts-cli.exe `
   -m models\base_small `
   -t "Hi, this is lunavox speaking English." `
   -r ref\ref_0.6B.json `
   --instruct "Natural and clear speech with a pleasant tone." `
   -o output\lunavox_natural_speech.wav `
   --stats-json output\lunavox_natural_speech.json