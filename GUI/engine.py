import os
import json
import subprocess
from pathlib import Path

class LunaVoxEngine:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.models_dir = self.root_dir / "models"
        self.build_dir = self.root_dir / "build"
        self.ref_dir = self.root_dir / "ref"
        self.logs_dir = self.root_dir / "logs"
        self.cli_path = self.build_dir / "qwen3-tts-cli.exe"

    def get_backend_info(self):
        metadata_path = self.build_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8-sig') as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def discover_models(self):
        models = []
        if not self.models_dir.exists():
            return models
        
        for d in self.models_dir.iterdir():
            if d.is_dir():
                profile_path = d / "model_profile.json"
                if profile_path.exists():
                    try:
                        with open(profile_path, 'r', encoding='utf-8-sig') as f:
                            profile = json.load(f)
                            # Pretty name generation
                            size = profile.get("model_size", "unknown").upper()
                            mtype = profile.get("model_type", "unknown").capitalize()
                            display_name = f"{d.name} (Qwen3-TTS-12Hz-{size}-{mtype})"
                            models.append({
                                "id": d.name,
                                "name": display_name,
                                "type": profile.get("model_type", "base"),
                                "path": str(d.absolute()),
                                "profile": profile,
                                "instruct_support": profile.get("instruct_support", False)
                            })
                    except Exception:
                        continue
        return models

    def discover_references(self):
        refs = []
        if not self.ref_dir.exists():
            return refs
        for f in self.ref_dir.iterdir():
            if f.suffix in ['.wav', '.json']:
                refs.append(str(f.relative_to(self.root_dir)))
        return refs

    def run_synthesis(self, args_dict):
        cmd = [str(self.cli_path)]
        
        # Required parameters
        cmd.extend(["-m", args_dict["model_path"]])
        cmd.extend(["-t", args_dict["text"]])
        
        output_path = args_dict.get("output", "output/out.wav")
        # Ensure output directory exists
        out_dir = self.root_dir / os.path.dirname(output_path)
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["-o", output_path])

        # Mode based parameters
        mtype = args_dict["model_type"]
        if args_dict.get("language") and args_dict["language"] != "auto":
            cmd.extend(["-l", args_dict["language"]])

        if mtype == "base":
            if args_dict.get("reference"):
                cmd.extend(["-r", args_dict["reference"]])
                if args_dict.get("ref_text"):
                    cmd.extend(["--ref-text", args_dict["ref_text"]])
        elif mtype == "custom":
            cmd.append("--mode")
            cmd.append("custom")
            cmd.extend(["--speaker", args_dict["speaker"]])
            if args_dict.get("instruct"):
                cmd.extend(["--instruct", args_dict["instruct"]])
        elif mtype == "design":
            cmd.append("--mode")
            cmd.append("design")
            cmd.extend(["--instruct", args_dict["instruct"]])

        # Advanced parameters
        if round(args_dict.get("temperature", 0.6), 2) != 0.6:
            cmd.extend(["--temperature", str(args_dict["temperature"])])
        if round(args_dict.get("predictor_temp", 0.6), 2) != 0.6:
            cmd.extend(["--predictor-temperature", str(args_dict["predictor_temp"])])
        if args_dict.get("max_new_tokens", 400) != 400:
            cmd.extend(["--max-tokens", str(args_dict["max_new_tokens"])])
        if args_dict.get("seed", 42) != 42:
            cmd.extend(["--seed", str(args_dict["seed"])])
        
        if args_dict.get("top_k"):
            cmd.extend(["--predictor-top-k", str(args_dict["top_k"])])
        if args_dict.get("top_p"):
            cmd.extend(["--predictor-top-p", str(args_dict["top_p"])])
        if args_dict.get("repetition_penalty", 1.05) != 1.05:
            cmd.extend(["--repetition-penalty", str(args_dict["repetition_penalty"])])
        if args_dict.get("threads", 4) != 4:
            cmd.extend(["-j", str(args_dict["threads"])])

        print(f"Executing: {' '.join(cmd)}")
        return subprocess.Popen(cmd, cwd=str(self.root_dir), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def get_command_string(self, args_dict):
        # Helper to generate a nice display string of the command
        cmd = [f".\\build\\qwen3-tts-cli.exe `"]
        
        cmd.append(f"  -m models/{args_dict['model_id']} `")
        cmd.append(f"  -t \"{args_dict['text']}\" `")
        
        output_path = args_dict.get("output", "output/out.wav")
        cmd.append(f"  -o {output_path} `")

        if args_dict.get("language") and args_dict["language"] != "auto":
            cmd.append(f"  -l {args_dict['language']} `")

        mtype = args_dict["model_type"]
        if mtype == "base":
            if args_dict.get("reference"):
                cmd.append(f"  -r \"{args_dict.get('reference')}\" `")
                if args_dict.get("ref_text"):
                    cmd.append(f"  --ref-text \"{args_dict['ref_text']}\" `")
        elif mtype == "custom":
            cmd.append("  --mode custom `")
            cmd.append(f"  --speaker {args_dict['speaker']} `")
            if args_dict.get("instruct"):
                cmd.append(f"  --instruct \"{args_dict['instruct']}\" `")
        elif mtype == "design":
            cmd.append("  --mode design `")
            cmd.append(f"  --instruct \"{args_dict['instruct']}\" `")

        # Hide defaults (using round for float safety)
        if round(args_dict.get("temperature", 0.6), 2) != 0.6:
            cmd.append(f"  --temperature {args_dict['temperature']} `")
        if round(args_dict.get("predictor_temp", 0.6), 2) != 0.6:
            cmd.append(f"  --predictor-temperature {args_dict['predictor_temp']} `")
        if args_dict.get("max_new_tokens", 400) != 400:
            cmd.append(f"  --max-tokens {args_dict['max_new_tokens']} `")
        if args_dict.get("seed", 42) != 42:
            cmd.append(f"  --seed {args_dict['seed']} `")
        if args_dict.get("top_k", 50) != 50:
            cmd.append(f"  --predictor-top-k {args_dict['top_k']} `")
        if round(args_dict.get("top_p", 1.0), 2) != 1.0:
            cmd.append(f"  --predictor-top-p {args_dict['top_p']} `")
        if round(args_dict.get("repetition_penalty", 1.05), 2) != 1.05:
            cmd.append(f"  --repetition-penalty {args_dict['repetition_penalty']} `")
        if args_dict.get("threads", 4) != 4:
            cmd.append(f"  -j {args_dict['threads']} `")
        
        # Clean up last backtick
        if cmd[-1].endswith(" `"):
            cmd[-1] = cmd[-1][:-2]
            
        return "\n".join(cmd)
