import argparse
import shlex
from rich.table import Table
from typing import Optional, Callable
from .Audio.ReferenceAudio import ReferenceAudio
from .Utils.Shared import console, context
from .Utils.UserData import userdata_manager
from .ModelManager import model_manager
from .Core.TTSPlayer import tts_player


class Client:
    def __init__(self):
        self.commands: dict[str, Callable] = {
            'load': self._handle_load,
            'unload': self._handle_unload,
            'speaker': self._handle_speaker,
            'prompt': self._handle_prompt,
            'say': self._handle_say,
            'stop': self._handle_stop,
            'help': self._handle_help,
        }

    def _handle_load(self, args_list: list):
        """
        Load a character model (if model_path is omitted, the last used path will be applied).
        Usage: /load <character_name> <model_path>
        """
        parser = argparse.ArgumentParser(prog="/load", description=self._handle_load.__doc__.strip())
        parser.add_argument('character', help='Name of the character to load.')
        parser.add_argument('path', nargs='?', default=None,
                            help='Directory path of the character model. Use double quotes if it contains backslashes. (Optional)')

        try:
            args = parser.parse_args(args_list)
            model_path: Optional[str] = args.path
            all_cached_paths: dict[str, str] = userdata_manager.get('last_model_paths', {})

            # If the user didn't provide a path, try loading from cache
            if model_path is None:
                if not all_cached_paths or args.character not in all_cached_paths:
                    console.print("[bold red]Error:[/bold red] You did not provide a model folder path.")
                    return
                model_path = all_cached_paths[args.character]
                console.print(f"No path provided, using cached path: [green]{model_path}[/green]")

            # Load and update cache
            model_manager.load_character(character_name=args.character, model_dir=model_path)
            all_cached_paths[args.character] = model_path
            userdata_manager.set('last_model_paths', all_cached_paths)
            console.print(f"Character '{args.character}' loaded successfully!")

        except SystemExit:
            pass  # Catch argparse -h or errors to prevent program exit
        except Exception as e:
            console.print(f"[bold red]An unknown error occurred while loading:[/bold red] {e}")

    def _handle_unload(self, args_list: list):
        """
        Unload a character model and release resources.
        Usage: /unload <character_name>
        """
        parser = argparse.ArgumentParser(prog="/unload", description=self._handle_unload.__doc__.strip())
        parser.add_argument('character', help='Name of the character to unload.')
        try:
            args = parser.parse_args(args_list)
            model_manager.remove_character(character_name=args.character)
            console.print(f"Character '{args.character}' has been unloaded.")
        except SystemExit:
            pass

    def _handle_speaker(self, args_list: list):
        """
        Switch the current speaker.
        Usage: /speaker <character_name>
        """
        parser = argparse.ArgumentParser(prog="/speaker", description=self._handle_speaker.__doc__.strip())
        parser.add_argument('character', help='Name of the character to switch to.')
        try:
            args = parser.parse_args(args_list)
            if not model_manager.has_character(args.character):
                console.print(
                    "[bold red]Error:[/bold red] The character does not exist. Please load the character first.")
                return
            context.current_speaker = args.character
            console.print(f"Current speaker set to '{args.character}'.")
        except SystemExit:
            pass

    def _handle_prompt(self, args_list: list):
        """
        Set reference audio and text.
        Usage: /prompt <audio_path> <text>
        """
        parser = argparse.ArgumentParser(prog="/prompt", description=self._handle_prompt.__doc__.strip())
        parser.add_argument('audio_path', help='Path to the reference audio.')
        parser.add_argument('text', help='Text corresponding to the reference audio.')
        try:
            args = parser.parse_args(args_list)
            context.current_prompt_audio = ReferenceAudio(prompt_wav=args.audio_path, prompt_text=args.text)
            console.print("Reference audio set successfully.")
        except SystemExit:
            pass

    def _handle_say(self, args_list: list):
        """
        Text-to-speech synthesis.
        Usage: /say <text_to_say> [-o/--output path] [--play]
        """
        parser = argparse.ArgumentParser(prog="/say", description=self._handle_say.__doc__.strip())
        parser.add_argument('text', help='Text to convert to speech.')
        parser.add_argument('-o', '--output', help='File path to save the audio. (Optional)')
        parser.add_argument('--play', action='store_true', help='Play the generated audio. (Optional)')
        try:
            args = parser.parse_args(args_list)
            tts_player.start_session(
                play=args.play,
                save_path=args.output
            )
            tts_player.feed(args.text)
            tts_player.end_session()
            tts_player.wait_for_tts_completion()
        except SystemExit:
            pass

    @staticmethod
    def _handle_stop(args_list: list):
        """
        Stop all current and pending tasks.
        """
        try:
            tts_player.stop()
            console.print("All tasks have been stopped.")
        except SystemExit:
            pass

    def _handle_help(self, args_list: list):
        """
        Display help information for all commands.
        """
        console.print("\nAvailable commands:", justify="left")

        table = Table(box=None, show_header=False, pad_edge=False)
        table.add_column("Command", style="bold cyan", width=15)
        table.add_column("Description")

        for cmd, handler in self.commands.items():
            doc = handler.__doc__
            if not doc:
                description = "[italic]No description[/italic]"
            else:
                # Clean and split docstring
                doc_lines = [line.strip() for line in doc.strip().split('\n')]
                description = "\n".join(doc_lines) + "\n"

            table.add_row(f"/{cmd}", description)

        console.print(table)

    def run(self):
        """
        Start the interactive main loop of the client.
        """
        console.print(
            "Welcome to the [bold cyan]LunaVox[/bold cyan] CLI. Type [bold blue]/help[/bold blue] for help, press Ctrl+C to exit."
        )

        while True:
            try:
                raw_input = console.input("[bold]>> [/bold]")

                if not raw_input:
                    continue
                if not raw_input.startswith('/'):
                    console.print("[bold red]Error:[/bold red] Commands must start with '/'. Use /help for assistance.")
                    continue

                parts = shlex.split(raw_input[1:])
                if not parts:
                    continue

                command_name = parts[0].lower()
                command_args = parts[1:]

                handler = self.commands.get(command_name)
                if handler:
                    handler(command_args)
                else:
                    console.print(f"[bold red]Error:[/bold red] Unknown command '[yellow]/{command_name}[/yellow]'.")

            except (KeyboardInterrupt, EOFError):
                break
