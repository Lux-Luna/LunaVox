import argparse
import shlex
import logging
from typing import Optional, Callable
from .Audio.ReferenceAudio import ReferenceAudio
from .Utils.Shared import context
from .Utils.UserData import userdata_manager
from .ModelManager import model_manager
from .Core.TTSPlayer import tts_player

# Use print for CLI output, or logger if preferred.
# For a CLI shell, print is standard.
# If we want to support the user's request of "standard logging", we could use logger,
# but for an interactive shell, print is expected.
# I will use print for interaction, and logger for internal logs.

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
                    print("Error: You did not provide a model folder path.")
                    return
                model_path = all_cached_paths[args.character]
                print(f"No path provided, using cached path: {model_path}")

            # Load and update cache
            model_manager.load_character(character_name=args.character, model_dir=model_path)
            all_cached_paths[args.character] = model_path
            userdata_manager.set('last_model_paths', all_cached_paths)
            print(f"Character '{args.character}' loaded successfully!")

        except SystemExit:
            pass  # Catch argparse -h or errors to prevent program exit
        except Exception as e:
            print(f"An unknown error occurred while loading: {e}")

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
            print(f"Character '{args.character}' has been unloaded.")
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
                print("Error: The character does not exist. Please load the character first.")
                return
            context.current_speaker = args.character
            print(f"Current speaker set to '{args.character}'.")
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
            print("Reference audio set successfully.")
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
            print("All tasks have been stopped.")
        except SystemExit:
            pass

    def _handle_help(self, args_list: list):
        """
        Display help information for all commands.
        """
        print("\nAvailable commands:")
        print(f"{'Command':<15} Description")
        print("-" * 40)

        for cmd, handler in self.commands.items():
            doc = handler.__doc__
            if not doc:
                description = "No description"
            else:
                # Clean and split docstring
                doc_lines = [line.strip() for line in doc.strip().split('\n')]
                description = doc_lines[0] # Take first line for summary

            print(f"/{cmd:<14} {description}")

    def run(self):
        """
        Start the interactive main loop of the client.
        """
        print(
            "Welcome to the LunaVox CLI. Type /help for help, press Ctrl+C to exit."
        )

        while True:
            try:
                raw_input = input(">> ")

                if not raw_input:
                    continue
                if not raw_input.startswith('/'):
                    print("Error: Commands must start with '/'. Use /help for assistance.")
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
                    print(f"Error: Unknown command '/{command_name}'.")

            except (KeyboardInterrupt, EOFError):
                break
