import threading


_nltk_setup_lock = threading.Lock()
_nltk_setup_done = False


def ensure_nltk_data() -> None:
    """Ensure required NLTK datasets are available.

    Downloads quietly on first use and no-ops on subsequent calls.
    Swallows errors to avoid breaking runtime in offline environments.
    """
    global _nltk_setup_done
    if _nltk_setup_done:
        return

    with _nltk_setup_lock:
        if _nltk_setup_done:
            return
        try:
            import nltk

            resources_and_paths = {
                "punkt": "tokenizers/punkt",
                "punkt_tab": "tokenizers/punkt_tab",
                "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
                "cmudict": "corpora/cmudict",
            }

            for resource, path in resources_and_paths.items():
                try:
                    nltk.data.find(path)
                except LookupError:
                    try:
                        nltk.download(resource, quiet=True)
                    except Exception:
                        # Ignore download failures (e.g., offline). Let runtime raise if truly needed.
                        pass
        except Exception:
            # If nltk is missing or any unexpected error occurs, do not hard-fail here.
            pass
        finally:
            _nltk_setup_done = True


