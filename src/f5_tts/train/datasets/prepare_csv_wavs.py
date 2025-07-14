import concurrent.futures
import multiprocessing
import os
import shutil
import signal
import subprocess  # For invoking ffprobe
import sys
from contextlib import contextmanager
from pathlib import Path
import argparse
import csv
import json
from importlib.resources import files

import torchaudio
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm

# It's good practice to add a guard for scripts that might be imported
if __name__ == "__main__":
    sys.path.append(os.getcwd())
    from f5_tts.model.utils import convert_char_to_pinyin


# --- Configuration ---
PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt")
MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU free
THREAD_NAME_PREFIX = "AudioTextProcessor"

# Use a simple flag for signal handling to avoid complexities with global executors
shutdown_flag = multiprocessing.Event()


@contextmanager
def graceful_exit(executor):
    """Context manager for graceful shutdown on signals."""
    def signal_handler(signum, frame):
        print("\nReceived signal to terminate. Requesting graceful shutdown...")
        shutdown_flag.set()
        # The executor shutdown logic is now handled in the main function's finally block.

    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        yield
    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)


def get_audio_duration(audio_path, timeout=5):
    """
    Get the duration of an audio file in seconds using ffprobe, with a fallback to torchaudio.
    """
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=timeout
        )
        duration_str = result.stdout.strip()
        if duration_str:
            return float(duration_str)
        raise ValueError("Empty duration string from ffprobe.")
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError, ValueError) as e:
        # FileNotFoundError is added in case ffprobe is not installed
        print(f"Warning: ffprobe failed for {audio_path} ({type(e).__name__}). Falling back to torchaudio.")
        try:
            audio, sample_rate = torchaudio.load(audio_path)
            duration = audio.shape[1] / sample_rate
            if duration <= 0:
                raise ValueError("Non-positive duration from torchaudio.")
            return duration
        except Exception as torchaudio_e:
            raise RuntimeError(f"Both ffprobe and torchaudio failed for {audio_path}: {torchaudio_e}")


# OPTIMIZATION: New worker function combines I/O and CPU tasks.
def process_item(audio_path_text_pair, polyphone):
    """
    Processes a single audio-text pair.
    1. Checks audio file existence and duration (I/O-bound).
    2. Converts text to pinyin (CPU-bound).
    Returns a dictionary with the processed data or None on failure.
    """
    audio_path, text = audio_path_text_pair
    
    if shutdown_flag.is_set():
        return None

    if not audio_path.exists():
        print(f"Warning: Audio file not found: {audio_path}. Skipping.")
        return None

    try:
        duration = get_audio_duration(audio_path)
        
        # OPTIMIZATION: Pinyin conversion is now inside the threaded worker.
        # It's wrapped in a list because the original function expects a batch.
        converted_text = convert_char_to_pinyin([text], polyphone=polyphone)[0]
        
        return {
            "audio_path": audio_path.as_posix(),
            "text": converted_text,
            "duration": duration,
        }
    except Exception as e:
        print(f"Warning: Failed to process {audio_path} due to error: {e}. Skipping.")
        return None


def read_audio_text_pairs(metadata_path: Path):
    """Reads the metadata CSV and yields Path objects and text."""
    parent_dir = metadata_path.parent
    with open(metadata_path, mode="r", newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.reader(csvfile, delimiter="|")
        try:
            next(reader)  # Skip header
        except StopIteration:
            return # Empty file
            
        for row in reader:
            if len(row) >= 2:
                audio_file, text = row[0].strip(), row[1].strip()
                yield (parent_dir / audio_file, text)


def prepare_csv_wavs_dir(input_dir: Path, num_workers: int = None):
    metadata_path = input_dir / "metadata.csv"
    wavs_path = input_dir / "wavs"
    if not (metadata_path.is_file() and wavs_path.is_dir()):
        raise FileNotFoundError(f"Required 'metadata.csv' or 'wavs' directory not found in {input_dir}")

    audio_path_text_pairs = list(read_audio_text_pairs(metadata_path))
    
    polyphone = True
    total_files = len(audio_path_text_pairs)
    if total_files == 0:
        raise ValueError("No audio-text pairs found in metadata.csv.")

    worker_count = num_workers if num_workers is not None else min(MAX_WORKERS, total_files)
    print(f"\nProcessing {total_files} audio files using {worker_count} workers...")

    results = []
    executor = None
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix=THREAD_NAME_PREFIX
        ) as exec:
            executor = exec
            with graceful_exit(executor):
                # Submit all jobs
                futures = [executor.submit(process_item, pair, polyphone) for pair in audio_path_text_pairs]

                # Process results as they complete, with a progress bar
                for future in tqdm(concurrent.futures.as_completed(futures), total=total_files, desc="Processing"):
                    if shutdown_flag.is_set():
                        print("Cancellation requested, stopping job submission.")
                        break # Stop processing new results
                    
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        # This catches errors from within the future itself
                        print(f"Error processing a future: {e}")
    finally:
        if executor:
            print("Shutting down thread pool...")
            # Cancel pending futures on shutdown
            executor.shutdown(wait=False, cancel_futures=True)
        if shutdown_flag.is_set():
            print("Cleanup complete. Exiting.")
            sys.exit(1)

    if not results:
        raise RuntimeError("No valid audio files were processed!")

    # OPTIMIZATION: Data is already processed. We just need to aggregate it.
    durations = [res["duration"] for res in results]
    vocab_set = set()
    for res in results:
        vocab_set.update(list(res["text"]))

    return results, durations, vocab_set


def save_prepped_dataset(out_dir: Path, result: list, duration_list: list, text_vocab_set: set, is_finetune: bool):
    out_dir.mkdir(exist_ok=True, parents=True)
    print(f"\nSaving to {out_dir}...")

    raw_arrow_path = out_dir / "raw.arrow"
    with ArrowWriter(path=raw_arrow_path.as_posix(), writer_batch_size=200) as writer:
        for line in tqdm(result, desc="Writing to raw.arrow"):
            writer.write(line)

    dur_json_path = out_dir / "duration.json"
    with open(dur_json_path, "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    voca_out_path = out_dir / "vocab.txt"
    if is_finetune:
        shutil.copy2(PRETRAINED_VOCAB_PATH, voca_out_path)
    else:
        with open(voca_out_path, "w", encoding="utf-8") as f:
            for vocab in sorted(text_vocab_set):
                f.write(f"{vocab}\n")

    dataset_name = out_dir.stem
    print(f"\nFor {dataset_name}, sample count: {len(result)}")
    print(f"For {dataset_name}, vocab size is: {len(text_vocab_set)}")
    print(f"For {dataset_name}, total {sum(duration_list) / 3600:.2f} hours")


def prepare_and_save_set(inp_dir: str, out_dir: str, is_finetune: bool = True, num_workers: int = None):
    if is_finetune and not PRETRAINED_VOCAB_PATH.exists():
        raise FileNotFoundError(f"Finetune mode requires a pretrained vocab.txt, not found at: {PRETRAINED_VOCAB_PATH}")
    
    input_path = Path(inp_dir)
    output_path = Path(out_dir)

    sub_result, durations, vocab_set = prepare_csv_wavs_dir(input_path, num_workers=num_workers)
    save_prepped_dataset(output_path, sub_result, durations, vocab_set, is_finetune)


def main():
    if shutil.which("ffprobe") is None:
        print("Warning: ffprobe is not found in PATH. Duration extraction will rely solely on torchaudio, which may be slower.")

    parser = argparse.ArgumentParser(
        description="Prepare and save a TTS dataset from a csv/wavs structure.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # For fine-tuning (default behavior)
  python prepare_csv_wavs.py /path/to/input /path/to/output

  # For pre-training (building a new vocabulary)
  python prepare_csv_wavs.py /path/to/input /path/to/output --pretrain

  # With a specific number of worker threads
  python prepare_csv_wavs.py /path/to/input /path/to/output --workers 8
        """,
    )
    parser.add_argument("inp_dir", type=str, help="Input directory containing metadata.csv and a 'wavs' folder.")
    parser.add_argument("out_dir", type=str, help="Output directory to save the prepared data.")
    parser.add_argument("--pretrain", action="store_true", help="Enable pre-training mode (generates a new vocab). Default is fine-tune mode.")
    parser.add_argument("--workers", type=int, default=None, help=f"Number of worker threads (default: system-dependent, up to {MAX_WORKERS}).")
    args = parser.parse_args()

    try:
        prepare_and_save_set(args.inp_dir, args.out_dir, is_finetune=not args.pretrain, num_workers=args.workers)
    except (KeyboardInterrupt, SystemExit) as e:
        # Catches manual Ctrl+C and sys.exit(1) from the signal handler
        print("\nOperation cancelled by user or signal. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
