# -*- coding: utf-8 -*-
import argparse
import sys

from audiblez.voices import available_voices_str, voices


def cli_main():
    voices_str = ", ".join(voice for voice_list in voices.values() for voice in voice_list)
    epilog = (
        "examples:\n"
        "  audiblez book.epub -v af_sky\n"
        "  audiblez book.epub -v af_sky -s 1.5 --output ./audiobooks\n\n"
        "to run the GUI:\n"
        "  audiblez-ui\n\n"
        "available voices:\n"
        f"{available_voices_str}"
    )
    parser = argparse.ArgumentParser(
        description="Generate an M4B audiobook from an EPUB e-book.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("epub_file_path", help="Path to the EPUB file")
    parser.add_argument("-v", "--voice", default="af_sky", help=f"Choose narrating voice. Available: {voices_str}")
    parser.add_argument("-p", "--pick", action="store_true", help="Interactively select chapters to read")
    parser.add_argument("-s", "--speed", default=1.0, type=float, help="Set speed from 0.5 to 2.0")
    parser.add_argument("-c", "--cuda", action="store_true", help="Use CUDA in Torch if available")
    parser.add_argument("-o", "--output", default=".", metavar="FOLDER", help="Output folder for temporary and final files")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.cuda:
        import torch.cuda

        if torch.cuda.is_available():
            print("CUDA GPU available")
            torch.set_default_device("cuda")
        else:
            print("CUDA GPU not available. Defaulting to CPU.")

    from audiblez.core import main

    main(
        args.epub_file_path,
        voice=args.voice,
        pick_manually=args.pick,
        speed=args.speed,
        output_folder=args.output,
    )


if __name__ == "__main__":
    cli_main()
