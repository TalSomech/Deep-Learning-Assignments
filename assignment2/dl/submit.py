import os
import zipfile


# Files to include in code submission
_CODE_FILES = [
    "transformers.py",
    "rnn_lstm_captioning.py",
]

_NOTEBOOK_FILES = [
    "Transformers.ipynb",
    "rnn_lstm_captioning.ipynb",
    "Self_Supervised_Learning.ipynb",
    "CLIP_DINO.ipynb",
]

# Small submission files needed for grading
_SUBMISSION_FILES = [
    "rnn_lstm_attention_submission.pt",
]

# Files/folders to EXCLUDE (large files that bloat submission)
_EXCLUDE_PATTERNS = [
    "pretrained_model",
    ".pth",
    ".mp4",
    "__pycache__",
    ".pyc",
    "data/",
    "datasets/",
]


def make_code_submission(assignment_path):
    """Create Assignment 2 code submission zip file."""
    zip_path = os.path.join(assignment_path, "a2_code_submission.zip")
    print("Creating code submission zip...")
    print(f"Writing to: {zip_path}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add Python files
        for filename in _CODE_FILES:
            in_path = os.path.join(assignment_path, filename)
            if os.path.isfile(in_path):
                zf.write(in_path, filename)
                print(f"  Added: {filename}")
            else:
                print(f"  Warning: Could not find {filename}")

        # Add notebook files
        for filename in _NOTEBOOK_FILES:
            in_path = os.path.join(assignment_path, filename)
            if os.path.isfile(in_path):
                zf.write(in_path, filename)
                print(f"  Added: {filename}")
            else:
                print(f"  Warning: Could not find {filename}")

        # Add small submission files (for grading)
        for filename in _SUBMISSION_FILES:
            in_path = os.path.join(assignment_path, filename)
            if os.path.isfile(in_path):
                # Check file size (should be < 10MB)
                size_mb = os.path.getsize(in_path) / (1024 * 1024)
                if size_mb < 10:
                    zf.write(in_path, filename)
                    print(f"  Added: {filename} ({size_mb:.2f} MB)")
                else:
                    print(f"  Skipped: {filename} (too large: {size_mb:.2f} MB)")
            else:
                print(f"  Note: {filename} not found (will be created when you run the notebook)")

    # Print final size
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nCode submission created: {zip_path}")
    print(f"Size: {zip_size_mb:.2f} MB")

    if zip_size_mb > 50:
        print("\n⚠️  WARNING: Submission is larger than 50MB!")
        print("    This may cause upload issues. Check for large files.")

    return zip_path


def make_inline_pdf(assignment_path):
    """Create inline PDF from all notebooks."""
    import subprocess

    pdf_path = os.path.join(assignment_path, "a2_inline_submission.pdf")
    print("\nCreating inline PDF submission...")
    print("(This may take a few minutes...)")

    # Convert each notebook to PDF
    pdf_files = []
    for notebook in _NOTEBOOK_FILES:
        nb_path = os.path.join(assignment_path, notebook)
        if os.path.isfile(nb_path):
            print(f"  Converting: {notebook}")
            try:
                # Convert notebook to PDF using nbconvert
                result = subprocess.run(
                    ["jupyter", "nbconvert", "--to", "pdf", nb_path,
                     "--output-dir", assignment_path],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per notebook
                )
                pdf_name = notebook.replace(".ipynb", ".pdf")
                pdf_file = os.path.join(assignment_path, pdf_name)
                if os.path.isfile(pdf_file):
                    pdf_files.append(pdf_file)
                    print(f"    ✓ Created: {pdf_name}")
                else:
                    print(f"    ✗ Failed to create {pdf_name}")
                    if result.stderr:
                        print(f"      Error: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                print(f"    ✗ Timeout converting {notebook}")
            except Exception as e:
                print(f"    ✗ Error: {e}")

    if pdf_files:
        # Try to merge PDFs
        try:
            from PyPDF2 import PdfMerger
            print("\nMerging PDFs...")
            merger = PdfMerger()
            for pdf in pdf_files:
                merger.append(pdf)
            merger.write(pdf_path)
            merger.close()

            # Clean up individual PDFs
            for pdf in pdf_files:
                os.remove(pdf)

            pdf_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            print(f"\n✓ Inline PDF created: {pdf_path}")
            print(f"  Size: {pdf_size_mb:.2f} MB")

        except ImportError:
            print("\n⚠️  PyPDF2 not installed. Individual PDFs created:")
            for pdf in pdf_files:
                print(f"  {os.path.basename(pdf)}")
            print("\nTo merge them, install PyPDF2: !pip install PyPDF2")
            print("Or merge manually using a PDF tool.")
    else:
        print("\n⚠️  No PDFs were created.")
        print("You may need to install LaTeX: !apt-get install texlive-xetex texlive-fonts-recommended")
        print("\nAlternative: Use File > Print > Save as PDF in Colab for each notebook,")
        print("then merge the PDFs manually.")

    return pdf_path


def make_assignment2_submission(assignment_path):
    """Create Assignment 2 submission files (code zip and inline PDF)."""
    print("=" * 60)
    print("       Assignment 2 Submission Generator")
    print("=" * 60)
    print()
    print("NOTE: Do NOT include these large files in your submission:")
    print("  - pretrained_model/*.pth (SimCLR weights)")
    print("  - dino_res.mp4 (DINO video)")
    print("  - data/ folder (datasets)")
    print()

    # Create code submission
    code_zip = make_code_submission(assignment_path)

    # Create inline PDF
    try:
        pdf_path = make_inline_pdf(assignment_path)
    except Exception as e:
        print(f"\n⚠️  Could not create inline PDF automatically: {e}")
        print("\nPlease create the PDF manually:")
        print("  1. Open each notebook in Colab")
        print("  2. File > Print > Save as PDF")
        print("  3. Merge the PDFs using a PDF tool")
        pdf_path = None

    print("\n" + "=" * 60)
    print("                  Submission Summary")
    print("=" * 60)
    print("\nPlease download and submit to Lemida:")
    print("  1. a2_code_submission.zip")
    print("  2. a2_inline_submission.pdf")
    print("  3. students.txt (create manually with your names and IDs)")
    print()
    print("Don't forget to create students.txt!")
    print("Format:")
    print("  FirstName_LastName StudentID")
    print("  FirstName_LastName StudentID")
    print("=" * 60)

