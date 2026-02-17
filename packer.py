import os


def consolidate_py_files(output_filename="consolidated_code.txt"):
    # Get the name of the current script to avoid self-inclusion
    current_script = os.path.basename(__file__)

    # Open the output file in write mode
    with open(output_filename, "w", encoding="utf-8") as outfile:
        # Walk through the current directory and all subdirectories
        for root, dirs, files in os.walk("."):
            for file in files:
                # Check for .py extension and ensure it's not this script
                if file.endswith(".py") and file != current_script:
                    file_path = os.path.join(root, file)

                    # Add a header for clarity in the text file
                    outfile.write(f"\n{'=' * 50}\n")
                    outfile.write(f"FILE: {file_path}\n")
                    outfile.write(f"{'=' * 50}\n\n")

                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            outfile.write(infile.read())
                            outfile.write("\n")
                    except Exception as e:
                        outfile.write(f"[Error reading file {file}: {e}]\n")

    print(f"Success! All Python files have been packed into {output_filename}")


if __name__ == "__main__":
    consolidate_py_files()