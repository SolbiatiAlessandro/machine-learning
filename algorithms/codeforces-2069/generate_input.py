#!/usr/bin/env python3

def main():
    input_filename = "Blong.in"
    output_filename = "Blonglong.in"

    # Read the original lines (stripping newlines)
    with open(input_filename, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    if not lines:
        print("The input file is empty!")
        return

    total_original = len(lines)

    # Calculate how many full copies and extra lines we need
    copies = 333 // total_original
    remainder = 333 % total_original

    # Construct the new list of lines
    new_lines = lines * copies + lines[:remainder]

    # Verify we have exactly 333 lines
    assert len(new_lines) == 333, f"Expected 333 lines, but got {len(new_lines)}"

    # Write the new content to the output file
    with open(output_filename, "w") as f:
        f.write("\n".join(new_lines))

    print(f"Output file '{output_filename}' created with 333 lines.")

if __name__ == '__main__':
    main()

