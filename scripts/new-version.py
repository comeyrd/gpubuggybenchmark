import os
import argparse
import sys

PROJECT_DIR = "../src/"


def copy_and_replace_bug(bigram: str, template_dir, output_dir,file_extension,kernel,description=""):
    bigram_upper = bigram.upper()
    bigram_lower = bigram.lower()
    template_file = f"{template_dir}/{kernel}-reference{file_extension}"
    if not os.path.exists(template_file):
        print(f"Error: Template file '{template_file}' does not exist.")
        return

    with open(template_file, 'r') as f:
        content = f.read()

    # Replace identifiers
    content = content.replace("REFERENCE", bigram_upper)
    content = content.replace("reference", bigram_lower)
    content = content.replace("Reference", bigram_upper)
    if file_extension == ".cu":
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if '#include "cuda-utils.hpp"' in line:
                lines.insert(i + 1, "//"+description)
                break
        content = "\n".join(lines)
    elif file_extension == ".hpp":
        os.path.join(output_dir,"include")
    else:
        print(f"Error: Unsupported file extension '{file_extension}'.")
        return

    # Save new file
    new_filename = os.path.join(output_dir, f"{kernel}-{bigram_lower}{file_extension}")
    with open(new_filename, 'w') as f:
        f.write(content)

    print(f"Created: {new_filename}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process bigram argument")
    parser.add_argument("kernel",type=str, help="The kernel on which you want to add a version")
    parser.add_argument("bigram", type=str, help="A 2-character bigram identifier")
    parser.add_argument("desc", type=str, nargs='*', default=[], help="The description of the version (optional)")

    args = parser.parse_args()

    bigram = args.bigram
    if len(bigram) != 2:
        print("Error: Bigram must be exactly 2 characters long.")
        sys.exit(1)
    description = " ".join(args.desc) if args.desc else ""

    if not os.path.exists(os.path.join(PROJECT_DIR,args.kernel)):
        print(f"Error: Kernel {args.kernel} does not exist ")
        sys.exit(1)

    template_dir = os.path.join(PROJECT_DIR,args.kernel)
    output_dir = os.path.join(PROJECT_DIR,args.kernel,"flawed")
    print(output_dir)
    copy_and_replace_bug(bigram,template_dir,output_dir,".hpp",args.kernel)
    copy_and_replace_bug(bigram,template_dir,output_dir,".cu",args.kernel,description)

