import os
import argparse
import sys

PROJECT_DIR = "../src/"
FLAWED_DIR = "flawed"
ALL_BUGS = {
    "MC":"To much data movement between the host and the device",
    "MZ":"Having an array with fixed size, but dynamic input size",
    "ML":"Memory Leak",
    "CS":"Too much synchronisation between host and device",
    "DG":"Data-locality - Using global for shared memory",
    "DR":"Data-locality - Using register for shared or global memory",
    "DS":"Dala-locality - Using shared memory for register memory",
    "DS":"Dala-layout - Suboptimal layout, having ** instead of *",
    "RC":"Loss of information - Race condition",
    "RO":"Redundant Operation - Doing an operation in a loop that is invariant to the iteration",
    "UO":"Unnecessary Operation - Computation that is not used in the algorithm",
    "UD":"Unnecessary Operation - data access that is not used in the algorithm",
    "IL":"Inneficient data structure or computation library",
    "CF":"Inneficient Cache access -  False sharing (sharing cache line)",
    "CL":"Inneficient Cache access - Non linear data access",
    "BC":"Not Enqueing - Using blocking calls for memory copy or kernel excecution",
    "LU":"Suboptimal code generation by the compiler - Missed Loop Unrolling",
    "LU":"Suboptimal code generation by the compiler - Missed Function inlining",
    "PS":"Missing Paralelism - missing SIMD parallelism",
    "PG":"Missing Paralelism - missing GPU parallelism",
    "PT":"Missing Paralelism - missing TASK parallelism",
    "PW":"Inefficient parallelization - Inefficient work partitionning",
    "MA":"Memory Overhead - Repeated calls for allocation instead of one big allocation",
    "MO":"Memory Overhead - Redundant memory operation",
}

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
                lines.insert(i + 2,"//TODO")
                break
        content = "\n".join(lines)
    elif file_extension == ".hpp":
        output_dir = os.path.join(output_dir,"include")
    else:
        print(f"Error: Unsupported file extension '{file_extension}'.")
        return

    # Save new file
    os.makedirs(output_dir,exist_ok=True)
    new_filename = os.path.join(output_dir, f"{kernel}-{bigram_lower}{file_extension}")
    if not os.path.exists(new_filename):
        with open(new_filename, 'w') as f:
            f.write(content)
        print(f"Created: {new_filename}")
    else :
        print(f"{new_filename} already exists, skipping...")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process bigram argument")

    parser.add_argument("kernel", type=str, help="The kernel on which you want to add a version")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-a", "--all", action="store_true", help="Generate all versions for this kernel")
    group.add_argument("-b","--birgram", type=str, help="A 2-character bigram identifier")
    parser.add_argument("desc", type=str, nargs="*", default=[], help="The description of the version (optional)")

    args = parser.parse_args()

    # Manual logic to enforce mutual exclusivity
    if args.all and args.desc:
        parser.error("You can't specify both --all and bigram arguments.")

    if not os.path.exists(os.path.join(PROJECT_DIR,args.kernel)):
        print(f"Error: Kernel {args.kernel} does not exist ")
        sys.exit(1)

    template_dir = os.path.join(PROJECT_DIR,args.kernel)
    output_dir = os.path.join(PROJECT_DIR,args.kernel,FLAWED_DIR)
    os.makedirs(output_dir,exist_ok=True)
    if args.all:
        for bigram,desc in ALL_BUGS.items():
            copy_and_replace_bug(bigram,template_dir,output_dir,".hpp",args.kernel)
            copy_and_replace_bug(bigram,template_dir,output_dir,".cu",args.kernel,desc)
    else :     
        if len(args.bigram) != 2:
            print("Error: Bigram must be exactly 2 characters long.")
            sys.exit(1)
        description = " ".join(args.desc) if args.desc else ""
        copy_and_replace_bug(args.bigram,template_dir,output_dir,".hpp",args.kernel)
        copy_and_replace_bug(args.bigram,template_dir,output_dir,".cu",args.kernel,description)

