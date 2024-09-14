

# feed list of test_cases/input/*.txt into main program,
# and compare(diff) the output with corresponding test_cases/output/*.txt

import os
import subprocess
import difflib

# https://stackoverflow.com/questions/32500167/how-to-show-diff-of-two-string-sequences-in-colors
red = lambda text: f"\033[38;2;255;0;0m{text}\033[38;2;255;255;255m"
green = lambda text: f"\033[38;2;0;255;0m{text}\033[38;2;255;255;255m"
blue = lambda text: f"\033[38;2;0;0;255m{text}\033[38;2;255;255;255m"
white = lambda text: f"\033[38;2;255;255;255m{text}\033[38;2;255;255;255m"

def diff_string(old, new):
    result = ""
    codes = difflib.SequenceMatcher(a=old, b=new).get_opcodes()
    for code in codes:
        if code[0] == "equal": 
            result += white(old[code[1]:code[2]])
        elif code[0] == "delete":
            result += red(old[code[1]:code[2]])
        elif code[0] == "insert":
            result += green(new[code[3]:code[4]])
        elif code[0] == "replace":
            result += (red(old[code[1]:code[2]]) + green(new[code[3]:code[4]]))
    return result

def main():
    # get the list of input files
    filenames = sorted([f for f in os.listdir('test_cases/input') if f.endswith('.txt')])

    # run the main program for each input file
    for filename in filenames:
        print(f'Running test for {filename}...')
        input_file = os.path.join('test_cases/input', filename)
        output_file = os.path.join('test_cases/output', filename)

        # run the main program
        command = "cargo run < " + input_file
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _stderr = process.communicate()
        stdout = stdout.decode('utf-8').strip()

        # read the expected output
        with open(output_file, 'r') as f:
            expected_output = f.read().strip()

        # compare the output
        if stdout == expected_output:
            print('Test passed')
        else:
            print('Test failed')
            print(diff_string(stdout, expected_output))
            print()



if __name__ == '__main__':
    main()