def replace_first_occurrence_and_sort(input_file, output_file, x, y):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        modified_line = line.replace(x, y, 1)
        modified_lines.append(modified_line.strip())

    # Split the lines into columns, assuming space-separated text
    split_lines = [line.split() for line in modified_lines]

    # Sort the lines based on the frame number (first column, as integer)
    sorted_lines = sorted(split_lines, key=lambda line: int(line[0]))

    # Join the columns back into lines and write the sorted lines to the output file
    with open(output_file, 'w') as file:
        for line in sorted_lines:
            sorted_line = ' '.join(line)
            file.write(sorted_line + '\n')

if __name__ == "__main__":
    M="D1"
    x = "D3_b_"
    y = ""

    # input_file = f"runs/summaries/{M}/v5ssummary.txt"
    # output_file = f"runs/summaries/{M}/v5ssummary_fixed.txt"
    input_file = f"runs/summaries/80vpc/background_summary.txt"
    output_file = f"runs/summaries/D3/background_summary_fixed.txt"
    replace_first_occurrence_and_sort(input_file, output_file, x, y)




