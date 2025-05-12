from suta import *
from data import *

def split_file_at_line(input_file_path, output_file1, output_file2, split_line=7118):
    """
    Split a text file into two separate files at a specified line number.
    
    Args:
        input_file_path (str): Path to the input text file
        output_file1 (str): Path for the first output file (lines 1 to split_line-1)
        output_file2 (str): Path for the second output file (lines split_line to end)
        split_line (int): Line number to split at (default: 7118)
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            # Read all lines from the file
            all_lines = input_file.readlines()
            
            # Split the lines into two parts
            first_part = all_lines[:split_line-1]
            second_part = all_lines[split_line-1:]
            
            # Write the first part to the first output file
            with open(output_file1, 'w', encoding='utf-8') as out1:
                out1.writelines(first_part)
            
            # Write the second part to the second output file
            with open(output_file2, 'w', encoding='utf-8') as out2:
                out2.writelines(second_part)
                
        print(f"File split successfully:")
        print(f"- Lines 1-{split_line-1} written to {output_file1}")
        print(f"- Lines {split_line}-{len(all_lines)} written to {output_file2}")
        
    except FileNotFoundError:
        print(f"Error: File '{input_file_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    wer_processor = transcriptionProcessor(task='transcribe')
    wer_processor.process_file(f'encoder_choose_lr10/result.txt')
    wer_list = wer_processor.step_mean_wer()
    
    print(wer_list)
    # split_file_at_line(input_file, output_file1, output_file2)