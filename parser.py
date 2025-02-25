import os
from pyhtml2pdf import converter
from PyPDF2 import PdfMerger

# Define the directory containing the HTML files
input_dir = '/Users/abhishekshah/Desktop/financial_doc_analyser/AAPL'
output_dir = '/Users/abhishekshah/Desktop/financial_doc_analyser/AAPL'

# Define the order in which the files should be processed
file_order = ['10-K', '10-Q', '8-K']

# Function to categorize files based on their type
def categorize_files(files):
    categorized = {'10-K': [], '10-Q': [], '8-K': []}
    for file in files:
        if '10k' in file.lower() or '10-k' in file.lower():
            categorized['10-K'].append(file)
        elif '10q' in file.lower() or '10-q' in file.lower():
            categorized['10-Q'].append(file)
        elif '8k' in file.lower() or '8-k' in file.lower():
            categorized['8-K'].append(file)
    return categorized

# Get all HTML files in the input directory
html_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.htm') or f.endswith('.html')]

# Categorize the files
categorized_files = categorize_files(html_files)

# Temporary PDF files for merging
temp_pdf_files = []

# Process files in the specified order
for file_type in file_order:
    if categorized_files[file_type]:
        print(f"Processing {file_type} files...")
        for idx, file_path in enumerate(categorized_files[file_type]):
            output_pdf = os.path.join(output_dir, f'temp_{file_type}_{idx}.pdf')
            try:
                # Convert HTML to PDF using pyhtml2pdf
                converter.convert(file_path, output_pdf)
                temp_pdf_files.append(output_pdf)
                print(f"Converted {file_path} to {output_pdf}")
            except Exception as e:
                print(f"Error converting {file_path}: {str(e)}")
    else:
        print(f"No {file_type} files found.")

# Merge all temporary PDFs into a single PDF
if temp_pdf_files:
    merger = PdfMerger()
    for pdf in temp_pdf_files:
        merger.append(pdf)
    
    final_pdf_path = os.path.join(output_dir, 'combined_financial_reports.pdf')
    merger.write(final_pdf_path)
    merger.close()
    
    # Clean up temporary files
    for pdf in temp_pdf_files:
        try:
            os.remove(pdf)
        except Exception as e:
            print(f"Warning: Could not remove temporary file {pdf} - {str(e)}")
    
    print(f"\nSuccessfully created combined PDF at:\n{final_pdf_path}")
else:
    print("No PDF files were created to combine.")

print("\nProcess completed.")