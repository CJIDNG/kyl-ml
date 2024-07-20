from bs4 import BeautifulSoup

# Function to remove HTML tags
def remove_html_tags(input_file, output_file):
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Parse the HTML content
    soup = BeautifulSoup(content, 'html.parser')
    
    # Get text without HTML tags
    text = soup.get_text(separator="\n")
    
    # Write the text to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)

# Specify input and output file paths
input_file = './text.txt'
output_file = './text_clean.txt'

# Call the function
remove_html_tags(input_file, output_file)
