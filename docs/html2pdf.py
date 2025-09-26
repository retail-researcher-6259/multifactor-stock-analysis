import pdfkit

# Windows example - adjust path as needed
config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')

# Enhanced options for professional document layout
options = {
    'print-media-type': '',  # Ensures CSS print rules are applied
    'no-outline': None,
    'page-size': 'A4',
    'margin-top': '0.75in',
    'margin-right': '0.75in',
    'margin-bottom': '0.75in',
    'margin-left': '0.75in',
    'encoding': "UTF-8",
    'enable-local-file-access': None,  # Allows local CSS and images
}

# Convert HTML file to PDF with enhanced configuration
pdfkit.from_file('msas_complete_documentation_pdf.html', 'MSAS_complete_documentation.pdf',
                 configuration=config, options=options)
