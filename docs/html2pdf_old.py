import pdfkit

# Windows example - adjust path as needed
config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')

# Convert HTML file to PDF with configuration
pdfkit.from_file('msas_complete_documentation.html', 'MSAS_complete_documentation.pdf', configuration=config)
