from lxml import etree
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Parse XML file
xml_file = "datasets\Textbooks\Advanced Macroeconomics - Romer_abbyy.xml"
tree = etree.parse(xml_file)
root = tree.getroot()

# Create a PDF
pdf_file = "datasets\Textbooks\Advanced Macroeconomics - Romer_abbyy.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
width, height = letter

y_position = height - 50  # Start position for text

# Extract and write XML content to PDF
for element in root.iter():
    text = f"{element.tag}: {element.text}"
    c.drawString(50, y_position, text)
    y_position -= 20  # Move down for the next line

    if y_position < 50:  # Start a new page if space is running out
        c.showPage()
        y_position = height - 50

c.save()
print(f"PDF saved as {pdf_file}")
