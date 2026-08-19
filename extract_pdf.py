import PyPDF2
pdf = PyPDF2.PdfReader('MedShieldFL_100_Tasks_Build_Guide.pdf')
with open('tasks.txt', 'w', encoding='utf-8') as f:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            f.write(text)
            f.write('\n')
