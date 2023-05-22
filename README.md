# Table-OCR-Extractor
Project for extracting tables in Images or PDFs.

This project allows you to extract tables from images or PDFs and perform your desired operations on them.
To use this code; simply add it to your project and call the basic functions as shown in <a href="./Example.pdf">Example.pdf</a>

All essential functions have been documented, therefore you can see more about how to use them using Python's help function.

#### Note:
To figure out which text is in a given cell (row, column). You can use the `table.as_json()` function and then read from it the necessary row and column positions (see PDF above).
<br>OR<br>
For a more concrete view, you could look into `table.celltext_positions` and `table.cell_textdata` variables.

# Dependencies
- pytesseract
- OpenCV
- Numpy
- pymupdf

Report Bugs to: <a href="josephmulindwa490@gmail.com">josephmulindwa490@gmail.com</a>
