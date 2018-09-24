# UW Coursework
This repository contains personal solutions to homeworks and exams for courses taken at the University of Washington. Solutions are provided with no guarantees to their accuracy. 

Any time MATLAB code was provided for an assignment, I ported it to Python (Numpy/Scipy). This may be of particular use for future students in courses where MATLAB is used frequently.

As I upload files I am going through them to standardize formatting and correct errors which I made when submitting my assignments. In general I am relying on grader feedback for this, so there are likely additional errors. Feel free to mark these as issues in Github and they will be corrected.

In order to compile the LaTeX files you will need to download the `TeX_headers` folder. I use XeLaTeX to generate the PDF files, but pdflatex may work.

### Repository Contents
Course | Date Taken | Uploaded | Corrected 
-|-|-|-
AMATH 514 | Winter 2018 | yes | no
AMATH 561 | Autumn 2017 | yes | yes (except ch1/2) 
AMATH 562 | Winter 2018 | yes | yes (except ch6,ch10)
AMATH 563 | Spring 2019 | yes | no<sup>1</sup>
AMATH 567 | Autumn 2017 | |
AMATH 584 | Autumn 2017 | yes | yes
AMATH 585 | Winter 2018 | yes | yes
AMATH 586 | Spring 2018 | yes (except midterm) | yes
 
1. I have no intention of updating the reports for this calss. However, my python examples and code are in working condition. Note that my tensorflow was a GPU version, so its possible that some stuff may not work properly. However, most of the code is just numpy/scipy. 

### Note on Academic Integrity
Copying solutions from any source is not only detrimental to your own learning but is plagiarism. Don't do this. If you do end up looking up a solution for an assignment, cite your source.

I have provided these questions and solutions so in the hopes that they can be used for reference when preparing for exams, as practice problems/additional problems for self study, and so that the LaTeX sources are available.

If you would like the problem statements without the solutions, an easy way to do this would be to redefine the `solution` environment and then recompile the document. Alternatively you could use regex or context free grammars to delete all the `solution` environments.
