import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode
import textdistance
import numpy as np
import re

# Read the .bib file
with open('thesis.bib', 'r', encoding='utf8') as bibtex_file:
    bib_database = bibtexparser.load(bibtex_file)


# Define the array with bracketed references and their alternatives
references1 = [
    "[1] Lo EH, Dalkara T, Moskowitz MA. Mechanisms, challenges and opportunities in stroke. Nat Rev Neurosci. 2003 May;4[5]:399–414. ",
    "[2] Ebinger M, Brunecker P, Jungehülsing GJ, Malzahn U, Kunze C, Endres M, et al. Reliable Perfusion Maps in Stroke MRI Using Arterial Input Functions Derived From Distal Middle Cerebral Artery Branches. Stroke. 2010 Jan;41[1]:95–101. ",
    "[3] Laughlin B, Chan A, Tai WA, Moftakhar P. RAPID automated CT perfusion in clinical practice. Pract Neurol. 2019;2019:41–55. ",
    "[4] Demeestere J, Wouters A, Christensen S, Lemmens R, Lansberg MG. Review of Perfusion Imaging in Acute Ischemic Stroke. Stroke. 2020 Mar;51[3]:1017–24. ",
    "[5] Phipps MS, Cronin CA. Management of acute ischemic stroke. BMJ. 2020 Feb 13;368:l6983. ",
    "[6] Powers WJ. Acute Ischemic Stroke. New England Journal of Medicine. 2020 Jul 16;383[3]:252–60. ",
    "[7] Tawil SE, Muir KW. Thrombolysis and thrombectomy for acute ischaemic stroke. Clin Med (Lond). 2017 Apr;17[2]:161–5. ",
    "[8] Lin L, Bivard A, Parsons MW. Perfusion Patterns of Ischemic Stroke on Computed Tomography Perfusion. J Stroke. 2013 Sep 27;15[3]:164–73. ",
    "[9] Nielsen A, Hansen MB, Tietze A, Mouridsen K. Prediction of Tissue Outcome and Assessment of Treatment Effect in Acute Ischemic Stroke Using Deep Learning. Stroke. 2018 Jun;49[6]:1394–401. ",
    "[10] Yu Y, Xie Y, Thamm T, Gong E, Ouyang J, Huang C, et al. Use of Deep Learning to Predict Final Ischemic Stroke Lesions From Initial Magnetic Resonance Imaging. JAMA Network Open. 2020 Mar 12;3[3]:e200772. ",
    "[11] Giacalone M, Rasti P, Debs N, Frindel C, Cho TH, Grenier E, et al. Local spatio-temporal encoding of raw perfusion MRI for the prediction of final lesion in stroke. Medical Image Analysis. 2018 Dec 1;50:117–26. ",
    "[12] Pinto A, Mckinley R, Alves V, Wiest R, Silva CA, Reyes M. Stroke Lesion Outcome Prediction Based on MRI Imaging Combined With Clinical Information. Frontiers in Neurology [Internet]. 2018 [cited 2023 Feb 23];9. Available from: https://www.frontiersin.org/articles/10.3389/fneur.2018.01060",
    "[13] Robben D, Boers AMM, Marquering HA, Langezaal LLCM, Roos YBWEM, van Oostenbrugge RJ, et al. Prediction of final infarct volume from native CT perfusion and treatment parameters using deep learning. Med Image Anal. 2020 Jan;59:101589. ",
    "[14] Amador K, Wilms M, Winder A, Fiehler J, Forkert ND. Predicting treatment-specific lesion outcomes in acute ischemic stroke from 4D CT perfusion imaging using spatio-temporal convolutional neural networks. Medical Image Analysis. 2022 Nov 1;82:102610. ",
    "[15] Winder AJ, Wilms M, Amador K, Flottmann F, Fiehler J, Forkert ND. Predicting the tissue outcome of acute ischemic stroke from acute 4D computed tomography perfusion imaging using temporal features and deep learning. Front Neurosci. 2022;16:1009654. ",
    "[16] Amador K, Winder A, Fiehler J, Wilms M, Forkert ND. Hybrid Spatio-Temporal Transformer Network for Predicting Ischemic Stroke Lesion Outcomes from 4D CT Perfusion Imaging. In: Wang L, Dou Q, Fletcher PT, Speidel S, Li S, editors. Medical Image Computing and Computer Assisted Intervention – MICCAI 2022. Cham: Springer Nature Switzerland; 2022. p. 644–54. [Lecture Notes in Computer Science]. ",
    "[17] Astrup J, Siesjö BK, Symon L. Thresholds in cerebral ischemia - the ischemic penumbra. Stroke. 1981 Nov;12[6]:723–5. ",
    "[18] Lo Vercio L, Amador K, Bannister JJ, Crites S, Gutierrez A, MacDonald ME, et al. Supervised machine learning tools: A tutorial for clinicians. Journal of Neural Engineering. 2020;17[6]:L., Amador, K., Bannister, J., Crites, S., Gutierrez, A., MacDonald, M.E., Moore, J., Mouches, P., Rajasheka, D., Schimert, S., Subbanna, N., Tuladhar, A., Wang, N., Wilms, M., Winder, A., Forkert, N.D.: Supervised machine learning tools: a tutorial for clinicians. Journal of Neural Engineering 17(6), 062001. ",
    "[19] MacEachern SJ, Forkert ND. Machine learning for precision medicine. Genome. 2021;64[4]:416–25. ",
    "[20] Yedavalli VS, Tong E, Martin D, Yeom KW, Forkert ND. Artificial intelligence in stroke imaging: Current and future perspectives. Clinical Imaging. 2021 Jan 1;69:246–54. ",
    "[21] Birenbaum D, Bancroft LW, Felsberg GJ. Imaging in Acute Stroke. West J Emerg Med. 2011 Feb;12[1]:67–76. ",
    "[22] Fiehler J, Thomalla G, Bernhardt M, Kniep H, Berlis A, Dorn F, et al. ERASER: a thrombectomy study with predictive analytics end point. Stroke. 2019;50[5]:1275–8. ",
    "[23] Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis (TRIPOD): the TRIPOD statement. Ann Intern Med. 2015 Jan 6;162[1]:55–63. ",
    "[24] Forkert ND, Cheng B, Kemmling A, Thomalla G, Fiehler J. ANTONIA perfusion and stroke. Methods of information in medicine. 2014;53[06]:469–81. ",
    "[25] Winder A, d’Esterre CD, Menon BK, Fiehler J, Forkert ND. Automatic arterial input function selection in CT and MR perfusion datasets using deep convolutional neural networks. Medical Physics. 2020;47[9]:4199–211. ",
    "[26] Avants BB, Tustison NJ, Song G, Cook PA, Klein A, Gee JC. A reproducible evaluation of ANTs similarity metric performance in brain image registration. NeuroImage. 2011 Feb 1;54[3]:2033–44. ",
    "[27] Rubin J, Abulnaga SM. CT-To-MR Conditional Generative Adversarial Networks for Ischemic Stroke Lesion Segmentation. In: 2019 IEEE International Conference on Healthcare Informatics (ICHI). 2019. p. 1–7. ",
    "[28] He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. In 2016 [cited 2023 Feb 23]. p. 770–8. Available from: https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html",
    "[29] Isola P, Zhu JY, Zhou T, Efros AA. Image-To-Image Translation With Conditional Adversarial Networks. In 2017 [cited 2023 Feb 23]. p. 1125–34. Available from: https://openaccess.thecvf.com/content_cvpr_2017/html/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.html",
    "[30] Cimflova P, Ospel JM, Marko M, Menon BK, Qiu W. Variability assessment of manual segmentations of ischemic lesion volume on 24-h non-contrast CT. Neuroradiology. 2022 Jun 1;64[6]:1165–73. ",
    "[31] Winder A, Wilms M, Fiehler J, Forkert ND. Treatment Efficacy Analysis in Acute Ischemic Stroke Patients Using In Silico Modeling Based on Machine Learning: A Proof-of-Principle. Biomedicines. 2021 Oct;9[10]:1357. ",
    "[32] Midgley SM, Stella DL, Campbell BC, Langenberg F, Einsiedel PF. CT brain perfusion: A static phantom study of contrast-to-noise ratio and radiation dose. Journal of Medical Imaging and Radiation Oncology. 2017;61[3]:361–6. ",
]




# Create a list to store the matched reference IDs
matched_ids = []

# Loop over each reference for example in Vancouver
for apa_ref in references1:
    # Extract the first author from the Vancouver reference
    apa_first_author = apa_ref.split('.')[0].split(',')[0].strip()
    apa_title_first_word = apa_ref.split('.')[1].split()[0].strip()

    # Initialize variables to hold the best match
    best_match = None
    best_match_score = 0
    best_apa = None
    best_bib = None

    # Loop over each of the top 5 BibTeX entries and calculate the similarity score based on the first word of the title
    for bib_entry in bib_database.entries:
        # Extract the first word from the BibTeX entry title
        if 'title' in bib_entry:
            bib_title_first_word = bib_entry['title'].split(" ")[0]
        else:
            bib_title_first_word = " "


        # Extract the first author from the BibTeX entry
        bib_first_author = bib_entry['author'].split(' and ')[0].split(',')[0].strip()

        # Calculate the similarity score
        score = textdistance.jaccard.normalized_similarity(apa_first_author, bib_first_author)
        score = score + textdistance.jaccard.normalized_similarity(apa_title_first_word, bib_title_first_word)

        # Update the best match if this score is higher
        if score > best_match_score:
            best_match = bib_entry['ID']
            best_match_score = score
            best_apa = apa_title_first_word
            best_bib = bib_title_first_word
            

    # Add the best match ID to the list
    # print(best_apa+" = "+best_bib)
    matched_ids.append(best_match)



# Save a list containing each item in the matched ids
latex_citations = matched_ids

# Print the matched IDs
print(latex_citations)

def expand_range(expansion):
    if "-" in expansion:
        start, end = expansion.split("-")
        return range(int(start), int(end)+1)
    elif "–" in expansion:
        start, end = expansion.split("–")
        return range(int(start), int(end)+1)
    else:
        return [int(expansion)]

references2 = latex_citations

references = []
for ref1, ref2 in zip(references1, references2):
    references.append([ref1, ref2])

# Open the manuscript file and read its contents
with open("manuscript.txt", "r", encoding='utf-8') as f:
    manuscript = f.read()

# Use regular expressions to find all bracketed references in the manuscript
matches = re.findall(r'(?<=\[)[\d, - –]+(?=\])', manuscript)

# Loop through the matches and replace each one with its alternative
for match in matches:
    orig_match = match
    # Split the match into individual items
    match_items = match.split(',')
    ints_from_expansion = []
    for item in match_items:
        ints_from_expansion.extend(expand_range(item))
    items = sorted(list(set(ints_from_expansion)))
    
    # Check if there are multiple items
    if len(items) > 1:
        # Format the items as a comma-separated list

        formatted_items = ",".join([str(references2[item-1]) for item in items])
        # Replace the match with the LaTeX command \cite{id1, id2, ...}
        manuscript = manuscript.replace(f"[{orig_match}]", f"\\cite{{{formatted_items}}}")
        print(f"[{orig_match}] replaced for \\cite{{{formatted_items}}}")
    else:
        # Replace the match with the single item citation
        manuscript = manuscript.replace(f"[{orig_match}]", f"\\cite{{{references2[items[0]-1]}}}")
        print(f"[{orig_match}] replaced for \\cite{{{references2[items[0]-1]}}}")

# Save the modified manuscript as a new file
with open("modified_manuscript.txt", "w", encoding='utf-8') as f:
    f.write(manuscript)