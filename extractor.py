import fitz
import re


def extract_project_data(path, chapter):
    doc = fitz.open(path)
    toc = doc.get_toc()
    search = []
    for i, item in enumerate(toc):
        if chapter in re.sub(r"\s+", "", item[1].lower(), flags=re.UNICODE):
            search.append(item)
            if i < len(toc)-1:
                search.append(toc[i+1])
            else:
                search.append([search[0][0], "", doc.pageCount-1])
                
    start_page = search[0][2]-2
    end_page = search[1][2]
    if search[1][1] != "":
        end_page += 2        
    for page_num in range(start_page, end_page+1):
        page = doc[page_num]
        if page.search_for(search[0][1])!=[]:
            search[0][2] = page_num
        if page.search_for(search[1][1])!=[]:
            search[1][2] = page_num

    text = ""
    for page_num in range(search[0][2], search[1][2]+1):
        page = doc[page_num]
        page_text = page.get_text()
        text += page_text + " "

    text = text.split(search[0][1])[1].split(search[1][1])[0]
    text = text.replace("\n", "")
    text = re.sub(r"[^a-zA-Z0-9.,;:\- ]", "", text)
    text = re.sub("\\s+", " ", text)
    return text