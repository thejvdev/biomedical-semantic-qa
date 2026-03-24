from pathlib import Path
import lxml.etree as ET


def parse_document(file_path: str | Path) -> list[dict]:
    articles_data = []
    context = ET.iterparse(file_path, events=("end",), tag="PubmedArticle")

    for _, elem in context:
        pmid_elem = elem.find(".//PMID")
        pmid = pmid_elem.text if pmid_elem is not None else None

        title_elem = elem.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else ""

        abstract_elems = elem.findall(".//AbstractText")
        if abstract_elems:
            abstract = " ".join([a.text.strip() for a in abstract_elems if a.text])
        else:
            abstract = ""

        chemical_elems = elem.findall(".//ChemicalList/Chemical/NameOfSubstance")
        chemicals = [c.text for c in chemical_elems if c.text]

        mesh_terms = []

        mesh_headings = elem.findall(".//MeshHeadingList/MeshHeading")
        for heading in mesh_headings:
            descriptor_elem = heading.find("DescriptorName")

            if descriptor_elem is not None and descriptor_elem.text:
                descriptor_text = descriptor_elem.text
                mesh_terms.append(descriptor_text)

                qualifier_elems = heading.findall("QualifierName")
                for qualifier in qualifier_elems:
                    if qualifier.text:
                        mesh_terms.append(f"{descriptor_text}/{qualifier.text}")

        if pmid:
            articles_data.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "chemicals": chemicals,
                    "mesh_terms": mesh_terms,
                }
            )

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    del context

    return articles_data


def flatten_article(article: dict) -> str:
    title = article.get("title", "")
    abstract = article.get("abstract", "")
    chemicals = ", ".join(article.get("chemicals", []))
    mesh = ", ".join(article.get("mesh_terms", []))
    return f"{title}\n{abstract}\nKeywords: {chemicals}\nMeSH: {mesh}"
