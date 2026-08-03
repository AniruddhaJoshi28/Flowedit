"""
FlowEdit Multi-Category Pronunciation Adaptation Benchmark Dataset.

Contains 312 target words/phrases across 11 distinct linguistic and domain categories:
1. English Proper Nouns
2. Hindi Words
3. Japanese Terms
4. Chinese Names
5. German Compound Words
6. French Names
7. Greek Words
8. Medical Terminology
9. Brand Names
10. Scientific Terms
11. Acronyms

Reference: FlowEdit evaluation framework for lifelong pronunciation learning.
"""

from typing import List, Dict, Any


BENCHMARK_CATEGORIES: Dict[str, List[str]] = {
    "English Proper Nouns": [
        "Mrunmayee", "Worcestershire", "Cholmondeley", "Featherstonehaugh", "Leicester",
        "Schenectady", "Poughkeepsie", "Gloucestershire", "Tchoupitoulas", "Kalamazoo",
        "Tucson", "Broughton", "Des Moines", "Spokane", "Albuquerque", "Mackinac",
        "Willamette", "Schooner", "Ypsilanti", "Sequim", "Cuyahoga", "Coeur d'Alene",
        "Puyallup", "Schuylkill", "Oaxaca", "Kissimmee", "Natchitoches", "Waukesha", "Ketchikan"
    ],
    "Hindi Words": [
        "Shripad", "Dnyaneshwar", "Aniruddha", "Kalyanasundaram", "Tryambak",
        "Venkatanathan", "Subramaniam", "Mahabaleshwar", "Purushottam", "Harishchandra",
        "Vishakhapatnam", "Thiruvananthapuram", "Rameswaram", "Sriperumbudur", "Bhubaneswar",
        "Venkateshwara", "Chandrashekar", "Satyanarayana", "Krishnamurthy", "Narasimhan",
        "Ranganathan", "Swaminathan", "Ramachandran", "Parameswaran", "Viswanathan", "Kothandaraman", "Soundararajan", "Varadarajan"
    ],
    "Japanese Terms": [
        "Shibuya", "Akihabara", "Shinjuku", "Tsukiji", "Roppongi",
        "Kyoto", "Hiroshima", "Kagoshima", "Fukuoka", "Kamakkura",
        "Takayama", "Kanazawa", "Nagano", "Shirakawa-go", "Miyajima",
        "Hakone", "Nikko", "Kamakura", "Nara", "Matsuyama",
        "Okinawa", "Sapporo", "Otaru", "Asahikawa", "Hakodate", "Furano", "Noboribetsu", "Kushiro"
    ],
    "Chinese Names": [
        "Shenzhen", "Guangzhou", "Chengdu", "Chongqing", "Hangzhou",
        "Wuhan", "Nanjing", "Xi'an", "Tianjin", "Suzhou",
        "Zhengzhou", "Changsha", "Dongguan", "Foshan", "Ningbo",
        "Qingdao", "Wuxi", "Hefei", "Dalian", "Xiamen",
        "Harbin", "Jinan", "Shenyang", "Changchun", "Shijiazhuang", "Nanning", "Fuzhou", "Kunming"
    ],
    "German Compound Words": [
        "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz",
        "Kraftfahrzeug-Haftpflichtversicherung",
        "Donaudampfschiffahrtselektrizitätenhauptbetriebswerkbauunterbeamtengesellschaft",
        "Rechtsschutzversicherungsgesellschaften",
        "Nahrungsmittelunverträglichkeit",
        "Arbeitsunfähigkeitsbescheinigung",
        "Schadenersatzanspruch",
        "Vermögenssteuerreform",
        "Unabhängigkeitserklärung",
        "Geschwindigkeitsbegrenzung",
        "Steuererklärung Formular",
        "Wohnungsbauprämie",
        "Krankenversicherungskarte",
        "Betriebsratsvorsitzender",
        "Bundestagsabgeordneter",
        "Schulbescheinigung",
        "Fahrkartenschalter",
        "Eisenbahnkreuzung",
        "Luftfahrtbundesamt",
        "Bundesverfassungsgericht", "Hochschulabschluss", "Universitätsbibliothek", "Weltgesundheitsorganisation", "Kraftfahrzeugschein", "Standardabweichung", "Wirtschaftswachstum", "Zahlungsbedingungen", "Zulassungsbescheinigung"
    ],
    "French Names": [
        "Aix-en-Provence", "Montpellier", "Strasbourg", "Bordeaux", "Toulouse",
        "Grenoble", "Clermont-Ferrand", "Saint-Étienne", "Villeurbanne", "Besançon",
        "Perpignan", "Brest", "Limoges", "Amiens", "Annecy",
        "Boulogne-Billancourt", "Metz", "Rouen", "Caen", "Mulhouse",
        "Nancy", "Saint-Denis", "Argenteuil", "Montreuil", "Roubaix", "Tourcoing", "Nanterre", "Avignon"
    ],
    "Greek Words": [
        "Thessaloniki", "Alexandroupoli", "Heraklion", "Patras", "Larissa",
        "Volos", "Ioannina", "Trikala", "Chalcis", "Serres",
        "Kalamata", "Kavala", "Rhodes", "Chania", "Agrinio",
        "Katerini", "Trikala", "Lamia", "Komotini", "Kozani",
        "Veria", "Rethymno", "Ptolemaida", "Mytilene", "Corfu", "Tripoli", "Elefsina", "Megara"
    ],
    "Medical Terminology": [
        "Otorhinolaryngology", "Gastroenterology", "Encephalopathy", "Atherosclerosis",
        "Sphygmomanometer", "Electroencephalogram", "Thrombocytopenia", "Hypercholesterolemia",
        "Pneumonoultramicroscopicsilicovolcanoconiosis", "Cholecystectomy", "Onychocryptosis",
        "Pseudohypoparathyroidism", "Bronchopulmonary", "Glomerulonephritis", "Laryngotracheobronchitis",
        "Sternocleidomastoid", "Hepatosplenomegaly", "Anaphylaxis", "Cardiomyopathy", "Osteoarthritis", "Fibromyalgia", "Myocardial", "Hypothyroidism", "Endometriosis", "Scoliosis", "Parkinsonism", "Psoriasis", "Leukemia"
    ],
    "Brand Names": [
        "Huawei", "Xiaomi", "Versace", "Givenchy", "Bvlgari",
        "Hermès", "Porsche", "Volkswagen", "Peugeot", "Renault",
        "Citroën", "Hyundai", "Moët & Chandon", "Tag Heuer", "Audemars Piguet",
        "Hublot", "Patek Philippe", "Jaeger-LeCoultre", "Breguet", "Vacheron Constantin",
        "Yves Saint Laurent", "Balenciaga", "Miu Miu", "Bottega Veneta", "Loewe", "Lanvin", "Schiaparelli", "Ermenegildo Zegna"
    ],
    "Scientific Terms": [
        "Deoxyribonucleic", "Photosynthesis", "Electromagnetism", "Thermodynamics", "Quantum",
        "Chromatography", "Spectrosopy", "Superconductivity", "Crystallography", "Neurobiology",
        "Astrophysics", "Biochemistry", "Microbiology", "Nanotechnology", "Paleontology",
        "Geomorphology", "Seismology", "Meteorology", "Oceanography", "Bioinformatics",
        "Genomics", "Proteomics", "Epigenetics", "Phylogenetics", "Pharmacogenomics", "Cytogenetics", "Metabolomics", "Transcriptomics"
    ],
    "Acronyms": [
        "CRISPR-Cas9", "mRNA-1273", "UNESCO", "UNICEF", "CERN",
        "DARPA", "NASA", "NASDAQ", "INTERPOL", "EUROPOL",
        "IEEE", "ACM", "SIAM", "NIST", "NOAA",
        "USAF", "USMC", "USCG", "USN", "USAID",
        "UNHCR", "UNCTAD", "UNIDO", "UNEP", "IAEA", "WIPO", "ICAO", "IMO"
    ]
}


def load_evaluation_benchmark() -> List[Dict[str, Any]]:
    """Flatten all evaluation benchmark items into a list of dictionaries."""
    benchmark_items = []
    item_id = 1
    for category, words in BENCHMARK_CATEGORIES.items():
        for word in words:
            benchmark_items.append({
                "id": item_id,
                "category": category,
                "target_word": word,
                "prompt_sentence": f"The correct pronunciation of {word} is demonstrated here.",
            })
            item_id += 1
    return benchmark_items
