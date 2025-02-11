# backend/data_processing/dept_mapping.py

import re
from functools import lru_cache
from fuzzywuzzy import fuzz

disciplines_dict = {
    'Science': {
        'Biology': [
            'biology', 'biological', 'microbiology', 'zoology', 'botany',
            'genetics', 'ecology', 'physiology', 'biochem', 'biophysics',
            'neuroscience', 'entomology', 'molecular biology', 'cell biology',
            'biomolecular', 'bioinformatics', 'biotechnology'
        ],
        'Chemistry': [
            'chemistry', 'chemical', 'biochem', 'biomolecular', 'chembiological',
            'chem/bio', 'chem/biomolecular', 'chem/biochemistry'
        ],
        'Physics': [
            'physics', 'astronomy', 'astrophysics', 'geophysics', 'optics',
            'physical science', 'physics & astronomy', 'physics & chemistry',
            'physics & planetary sciences'
        ],
        'Environmental Science': [
            'environmental', 'ecology', 'sustainability', 'conservation',
            'earth science', 'geoscience', 'marine science', 'oceanography',
            'forestry', 'natural resources', 'wildland resources',
            'environmental engineering', 'environmental design', 'wildlife'
        ],
        'Earth Science': [
            'geology', 'geomorphology', 'geochronology', 'atmospheric sciences',
            'geophysical sciences', 'earth & planetary sciences'
        ],
        'Materials Science': [
            'materials science', 'materials engineering', 'polymer science',
            'macromolecular science', 'metals', 'mineral engineering'
        ],
        'Agricultural Science': [
            'agronomy', 'horticulture', 'agricultural engineering', 'agriculture',
            'animal science', 'agribusiness', 'agricultural & environmental sciences',
            'agricultural economics', 'agriculture & resource economics'
        ],
        'Health Sciences': [
            'medicine', 'medical', 'biomedical', 'clinical', 'pathology',
            'pharmacology', 'anatomy', 'physiology', 'nursing', 'public health',
            'pharmacy', 'dietetics', 'occupational therapy', 'radiology',
            'speech language pathology', 'health administration',
            'health information technology', 'health care administration',
            'allied health', 'veterinary sciences', 'neuropsychiatry',
            'massage therapy', 'health care studies', 'health & kinesiology',
            'health & human performance', 'health & safety',
            'health & human services', 'holistic health care',
            'exercise & sport science', 'exercise science', 'wellness'
        ]
    },
    'Technology': {
        'Information Technology': [
            'information technology', 'it', 'information systems',
            'computer information systems', 'information science', 'informatics',
            'data science', 'computer & informational tech', 'computer information tech',
            'computer & math sciences', 'computer & information systems',
            'computer applications', 'computer applications office technologies',
            'computer applications-c.a.o.t.', 'computer skills',
            'computer business systems', 'computer information systems & business analytics',
            'computer business', 'computer & information tech.', 'comp networking & telecomm'
        ],
        'Computer Science': [
            'computer science', 'computing', 'software engineering',
            'computer engineering', 'cs', 'artificial intelligence',
            'machine learning', 'computing & engineering tech',
            'computer science-c.s.i.t.', 'computer and electrical engineering',
            'computer & math. sciences', 'computer management', 'computer systems technology',
            'computer & informational tech.', 'computer sci & info systems',
            'computer & information systems', 'computational science'
        ],
        'Data Science': [
            'data science', 'analytics', 'statistics', 'data analytics',
            'business analytics', 'decision science', 'decision & info science',
            'information decision sciences', 'quantitative methods',
            'quantitative theory', 'data science & analytics'
        ],
        'Multimedia Technology': [
            'multimedia', 'interactive multimedia product', 'interactive media design',
            'interactive games & media', 'interactive game development',
            'media & design', 'media design', 'media arts', 'new media art',
            'digital media', 'digital media production', 'web design & new media'
        ],
        'Communication Studies': [  # New sub-discipline
            'journalism', 'media studies', 'radio', 'tv', 'television', 'film', 'broadcasting',
            'mass communication', 'media communications', 'radio and television',
            'media journalism', 'radio tv film', 'radio tv', 'tv radio',
            'telecommunications', 'media & cinema studies', 'television film media studies',
            'media communications', 'communication design', 'communication/journalism',
            'communication culture', 'communications', 'media communications'
        ]
    },
    'Engineering': {
        'Mechanical Engineering': [
            'mechanical engineering', 'mechanical', 'mechatronics', 'mech', 'mech & materials engineering',
            'mech indust', 'mech & aerospace engineering', 'mech. & aerospace engineering',
            'mech & manuf engineerin', 'mech. aerospace engineering'
        ],
        'Electrical Engineering': [
            'electrical engineering', 'electrical', 'electronics', 'electronic',
            'eleccompenergy engineering', 'electrical & computer engineering',
            'electrical technology', 'eleccompenergy engineering'
        ],
        'Civil Engineering': [
            'civil engineering', 'civil', 'construction', 'structural engineering',
            'environmental engineering', 'urban planning', 'civil & environ engineering'
        ],
        'Chemical Engineering': [
            'chemical engineering', 'chemical', 'process engineering',
            'biochemical engineering', 'chem/biomolecular', 'chem/bio', 'chem/biochemistry',
            'chem/bioengineering'
        ],
        'Aerospace Engineering': [
            'aerospace engineering', 'aeronautical science', 'mech & aerospace engineering',
            'mech. & aerospace engineering', 'aeronautics', 'eleccompenergy engineering'
        ],
        'Industrial Engineering': [
            'industrial engineering', 'industrial design', 'industrial studies',
            'industrial technology', 'industrial design technology',
            'mech & materials engineering', 'mech indust', 'mech & manuf engineerin'
        ],
        'Materials Engineering': [
            'materials engineering', 'materials science', 'polymer science',
            'macromolecular science', 'metals', 'mineral engineering'
        ],
        'Environmental Engineering': [
            'environmental engineering', 'environmental design', 'wildland resources',
            'ecosystem science', 'environmental science & management'
        ]
    },
    'Mathematics': {
        'Pure Mathematics': [
            'pure mathematics', 'mathematics', 'math', 'mathematical sciences',
            'mathematical science', 'math & statistics', 'maths', 'mathematics education',
            'math & computing', 'maths'
        ],
        'Applied Mathematics': [
            'applied mathematics', 'computational mathematics', 'applied math',
            'applied mathematical sciences', 'applied mathematics & statistics',
            'applied math & statistics'
        ],
        'Statistics': [
            'statistics', 'statistical', 'biostatistics', 'math & statistics',
            'statistical sciences', 'quantitative methods', 'applied statistics'
        ]
    },
    'Arts': {
        'Visual Arts': [
            'art', 'visual arts', 'fine arts', 'graphic design', 'photography',
            'ceramics', 'sculpture', 'painting', 'illustration', 'printmaking',
            'studio art', 'jewelry', 'graphic communication', 'visual communication design',
            'drawing', 'digital media', 'interactive media design', 'multimedia',
            'animation', 'film & television', 'media & design', 'fashion design',
            'fashion merchandising', 'design', 'design technology',
            'visual communication', 'graphic arts', 'graphic information technology',
            'visual effects', 'design merchandising textiles', 'fibers'
        ],
        'Performing Arts': [
            'music', 'theater', 'dance', 'drama', 'performing arts', 'acting',
            'performance', 'theatre', 'ensemble', 'harmony', 'movement sciences',
            'voice', 'piano', 'percussion', 'music performance', 'music education',
            'music therapy', 'sound design', 'music business', 'music & dramatic arts'
        ],
        'Literature': [
            'literature', 'english', 'comparative literature', 'creative writing',
            'writing', 'poetry', 'english literature', 'english & literature',
            'english language & literature', 'english developmental english',
            'writing studies', 'literary & cultural studies', 'writing & rhetoric', 'folklore'
        ],
        'Music': [
            'music', 'composition', 'music performance', 'music education',
            'music therapy', 'sound design', 'music business', 'music synthesis',
            'songwriting', 'music & dramatic arts', 'music production engineering',
            'contemporary music writing', 'ear training', 'guitar'
        ],
        'Design': [
            'design', 'interior design', 'industrial design', 'graphic design',
            'fashion design', 'textiles & clothing', 'design merchandising',
            'graphic communication', 'visual communication design',
            'design technology', 'apparel merchandising', 'apparel communication tech.'
        ],
        'Communication Studies': [  # Ensure that 'Communication Studies' is also under Arts if desired
            'journalism', 'media studies', 'radio', 'tv', 'television', 'film', 'broadcasting',
            'mass communication', 'media communications', 'radio and television',
            'media journalism', 'radio tv film', 'radio tv', 'tv radio',
            'telecommunications', 'media & cinema studies', 'television film media studies',
            'media communications', 'communication design', 'communication/journalism',
            'communication culture', 'communications', 'media communications'
        ]
    },
    'Humanities': {
        'History': [
            'history', 'historical studies', 'american studies', 'history & classics',
            'classical studies', 'classical & medieval studies', 'appalachian studies',
            'history & philosophy', 'american & new england studies', 'liberal studies'
            'history & political science', 'history & sociology', 'civilization'
        ],
        'Philosophy': [
            'philosophy', 'ethics', 'logic', 'philosophical studies'
        ],
        'Languages': [
            'languages', 'spanish', 'french', 'german', 'italian', 'japanese',
            'chinese', 'linguistics', 'foreign languages', 'modern languages',
            'language studies', 'russian', 'arabic', 'hebrew', 'swahili',
            'hmong', 'latin', 'greek', 'roman studies', 'near eastern studies',
            'scandinavian', 'east asian languages & literature',
            'portuguese', 'vietnamese', 'hawaiian'
        ],
        'Religious Studies': [
            'religion', 'religious studies', 'theology', 'biblical studies',
            'judaic studies', 'islamic studies', 'buddhist studies',
            'theology & religious studies', 'bible missions ministry',
            'theology & religious studies'
        ],
        'Cultural Studies': [
            'cultural studies', 'african american studies', 'african studies',
            'chicano latino studies', 'latin american & latino studies',
            'indigenous studies', 'native studies', 'pan african studies',
            'multicultural studies', 'gender & women', 'women & gender studies',
            'womens studies', 'ethnic studies'
        ]
    },
    'Social Sciences': {
        'Sociology': [
            'sociology', 'social sciences', 'social work', 'criminology',
            'anthropology', 'archaeology', 'labor studies', 'human relations',
            'social welfare', 'community development', 'organizational behavior',
            'human services', 'human relations', 'community health', 'family studies',
            'family & consumer science', 'consumer family science', 'child & youth',
            'family & child studies', 'family social science', 'human development',
            'human communication studies', 'human development & family studies',
            'social studies', 'gender studies', 'conflict analysis',
            'decision sciences', 'decision science', 'organizational leadership'
        ],
        'Psychology': [
            'psychology', 'counseling', 'behavioral sciences', 'neuroscience',
            'psychology & sociology', 'psychology & counseling',
            'counseling psychology', 'child psychology', 'educational psychology',
            'developmental psychology', 'cognitive science'
        ],
        'Anthropology': [
            'anthropology', 'cultural studies', 'ethnic studies', 'african studies',
            'chicano studies', 'latin american studies', 'indigenous studies',
            'native studies', 'pan african studies', 'gender & women studies'
        ],
        'Political Science': [
            'political science', 'politics', 'government', 'international relations',
            'public policy', 'public administration', 'policy studies',
            'international business', 'international studies',
            'political studies', 'public affairs', 'policy planning development',
            'politics', 'international politics', 'government & reading'
        ],
        'Communication Studies': [  # We can also ensure that 'Communication Studies' is also under Social Sciences if desired
            'journalism', 'media studies', 'radio', 'tv', 'television', 'film', 'broadcasting',
            'mass communication', 'media communications', 'radio and television',
            'media journalism', 'radio tv film', 'radio tv', 'tv radio',
            'telecommunications', 'media & cinema studies', 'television film media studies',
            'media communications', 'communication design', 'communication/journalism',
            'communication culture', 'communications', 'media communications'
        ]
    },
    'Business': {
        'Economics': [
            'economics', 'economic', 'agricultural economics', 'resource economics',
            'agricultural & resource economics', 'economics & finance','travel & tourism',
            'economics & business', 'micro-economic', 'agribusiness', 'retail'
        ],
        'Management': [
            'management', 'business administration', 'business management',
            'organizational leadership', 'entrepreneurship', 'strategic management',
            'management & info systems', 'management information systems',
            'management & entrepreneurship', 'strategy', 'management science',
            'management science & information systems', 'strategy, entrep. & venture innovation',
            'organizational behavior', 'organizational leadership', 'leadership & organization'
        ],
        'Marketing': [
            'marketing', 'advertising', 'public relations', 'sales',
            'marketing & management', 'marketing & business law',
            'business & computer technology', 'fashion merchandising'
        ],
        'Finance': [
            'finance', 'accounting', 'banking', 'financial services',
            'investment', 'accountancy', 'finance & business law',
            'finance & real estate', 'accountancy & taxation',
            'acctg & information mgmt', 'accountancy & info systems',
            'finance, accounting & cis', 'finance & economics',
            'business & info. technology', 'acctg & taxation'
        ],
        'Business Technology': [
            'business technology', 'business & computer technology',
            'business & info. technology', 'business & computer technology',
            'business & information systems', 'business sys analysis & tech',
            'business systems', 'business technology', 'business office technology',
            'business & info systems', 'business administration & finance',
            'business communications', 'business law', 'business economics',
            'business management', 'business analysis', 'business administration'
        ]
    },
    'Law': {
        'Criminal Law': [
            'criminal law', 'criminology', 'justice studies', 'law enforcement',
            'criminology & criminal justice', 'criminology & law', 'criminal justice'
        ],
        'Corporate Law': [
            'corporate law', 'business law', 'commercial law', 'contracts',
            'intellectual property', 'property law', 'wills & trusts'
        ],
        'International Law': [
            'international law', 'human rights law', 'diplomacy & world affairs',
            'international affairs', 'international business', 'transnational studies'
        ],
        'Legal Studies': [
            'legal studies', 'legal assisting', 'legal writing',
            'paralegal', 'law & society', 'criminology, law & society',
            'law & political science', 'paralegal studies', 'legal skills'
        ]
    },
    'Architecture and Design': {
        'Architecture': [
            'architecture', 'architectural studies', 'interior design',
            'environmental design'
        ],
        'Urban Planning': [
            'urban planning', 'city planning', 'regional planning',
            'urban design & development', 'geography & urban studies'
        ],
        'Interior Design': [
            'interior design', 'environmental design', 'interior architecture',
            'design & interior design'
        ],
        'Industrial Design': [
            'industrial design', 'product design', 'design', 'industrial studies'
        ]
    },
    'Interdisciplinary Studies': {
        'Interdisciplinary Studies': [
            'interdisciplinary', 'liberal arts', 'general studies', 'university studies',
            'integrated studies', 'multiple', 'not specified', 'liberal studies',
            'humanities & social sciences', 'science technology & society',
            'liberal arts & sciences', 'core humanities', 'core curriculum',
            'university college', 'residential college', 'foundation studies',
            'guided studies', 'university seminar', 'university community',
            'college success', 'student success', 'study skills', 'learning skills',
            'learning center', 'workload', 'multiple', 'core curriculum',
            'residential college', 'guided studies', 'learning center',
            'college success', 'study skills', 'learning skills', 'Science Technology & Society'
        ]
    }
}


# 1. Create lookup dictionary for faster matching
DEPT_LOOKUP = {}
for discipline, subdict in disciplines_dict.items():
    for sub_discipline, keywords in subdict.items():
        for kw in keywords:
            DEPT_LOOKUP[kw] = (discipline, sub_discipline)

@lru_cache(maxsize=1000)
def preprocess_department(dept):
    """Cache preprocessed department names"""
    if not dept:
        return ""
    dept = dept.lower()
    dept = dept.replace('&','and').replace('-',' ').replace('/',' ')
    dept = re.sub(r'[^a-z ]','',dept)
    return re.sub(r'\s+',' ',dept).strip()

@lru_cache(maxsize=5000)
def categorize_single_department(dept):
    """Cached department categorization"""
    if not dept:
        return ('Interdisciplinary Studies','Interdisciplinary Studies')
    
    pre_dept = preprocess_department(dept)
    
    # First try direct matching
    for keyword, category in DEPT_LOOKUP.items():
        if keyword in pre_dept:
            return category
            
    # If no direct match, try fuzzy matching only on unique keywords
    best_match = ('Interdisciplinary Studies','Interdisciplinary Studies')
    best_score = 0
    
    for keyword, category in DEPT_LOOKUP.items():
        score = fuzz.partial_ratio(pre_dept, keyword)
        if score >= 80 and score > best_score:
            best_score = score
            best_match = category
    
    return best_match

def map_departments(df):
    """Vectorized department mapping"""
    print("Starting department mapping...")
    
    # Fill missing values
    df['department'] = df['department'].fillna('')
    
    # Get unique departments first
    unique_depts = df['department'].unique()
    
    # Create mapping dictionary for unique values
    dept_mapping = {
        dept: categorize_single_department(dept) 
        for dept in unique_depts
    }
    
    # Map disciplines and sub-disciplines using the dictionary
    df['discipline'] = df['department'].map(lambda x: dept_mapping[x][0])
    df['sub_discipline'] = df['department'].map(lambda x: dept_mapping[x][1])
    
    print("Department mapping complete")
    return df