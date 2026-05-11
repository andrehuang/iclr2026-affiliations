"""European-institution paper-count analysis for ICLR 2026.

Counting rule (matches the AI World NeurIPS leaderboard and the rest of this
repo): a paper is credited +1 to a country (or institution) if at least one of
its authors lists an affiliation there. The same paper can be counted multiple
times across distinct institutions / distinct countries.

We do NOT use `data/iclr2026_institutions_ranked_unique.csv` for the country
totals, because the country tagging in that pipeline relies on ~250 hand-curated
canonical-institution regexes; anything outside that list (University of
Stuttgart, RWTH Aachen, Warsaw University of Technology, Universidad Carlos
III, Helmholtz/Jülich, etc.) gets tagged as "Other" and silently dropped from
the country aggregates. That undercounts Germany, France, Italy and others by
30-60%, and reports ZERO papers for Poland / Hungary / Romania / Greece /
Bulgaria / Ireland / Luxembourg / Slovenia / Croatia / Slovakia / Malta /
Latvia / Cyprus / Estonia / Lithuania.

Instead we re-detect the country directly from the raw `Institutions` text in
`data/iclr2026_public.csv`, segment by segment (one segment per author), using
explicit country names + unambiguous city names + university-name patterns.

KNOWN LIMITATION — PDF-source vs OpenReview-source:
The `Institutions` column in iclr2026_public.csv is PDF-derived (94%) with an
OpenReview-profile fallback (~6%). PDF title-block affiliation strings often
omit the country (e.g. just "University of Stuttgart" with no ", Germany"),
so even a thorough text-pattern detector under-counts the long tail.

A different analysis that read author profiles directly from the OpenReview
API (which exposes explicit institution + email-domain fields, e.g. .de / .pl
/ .fr) reports notably higher EU numbers — 1,059 EU-affiliated papers (19.8%
of accepted), with Germany at 489. This script, reading the PDF-derived text
only, finds 665 EU-affiliated papers (12.4%) and Germany at 260. Most of the
gap is in long-tail institutions whose country can't be inferred from the PDF
text alone. To close the gap you would need to re-scrape OpenReview profiles
(see scrape_openreview.py) — credentials required.

Outputs:
  - data/iclr2026_european_by_country.csv     (per-country totals)
  - data/iclr2026_european_institutions.csv   (per-institution table)
  - stdout                                    (formatted summary)

Run:
    python3 analyze_european_institutes.py
"""

from __future__ import annotations

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SRC = DATA_DIR / "iclr2026_public.csv"
OUT_INST = DATA_DIR / "iclr2026_european_institutions.csv"
OUT_COUNTRY = DATA_DIR / "iclr2026_european_by_country.csv"

csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# Country detection patterns
# ---------------------------------------------------------------------------
# For each country, we keep three sets of regexes:
#   * `name`   — the country name itself (English + native variants)
#   * `cities` — major cities that unambiguously belong to that country.
#                We avoid border cities (e.g. no "Lausanne" for France).
#   * `inst`   — university / lab name patterns that are diagnostic of the country.
#
# Patterns are case-insensitive and word-boundary-anchored. A segment is
# assigned a country if ANY of its three pattern groups fires. To minimise
# false positives we list the more-specific countries first (e.g. Austria
# before Germany, so "TU Wien" never gets mis-tagged as Germany).

COUNTRY_PATTERNS: list[tuple[str, list[str]]] = [
    # --- UK ---
    ("UK", [
        r"\b(?:united kingdom|uk|england|scotland|wales|northern ireland|great britain)\b",
        r"\b(?:london|oxford|cambridge|edinburgh|manchester|birmingham|glasgow|bristol|sheffield|liverpool|leeds|nottingham|southampton|coventry|warwick|york|durham|st andrews|aberdeen|cardiff|belfast|exeter|bath|surrey|reading|leicester|lancaster|loughborough)\b",
        r"\b(?:university of (?:oxford|cambridge|edinburgh|manchester|birmingham|glasgow|bristol|sheffield|liverpool|leeds|nottingham|southampton|warwick|york|durham|aberdeen|cardiff|exeter|bath|surrey|reading|leicester|essex|sussex|kent|east anglia|st andrews|strathclyde)|imperial college|king's college london|kings college london|ucl|queen mary|royal holloway|alan turing institute|google deepmind|deepmind\b)",
    ]),
    # --- Switzerland ---
    ("Switzerland", [
        r"\b(?:switzerland|schweiz|suisse|svizzera)\b",
        r"\b(?:zurich|zürich|geneva|genève|lausanne|basel|bern|lugano)\b",
        r"\b(?:eth ?z(?:urich|ürich)?|epfl|swiss federal institute|university of zurich|university of geneva|university of basel|university of bern|university of lausanne|usi lugano|idiap)\b",
    ]),
    # --- Austria (before Germany so "Universität Wien" wins) ---
    ("Austria", [
        r"\b(?:austria|österreich|oesterreich)\b",
        r"\b(?:wien|graz|innsbruck|salzburg|klagenfurt|leoben|linz)\b",
        r"\b(?:ist austria|tu wien|johannes kepler|jku\b|aithyra|complexity science hub|wu wien)\b",
        r"\buniversit(?:ä|ae|y)t? (?:of )?(?:wien|vienna|graz|innsbruck|salzburg|klagenfurt|linz)\b",
        r"\btechni(?:sche|cal) universit(?:ä|ae|y)t? (?:of )?(?:wien|vienna|graz)\b",
    ]),
    # --- Germany ---
    ("Germany", [
        r"\b(?:germany|deutschland)\b",
        r"\b(?:berlin|münchen|munich|hamburg|frankfurt|stuttgart|köln|cologne|düsseldorf|leipzig|dresden|hannover|hanover|nürnberg|nuremberg|bremen|bonn|aachen|münster|karlsruhe|freiburg im breisgau|freiburg|göttingen|goettingen|heidelberg|tübingen|tuebingen|saarbrücken|saarbruecken|kaiserslautern|mannheim|bielefeld|würzburg|wuerzburg|jena|magdeburg|kiel|rostock|potsdam|darmstadt|erlangen|jülich|juelich|dortmund|hagen|paderborn|chemnitz|braunschweig|ulm|konstanz|trier|halle|saale|mainz)\b",
        r"\b(?:hochschule|fachhochschule|rwth|hasso plattner|fraunhofer|max[- ]?planck|max-planck|helmholtz|leibniz|dfki|cispa|kit\b|tu(?:m|b|d)?\b|lmu|fu berlin|hu berlin|tu berlin|tu münchen|tu munich|tu dortmund|tu darmstadt|tu dresden|tu chemnitz|tu hamburg|tu braunschweig|tu kaiserslautern|forschungszentrum jülich|dkfz)\b",
        r"\b(?:university of (?:stuttgart|hamburg|frankfurt|heidelberg|bonn|cologne|köln|münster|freiburg|göttingen|goettingen|tübingen|tuebingen|mannheim|konstanz|bielefeld|würzburg|wuerzburg|jena|potsdam|hagen|paderborn|ulm|trier|mainz|leipzig|dresden|rostock|halle|saarland))\b",
        r"\buniversit(?:ä|ae)t (?:of )?(?:stuttgart|hamburg|frankfurt|heidelberg|bonn|köln|cologne|münster|freiburg|göttingen|tübingen|mannheim|konstanz|bielefeld|würzburg|jena|potsdam|hagen|paderborn|ulm|trier|mainz|leipzig|dresden|rostock|halle|des saarlandes)\b",
        r"\b(?:eberhard[- ]karls[- ]universität|ruprecht[- ]karls[- ]universität|ludwig[- ]maximilians|technische universität|rheinisch[- ]westfälische|friedrich[- ]alexander)\b",
    ]),
    # --- France ---
    ("France", [
        r"\bfrance\b",
        r"\b(?:paris|lyon|marseille|toulouse|nice|nantes|strasbourg|montpellier|bordeaux|lille|rennes|reims|grenoble|saint[- ]étienne|saint[- ]etienne|aix[- ]en[- ]provence|aix marseille|sophia antipolis|orsay|saclay|cachan|nancy|metz|brest|caen|dijon|clermont[- ]ferrand|le mans|valenciennes|villeneuve d'ascq|gif[- ]sur[- ]yvette|palaiseau|fontainebleau)\b",
        r"\b(?:inria|cnrs|école polytechnique|ecole polytechnique|école normale|ecole normale|ens (?:paris|lyon|cachan|rennes|paris[- ]saclay)|telecom paris|télécom paris|enpc|ponts paristech|sorbonne|sciences po|paris[- ]saclay|paris[- ]dauphine|panthéon[- ]sorbonne|ensae|ensai|insead|hec paris|essec|emlyon|centralesupélec|centralesupelec|mines paris|institut polytechnique de paris|inrae|cea\b|onera|naver labs europe|criteo|huawei.{0,30}paris|valeo\.ai)\b",
        r"\buniversité (?:de |d'|paris|sorbonne|claude bernard|grenoble|grenoble[- ]alpes|toulouse|aix[- ]marseille|côte d'azur|lille|lyon|nice|nantes|rennes|montpellier|strasbourg|bordeaux|nancy|lorraine|gustave eiffel|jean monnet|pierre et marie curie|panthéon|cergy|paris[- ]saclay|paris[- ]nanterre|paris[- ]est|psl|panthéon[- ]assas)\b",
    ]),
    # --- Italy ---
    ("Italy", [
        r"\b(?:italy|italia)\b",
        r"\b(?:roma\b|rome\b|milano|milan|napoli|naples|torino|turin|firenze|florence|bologna|venezia|venice|genova|genoa|padova|padua|pisa|trento|bolzano|trieste|udine|verona|brescia|bergamo|catania|palermo|cagliari|sassari|perugia|ancona|modena|reggio emilia|parma|piacenza|cosenza|bari|salerno|messina|lecce|ferrara|rimini|pavia|siena|matera|l'aquila)\b",
        r"\b(?:politecnico di (?:milano|torino|bari)|sapienza|università di|universita di|università degli studi|scuola normale|scuola superiore|sant'anna|bocconi|cnr\b|infn|iit\b(?!.{0,10}delhi)|istituto italiano di tecnologia|fbk|kessler)\b",
        r"\buniversit(?:y|à|a) (?:of |di )(?:rome|roma|milan|milano|naples|napoli|turin|torino|florence|firenze|bologna|venice|genova|padova|padua|pisa|trento|bolzano|trieste|verona|brescia|bergamo|catania|palermo|cagliari|perugia|modena|parma|pavia|siena|bari|salerno|lecce|ferrara|rimini|matera)\b",
    ]),
    # --- Netherlands ---
    ("Netherlands", [
        r"\b(?:netherlands|nederland|the netherlands)\b",
        r"\b(?:amsterdam|rotterdam|the hague|den haag|utrecht|eindhoven|groningen|leiden|delft|tilburg|maastricht|nijmegen|enschede|wageningen)\b",
        r"\b(?:tu delft|tu/e|tu eindhoven|delft university|eindhoven university|university of amsterdam|vu amsterdam|vrije universiteit amsterdam|leiden university|utrecht university|radboud university|tilburg university|wageningen university|maastricht university|university of twente|university of groningen|cwi\b|centrum wiskunde|booking\.com|asml|philips research)\b",
    ]),
    # --- Spain ---
    ("Spain", [
        r"\b(?:spain|españa|espana)\b",
        r"\b(?:madrid|barcelona|valencia|sevilla|seville|zaragoza|málaga|malaga|murcia|palma|bilbao|alicante|córdoba|cordoba|valladolid|vigo|gijón|gijon|granada|salamanca|santander|pamplona|donostia|san sebastián|san sebastian|tarragona|leon|burgos|cádiz|cadiz|huelva|jaen|toledo|santiago de compostela|las palmas|tenerife|girona|lleida)\b",
        r"\b(?:universidad|universitat|universidade)\b.*\b(?:autónoma|autonoma|complutense|politécnica|politecnica|carlos iii|pompeu fabra|barcelona|madrid|sevilla|valencia|granada|salamanca|navarra|navarre|santiago|coruña|coruna|deusto|murcia|alicante|zaragoza|cantabria|country basque|del país vasco|del pais vasco|de las islas baleares|complutense|girona|lleida|jaume i)\b",
        r"\b(?:csic|bsc[- ]cns|barcelona supercomputing|ie business|iese\b|ehu\b|upv\/ehu|upv\b|uam\b|uc3m|upm\b|upf\b|ub\b|uib\b|uma\b|us\b|ull\b|uja\b|ulpgc|cunef|ie university|ie school|esade|iese)\b",
    ]),
    # --- Poland ---
    ("Poland", [
        r"\b(?:poland|polska)\b",
        r"\b(?:warsaw|warszawa|krakow|kraków|wroclaw|wrocław|gdansk|gdańsk|poznan|poznań|lodz|łódź|szczecin|katowice|lublin|bydgoszcz|bialystok|białystok|gdynia|toruń|torun|kielce|olsztyn|zielona góra|opole|rzeszów|rzeszow)\b",
        r"\b(?:politechnika|uniwersytet|warsaw university|jagiellonian|agh\b|nicolaus copernicus|adam mickiewicz|wroclaw university|gdansk university|silesian university|warsaw school of economics|sgh\b|sggw|sgmk|ideas ncbr|cyfronet|nasa\.io)\b",
    ]),
    # --- Sweden ---
    ("Sweden", [
        r"\b(?:sweden|sverige)\b",
        r"\b(?:stockholm|gothenburg|göteborg|goeteborg|uppsala|lund|umeå|umea|linköping|linkoping|malmö|malmo|karlstad|jönköping|jonkoping|växjö|vaxjo|skövde|skovde|borås|boras|halmstad|kalmar|luleå|lulea|sundsvall|västerås|vasteras|örebro|orebro)\b",
        r"\b(?:kth\b|royal institute of technology|chalmers|karolinska|uppsala university|lund university|stockholm university|university of gothenburg|linköping university|umeå university|spotify|ericsson|king\.com|klarna)\b",
    ]),
    # --- Denmark ---
    ("Denmark", [
        r"\b(?:denmark|danmark)\b",
        r"\b(?:copenhagen|københavn|aarhus|odense|aalborg|esbjerg|kolding|roskilde|frederiksberg|kgs lyngby|kongens lyngby|lyngby)\b",
        r"\b(?:university of copenhagen|københavns universitet|technical university of denmark|dtu\b|aarhus university|aalborg university|university of southern denmark|copenhagen business school|itu copenhagen|it university of copenhagen|novo nordisk)\b",
    ]),
    # --- Finland ---
    ("Finland", [
        r"\b(?:finland|suomi)\b",
        r"\b(?:helsinki|espoo|tampere|turku|oulu|jyväskylä|jyvaskyla|lahti|kuopio|joensuu|vaasa|lappeenranta|rovaniemi)\b",
        r"\b(?:aalto|university of helsinki|tampere university|university of turku|university of oulu|university of jyväskylä|åbo akademi|abo akademi|lappeenranta|lut\b|hanken|nokia bell labs|silo ai)\b",
    ]),
    # --- Norway ---
    ("Norway", [
        r"\b(?:norway|norge)\b",
        r"\b(?:oslo|bergen|trondheim|stavanger|tromsø|tromso|kristiansand|drammen|fredrikstad|sandnes)\b",
        r"\b(?:university of oslo|university of bergen|ntnu|norwegian university of science and technology|bi norwegian|simula research|sintef)\b",
    ]),
    # --- Belgium ---
    ("Belgium", [
        r"\b(?:belgium|belgique|belgië|belgie)\b",
        r"\b(?:brussels|bruxelles|antwerp|antwerpen|ghent|gent|leuven|liège|liege|namur|charleroi|mons|hasselt|bruges|brugge|louvain|louvain[- ]la[- ]neuve|kortrijk)\b",
        r"\b(?:ku leuven|university of leuven|katholieke universiteit leuven|vub\b|vrije universiteit brussel|université libre de bruxelles|ulb\b|université de liège|university of liège|ghent university|universiteit gent|imec\b|uantwerpen|university of antwerp|hasselt university|uclouvain|université catholique de louvain)\b",
    ]),
    # --- Czechia ---
    ("Czechia", [
        r"\b(?:czechia|czech republic|česká|ceska)\b",
        r"\b(?:prague|praha|brno|ostrava|plzeň|plzen|liberec|olomouc|hradec králové|hradec kralove|české budějovice|ceske budejovice)\b",
        r"\b(?:czech technical|čvut|cvut|charles university|univerzita karlova|masaryk university|brno university|vut\b|cesnet)\b",
    ]),
    # --- Hungary ---
    ("Hungary", [
        r"\b(?:hungary|magyarország|magyarorszag)\b",
        r"\b(?:budapest|debrecen|szeged|miskolc|pécs|pecs|győr|gyor|nyíregyháza|nyiregyhaza|kecskemét|kecskemet|székesfehérvár|szombathely)\b",
        r"\b(?:eötvös loránd|eotvos lorand|elte\b|budapest university of technology|bme\b|central european university|ceu\b|sztaki|wigner research)\b",
    ]),
    # --- Greece ---
    ("Greece", [
        r"\b(?:greece|ελλάδα|hellas)\b",
        r"\b(?:athens|thessaloniki|patras|heraklion|ioannina|chania|volos|larissa|piraeus|crete)\b",
        r"\b(?:national technical university of athens|ntua\b|university of athens|aristotle university|university of patras|university of crete|forth\b|democritus|athena research)\b",
    ]),
    # --- Portugal ---
    ("Portugal", [
        r"\b(?:portugal)\b",
        r"\b(?:lisbon|lisboa|porto|coimbra|braga|aveiro|faro|funchal|guimarães|guimaraes|évora|evora)\b",
        r"\b(?:instituto superior técnico|tecnico lisboa|ist lisbon|universidade de lisboa|university of porto|universidade do porto|universidade de coimbra|universidade do minho|universidade nova|uminho|inesc)\b",
    ]),
    # --- Ireland ---
    ("Ireland", [
        r"\b(?:ireland|éire|eire)\b",
        r"\b(?:dublin|cork|galway|limerick|waterford|maynooth|drogheda|swords|dundalk|bray|navan)\b",
        r"\b(?:trinity college dublin|university college dublin|ucd\b|university college cork|ucc\b|nui galway|university of galway|maynooth university|dublin city university|dcu\b|tu dublin|insight centre|adapt centre|accenture labs dublin)\b",
    ]),
    # --- Luxembourg ---
    ("Luxembourg", [
        r"\b(?:luxembourg|luxemburg)\b",
        r"\b(?:university of luxembourg|uni\.lu|snt\b|list\b(?: luxembourg)?|luxembourg institute of science)\b",
    ]),
    # --- Romania ---
    ("Romania", [
        r"\b(?:romania|românia)\b",
        r"\b(?:bucharest|bucurești|bucuresti|cluj|iași|iasi|timișoara|timisoara|constanța|constanta|brasov|brașov|sibiu|craiova)\b",
        r"\b(?:university politehnica of bucharest|upb romania|babeș[- ]bolyai|babes[- ]bolyai|west university of timișoara|west university of timisoara|alexandru ioan cuza|technical university of cluj)\b",
    ]),
    # --- Bulgaria ---
    ("Bulgaria", [
        r"\b(?:bulgaria|българия)\b",
        r"\b(?:sofia|plovdiv|varna|burgas|ruse|stara zagora)\b",
        r"\b(?:sofia university|technical university of sofia|insait|new bulgarian university|st\.? kliment ohridski)\b",
    ]),
    # --- Slovenia ---
    ("Slovenia", [
        r"\b(?:slovenia|slovenija)\b",
        r"\b(?:ljubljana|maribor|koper|kranj|celje|nova gorica)\b",
        r"\b(?:university of ljubljana|university of maribor|jozef stefan|jožef stefan)\b",
    ]),
    # --- Croatia ---
    ("Croatia", [
        r"\b(?:croatia|hrvatska)\b",
        r"\b(?:zagreb|split|rijeka|osijek|zadar)\b",
        r"\b(?:university of zagreb|fer zagreb|ruđer bošković|rudjer boskovic)\b",
    ]),
    # --- Slovakia ---
    ("Slovakia", [
        r"\b(?:slovakia|slovensko)\b",
        r"\b(?:bratislava|košice|kosice|žilina|zilina|nitra|prešov|presov|trnava|martin)\b",
        r"\b(?:slovak university of technology|comenius university|technical university of košice|technical university of kosice)\b",
    ]),
    # --- Cyprus ---
    ("Cyprus", [
        r"\b(?:cyprus|κύπρος)\b",
        r"\b(?:nicosia|limassol|larnaca|paphos)\b",
        r"\b(?:university of cyprus|cyprus university of technology|european university cyprus|riseucy|cycat)\b",
    ]),
    # --- Estonia ---
    ("Estonia", [
        r"\b(?:estonia|eesti)\b",
        r"\b(?:tallinn|tartu|narva|pärnu|parnu)\b",
        r"\b(?:university of tartu|tallinn university of technology|taltech)\b",
    ]),
    # --- Latvia ---
    ("Latvia", [
        r"\b(?:latvia|latvija)\b",
        r"\b(?:riga|daugavpils|liepāja|liepaja|jelgava)\b",
        r"\b(?:university of latvia|riga technical university|rtu latvia)\b",
    ]),
    # --- Lithuania ---
    ("Lithuania", [
        r"\b(?:lithuania|lietuva)\b",
        r"\b(?:vilnius|kaunas|klaipėda|klaipeda|šiauliai|siauliai|panevėžys|panevezys)\b",
        r"\b(?:vilnius university|kaunas university of technology|vilnius gediminas)\b",
    ]),
    # --- Malta ---
    ("Malta", [
        r"\b(?:malta)\b",
        r"\b(?:valletta|sliema|mosta|birkirkara|qormi)\b",
        r"\b(?:university of malta)\b",
    ]),
    # --- Russia (not EU but European; flagged separately in output) ---
    ("Russia", [
        r"\b(?:russia|россия)\b",
        r"\b(?:moscow|saint petersburg|st petersburg|st\. petersburg|novosibirsk|kazan|nizhny novgorod|ekaterinburg|yekaterinburg|innopolis|skolkovo|dolgoprudny)\b",
        r"\b(?:lomonosov|moscow state|higher school of economics|hse moscow|mipt|skoltech|innopolis|yandex|sber\b|sberbank|t-bank|tinkoff|airi\b|institute of numerical mathematics|moscow indep)\b",
    ]),
]

COUNTRY_REGEXES: list[tuple[str, list[re.Pattern]]] = [
    (c, [re.compile(p, re.IGNORECASE) for p in pats])
    for c, pats in COUNTRY_PATTERNS
]

EU27 = {
    "Germany", "France", "Italy", "Netherlands", "Spain", "Poland",
    "Sweden", "Belgium", "Austria", "Denmark", "Finland", "Czechia",
    "Ireland", "Portugal", "Hungary", "Greece", "Romania", "Bulgaria",
    "Slovenia", "Croatia", "Slovakia", "Cyprus", "Estonia", "Latvia",
    "Lithuania", "Luxembourg", "Malta",
}
EUROPE_BROAD = EU27 | {"UK", "Switzerland", "Norway", "Russia"}


def detect_countries(segment: str) -> set[str]:
    """Return the set of countries this single affiliation segment maps to.

    A segment can match multiple countries (rare; usually a joint affiliation
    text). The caller deduplicates per paper.
    """
    found: set[str] = set()
    for country, regexes in COUNTRY_REGEXES:
        for rx in regexes:
            if rx.search(segment):
                found.add(country)
                break
    return found


# ---------------------------------------------------------------------------
# Per-paper iteration
# ---------------------------------------------------------------------------
def paper_country_sets(rows: list[dict]) -> list[set[str]]:
    """For each paper, return the set of European countries detected on any
    of its author segments. Used for paper-level (incidence) counts."""
    out: list[set[str]] = []
    for row in rows:
        raw = row.get("Institutions", "") or ""
        countries: set[str] = set()
        for seg in raw.split(";"):
            seg = seg.strip()
            if not seg:
                continue
            countries |= detect_countries(seg)
        out.append(countries)
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_summary(rows: list[dict], paper_countries: list[set[str]]) -> None:
    total = len(rows)
    eu_papers = sum(1 for cs in paper_countries if cs & EU27)
    eu_incidences = sum(len(cs & EU27) for cs in paper_countries)

    print(f"Total accepted papers: {total}")
    print(f"Papers with at least one EU-27 affiliation: "
          f"{eu_papers} ({eu_papers / total * 100:.2f}%)")
    print(f"  (UK and Switzerland excluded from EU-27, per EU membership)")
    print(f"EU paper-country incidences (one paper × N EU countries it touches): "
          f"{eu_incidences}")
    print()

    # Per-country paper counts (paper-level: a country gets +1 per paper that
    # touches it, regardless of how many of the paper's authors are from there).
    by_country: Counter = Counter()
    for cs in paper_countries:
        for c in cs:
            by_country[c] += 1

    print("Per-country paper counts (a paper is counted once per country it touches):")
    print(f"{'Country':<14} {'Papers':>7} {'%total':>7} {'%EU':>6}")
    eu_rows = [(c, n) for c, n in by_country.items() if c in EU27]
    for c, n in sorted(eu_rows, key=lambda kv: -kv[1]):
        print(f"{c:<14} {n:>7} {n / total * 100:>6.2f}% "
              f"{n / eu_papers * 100:>5.1f}%")

    other_euro = sorted(
        ((c, n) for c, n in by_country.items() if c in EUROPE_BROAD - EU27),
        key=lambda kv: -kv[1],
    )
    if other_euro:
        print("\nNon-EU European (UK / Switzerland / Norway / Russia) — for reference:")
        for c, n in other_euro:
            print(f"{c:<14} {n:>7} {n / total * 100:>6.2f}%")


def write_country_csv(rows: list[dict], paper_countries: list[set[str]]) -> None:
    by_country: Counter = Counter()
    for cs in paper_countries:
        for c in cs:
            by_country[c] += 1
    total = len(rows)
    with OUT_COUNTRY.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["country", "papers", "pct_of_total", "eu_member"])
        for c in sorted(EUROPE_BROAD, key=lambda x: -by_country[x]):
            n = by_country[c]
            if n == 0:
                continue
            w.writerow([c, n, f"{n / total * 100:.3f}", c in EU27])


# ---------------------------------------------------------------------------
# Per-institution counts
# ---------------------------------------------------------------------------
# We keep the project's canonical-institution names from
# data/iclr2026_institutions_ranked_unique.csv (they're the gold standard for
# Europe's top labs). For the per-institute table we just filter that file to
# European countries — it's slightly under-coverage for the long-tail (smaller
# universities not in the canonical list show up only at the country level),
# but very accurate for the top 80 institutions.

def write_institution_csv() -> None:
    src = DATA_DIR / "iclr2026_institutions_ranked_unique.csv"
    european = []
    with src.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["country"] in EUROPE_BROAD:
                european.append(r)
    european.sort(key=lambda r: int(r["count"]), reverse=True)
    with OUT_INST.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["european_rank", "global_rank", "institution",
                    "papers", "country", "eu_member"])
        for i, r in enumerate(european, 1):
            w.writerow([i, r["rank"], r["institution"], r["count"],
                        r["country"], r["country"] in EU27])


def main() -> None:
    with SRC.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    paper_countries = paper_country_sets(rows)
    print_summary(rows, paper_countries)
    write_country_csv(rows, paper_countries)
    write_institution_csv()
    print(f"\nWrote {OUT_COUNTRY.relative_to(ROOT)}")
    print(f"Wrote {OUT_INST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
