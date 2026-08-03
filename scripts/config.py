"""
Shared configuration for the pharmacogene functional-status pipeline.
Every other module imports from here — never hardcode paths, gene lists,
or API constants directly in a step file.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Gene panel
# ---------------------------------------------------------------------------

# Full 14-gene panel used for the initial ClinVar pull (broader net).
GENE_PANEL = [
    "CYP2D6", "CYP2C19", "CYP2C9", "CYP3A5", "CYP2B6", "CYP2A6",
    "CYP4F2", "DPYD", "TPMT", "NUDT15", "UGT1A1", "VKORC1",
    "SLCO1B1", "ABCG2",
]

# Locked 9-gene panel used from step 4 (crosswalk) onward.
LOCKED_PANEL = [
    "CYP2D6", "CYP2C19", "CYP2C9", "CYP3A5", "CYP2B6",
    "TPMT", "NUDT15", "UGT1A1", "SLCO1B1",
]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ALLELE_DEF_DIR = RAW_DIR / "allele_definitions"

for d in (RAW_DIR, PROCESSED_DIR, ALLELE_DEF_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ClinVar
# ---------------------------------------------------------------------------

CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)
CLINVAR_LOCAL = RAW_DIR / "variant_summary.txt.gz"

# ---------------------------------------------------------------------------
# ClinPGx / PharmGKB
# ---------------------------------------------------------------------------

GENE_PA_IDS = {
    "CYP2D6":  "PA128",
    "CYP2C19": "PA124",
    "CYP2C9":  "PA126",
    "CYP3A5":  "PA131",
    "CYP2B6":  "PA123",
    "CYP2A6":  "PA121",
    "CYP4F2":  "PA27121",
    "DPYD":    "PA145",
    "TPMT":    "PA356",
    "NUDT15":  "PA134963132",
    "UGT1A1":  "PA420",
    "VKORC1":  "PA133787052",
    "SLCO1B1": "PA134865839",
    "ABCG2":   "PA390",
}

CLINPGX_HAPLOTYPE_URL = "https://api.clinpgx.org/v1/site/gene/{pa_id}/haplotypes"

USABLE_FUNCTION_LABELS = {
    "Normal function", "Decreased function", "No function", "Increased function",
}
EXCLUDED_FUNCTION_LABELS = {"Uncertain function", "Unknown function"}

# ---------------------------------------------------------------------------
# VEP
# ---------------------------------------------------------------------------

VEP_REGION_URL = "https://rest.ensembl.org/vep/homo_sapiens/region"

# ---------------------------------------------------------------------------
# Conservation scores (step 6)
# ---------------------------------------------------------------------------

# phyloP100way is hg38-native -- queried directly, no liftover needed.
UCSC_API_URL = "https://api.genome.ucsc.edu/getData/track"
PHYLOP_TRACK = "phyloP100way"
PHYLOP_GENOME = "hg38"

# GERP is only hosted by UCSC for hg19, so variants must be lifted over
# from GRCh38 -> GRCh37 first via Ensembl's REST assembly-mapping endpoint.
ENSEMBL_MAP_URL = "https://rest.ensembl.org/map/human/GRCh38/{region}/GRCh37"
GERP_TRACK = "allHg19RS_BW"
GERP_GENOME = "hg19"

# ---------------------------------------------------------------------------
# Conservation scores (UCSC)
# ---------------------------------------------------------------------------

# phyloP100way is available natively for hg38 -- no liftover needed.
UCSC_API_URL = "https://api.genome.ucsc.edu/getData/track"
CONSERVATION_GENOME_HG38 = "hg38"
CONSERVATION_TRACK_HG38 = "phyloP100way"

# GERP is only hosted by UCSC for hg19 -- our ClinVar/VEP data is GRCh38,
# so GERP requires a hg38 -> hg19 liftover per variant before querying.
GERP_GENOME_HG19 = "hg19"
GERP_TRACK_HG19 = "allHg19RS_BW"
