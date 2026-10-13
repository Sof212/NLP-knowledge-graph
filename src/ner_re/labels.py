NER_TAG_DESCRIPTIONS = {
    "O": "Outside any entity",
    "geo": "Geographical entity",
    "gpe": "Geopolitical entity (city, country, state)",
    "per": "Person",
    "org": "Organization",
    "tim": "Time expression",
    "art": "Artifact",
    "nat": "Natural phenomenon",
    "eve": "Event",
}

WIKIDATA_PROPERTIES = {
    "P6": "head_of_government", "P17": "country", "P19": "place_of_birth",
    "P20": "place_of_death", "P22": "father", "P25": "mother", "P26": "spouse",
    "P27": "citizenship", "P30": "continent", "P31": "instance_of",
    "P35": "head_of_state", "P36": "capital", "P37": "official_language",
    "P39": "position_held", "P40": "child", "P50": "author", "P54": "sports_team",
    "P57": "director", "P58": "screenwriter", "P69": "educated_at", "P86": "composer",
    "P102": "political_party", "P108": "employer", "P112": "founded_by",
    "P118": "league", "P123": "publisher", "P127": "owned_by", "P131": "located_in",
    "P136": "genre", "P137": "operator", "P140": "religion", "P150": "contains",
    "P155": "follows", "P156": "followed_by", "P159": "headquarters",
    "P161": "cast_member", "P162": "producer", "P166": "award_received",
    "P170": "creator", "P171": "parent_taxon", "P172": "ethnic_group",
    "P175": "performer", "P176": "manufacturer", "P178": "developer",
    "P179": "series", "P190": "twinned_with", "P194": "legislative_body",
    "P205": "basin_country", "P206": "next_to_water", "P241": "military_branch",
    "P264": "record_label", "P272": "production_company", "P276": "location",
    "P279": "subclass_of", "P355": "subsidiary", "P361": "part_of",
    "P364": "original_language", "P400": "platform", "P403": "mouth_of_watercourse",
    "P449": "broadcaster", "P463": "member_of", "P488": "chairperson",
    "P495": "origin_country", "P527": "has_part", "P551": "residence",
    "P569": "date_of_birth", "P570": "date_of_death", "P571": "inception",
    "P576": "dissolved", "P577": "publication_date", "P580": "start_time",
    "P582": "end_time", "P585": "point_in_time", "P607": "conflict",
    "P674": "characters", "P676": "lyrics_by", "P706": "on_terrain",
    "P710": "participant", "P737": "influenced_by", "P740": "formation_location",
    "P749": "parent_org", "P800": "notable_work", "P807": "separated_from",
    "P840": "narrative_location", "P937": "work_location", "P1001": "jurisdiction",
    "P1056": "product", "P1198": "unemployment_rate", "P1336": "claimed_by",
    "P1344": "participant_in", "P1365": "replaces", "P1366": "replaced_by",
    "P1376": "capital_of", "P1412": "languages", "P1441": "present_in_work",
}


def property_name(pid):
    """
    Convert a Wikidata property id into a readable relation name.

    Parameters
    ----------
    pid : str
        Property id such as ``"P19"``.

    Returns
    -------
    str
        Readable name (e.g. ``"place_of_birth"``), or ``pid`` itself if
        unknown.
    """
    return WIKIDATA_PROPERTIES.get(pid, pid)
