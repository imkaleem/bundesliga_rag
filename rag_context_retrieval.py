#!/usr/bin/env python3
"""
Context retrieval module for a Retrieval-Augmented Generation (RAG) chatbot focused on Germany`s 1. Bundesliga.

The script processes colloquial user questions about football clubs in Germany’s
1. Bundesliga, resolves the referenced city or club, retrieves the current head
coach from Wikidata, and enriches the result with descriptive context from
Wikipedia.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import unicodedata
import urllib.parse
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_SUMMARY_URL_TMPL = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"

DEFAULT_USER_AGENT = (
    "bundesliga-rag-context-retrieval/1.0 "
    "https://github.com/imkaleem/bundesliga_rag; "
    "contact: mkaleemc@gmail.com "
    "requests"
)

##### Wikidata identifiers #####
# Bundesliga (association football league) item on Wikidata
BUNDESLIGA_QID = "Q82595"
# Association football club item
ASSOCIATION_FOOTBALL_CLUB_QID = "Q476028"
# head coach property
HEAD_COACH_PID = "P286"


# -----------------------------
# Exceptions
# -----------------------------
class RetrievalError(RuntimeError):
    """Base error for retrieval pipeline."""


class EntityResolutionError(RetrievalError):
    """Raised when we cannot map a question to a Bundesliga club."""


class UpstreamDataError(RetrievalError):
    """Raised when upstream sources (Wikidata/Wikipedia) are missing or inconsistent (or fail)."""


# -----------------------------
# Utilities
# -----------------------------
def normalize_text(text: str) -> str:
    """
    Normalize user input and labels to improve matching.
    - lowercases
    - strips accents (e.g., München -> Munchen)
    - removes punctuation (keeps letters, digits, spaces, hyphens)
    - collapses whitespace

    This is deliberately simple because the challenge assumes correct spelling
    (except upper/lowercase).
    """
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    # Replace punctuation with spaces (keep word chars, spaces, hyphens)
    text = re.sub(r"[^\w\s-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def qid_from_uri(uri: str) -> str:
    """Convert a Wikidata entity URI (Uniform Resource Identifier) to a QID (the last path component)."""
    return uri.rsplit("/", 1)[-1]


def safe_truncate(text: str, max_chars: int = 900) -> str:
    """Truncate long strings for logs/prompts without cutting mid-word too aggressively."""
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    cut = text[: max_chars - 1]
    # cut at last whitespace if possible
    last_space = cut.rfind(" ")
    if last_space > 0 and (max_chars - last_space) < 60:
        cut = cut[:last_space]
    return cut + "…"


# -----------------------------
# Data models
# -----------------------------
@dataclass(frozen=True)
class ClubInfo:
    club_qid: str
    club_label_en: str
    club_label_de: str
    city_label_en: str
    city_label_de: str
    club_alt_labels: List[str]
    city_alt_labels: List[str]

    @property
    def display_name(self) -> str:
        # Prefer English label when present
        return self.club_label_en or self.club_label_de or self.club_qid

    @property
    def city_display(self) -> str:
        return self.city_label_en or self.city_label_de or "Unknown city"


@dataclass(frozen=True)
class CoachInfo:
    """Model for a coach with labels, descriptions, and optional Wikipedia info."""
    coach_qid: str
    label_en: str
    label_de: str
    description_en: str
    description_de: str
    wikipedia_lang: Optional[str]
    wikipedia_title: Optional[str]

    @property
    def display_name(self) -> str:
        return self.label_en or self.label_de or self.coach_qid

    @property
    def display_description(self) -> str:
        return self.description_en or self.description_de or ""


# ------------------------
# HTTP client with retries
# ------------------------
class HttpClient:
    """HTTP client with retry logic."""
    def __init__(self, user_agent: str, timeout_s: float = 15.0):
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "application/sparql-results+json, application/json;q=0.9, */*;q=0.8",
            }
        )
        retries = Retry(
            total=4,
            backoff_factor=0.7,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get(self, url: str, *, params: Optional[dict] = None, headers: Optional[dict] = None) -> requests.Response:
        return self.session.get(url, params=params, headers=headers, timeout=self.timeout_s)

    def post(self, url: str, *, data: Optional[dict] = None, headers: Optional[dict] = None) -> requests.Response:
        return self.session.post(url, data=data, headers=headers, timeout=self.timeout_s)


# -----------------------------
# Wikidata client (SPARQL + API)
# -----------------------------
class WikidataClient:
    """
    Client for querying structured football-related data from Wikidata.

    This class is responsible for executing SPARQL queries against the Wikidata
    endpoint and returning normalized results required by the application,
    such as clubs currently participating in the 1. Bundesliga and their
    respective head coaches.
    """
    def __init__(self, http: HttpClient, logger: logging.Logger):
        self.http = http
        self.logger = logger

    def sparql(self, query: str, request_id: str) -> dict:
        """
        Query Wikidata Query Service. Returns parsed JSON.
        """
        self.logger.debug("[%s] SPARQL query:\n%s", request_id, query)
        resp = self.http.get(WIKIDATA_SPARQL_URL, params={"query": query, "format": "json"})
        self.logger.debug("[%s] SPARQL HTTP %s", request_id, resp.status_code)
        if resp.status_code != 200:
            raise UpstreamDataError(
                f"Wikidata SPARQL request failed (HTTP {resp.status_code}). "
                f"Try again later."
            )
        return resp.json()

    def wbgetentities(self, qids: List[str], request_id: str, languages: str = "en|de") -> dict:
        """
        Call Wikidata API (wbgetentities) for labels/descriptions/sitelinks.
        """
        ids = "|".join(qids)
        params = {
            "action": "wbgetentities",
            "ids": ids,
            "props": "labels|descriptions|sitelinks",
            "languages": languages,
            "format": "json",
        }
        resp = self.http.get(WIKIDATA_API_URL, params=params)
        self.logger.debug("[%s] wbgetentities HTTP %s for %s", request_id, resp.status_code, ids)
        if resp.status_code != 200:
            raise UpstreamDataError(
                f"Wikidata API request failed (HTTP {resp.status_code}). Try again later."
            )
        return resp.json()


# -----------------------------
# Wikipedia client
# -----------------------------
class WikipediaClient:
    """
    Client for retrieving unstructured descriptive content from Wikipedia.

    This class encapsulates access to the Wikipedia API and is responsible for
    fetching the introductory sections of articles used as contextual background
    information in the retrieval pipeline. It intentionally avoids factual
    assertions and complements structured data retrieved from Wikidata.
    """
    def __init__(self, http: HttpClient, logger: logging.Logger):
        self.http = http
        self.logger = logger

    def get_page_summary(self, title: str, request_id: str, lang: str = "en") -> Optional[str]:
        """
        Fetch the 'extract' from Wikipedia's REST summary endpoint.
        Returns None if not available.
        """
        if not title:
            return None
        encoded = urllib.parse.quote(title.replace(" ", "_"), safe="")
        url = WIKIPEDIA_SUMMARY_URL_TMPL.format(lang=lang, title=encoded)
        resp = self.http.get(url, headers={"Accept": "application/json"})
        self.logger.debug("[%s] Wikipedia summary %s HTTP %s", request_id, url, resp.status_code)
        if resp.status_code != 200:
            return None
        data = resp.json()
        extract = data.get("extract")
        if not extract:
            return None
        return extract.strip()


# -----------------------------
# Bundesliga index (city/club -> club QID)
# -----------------------------
class BundesligaIndex:
    """
    Builds a lightweight, local index that links city names (and common club aliases)
    to the set of 1. Bundesliga clubs.

    This is the piece that "connects an input string to a data model".
    """

    def __init__(self, wd: WikidataClient, logger: logging.Logger):
        self.wd = wd
        self.logger = logger
        self._built_at = 0.0
        self.city_to_clubs: Dict[str, List[ClubInfo]] = {}
        self.clubkey_to_clubs: Dict[str, List[ClubInfo]] = {}
        self._sorted_city_keys: List[str] = []
        self._sorted_club_keys: List[str] = []

    @property
    def built_at(self) -> float:
        return self._built_at

    def refresh(self, request_id: str) -> None:
        """
        Refreshes the internal mapping of Bundesliga clubs to their associated cities.

        - The method queries Wikidata for football clubs currently affiliated with the Bundesliga and derives
        a city-to-club mapping from these entities, reflecting current-season membership as maintained in Wikidata.

        - Season-specific queries (e.g., individual Bundesliga seasons) would require querying the relevant season entity
        and its participants; this is intentionally out of scope but can be added if needed.
        """

        query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT
  ?club
  (SAMPLE(?clubLabelEn) AS ?clubLabelEn)
  (SAMPLE(?clubLabelDe) AS ?clubLabelDe)
  (GROUP_CONCAT(DISTINCT ?clubAlt; separator="|") AS ?clubAltLabels)
  ?city
  (SAMPLE(?cityLabelEn) AS ?cityLabelEn)
  (SAMPLE(?cityLabelDe) AS ?cityLabelDe)
  (GROUP_CONCAT(DISTINCT ?cityAlt; separator="|") AS ?cityAltLabels)
WHERE {{
  ?club wdt:P31/wdt:P279* wd:{ASSOCIATION_FOOTBALL_CLUB_QID} ;
        wdt:P118 wd:{BUNDESLIGA_QID} .

  OPTIONAL {{ ?club wdt:P159 ?city1 . }}   # headquarters location
  OPTIONAL {{ ?club wdt:P131 ?city2 . }}   # located in the administrative territorial entity
  OPTIONAL {{ ?club wdt:P276 ?city3 . }}   # location
  BIND(COALESCE(?city1, ?city2, ?city3) AS ?city)

  OPTIONAL {{ ?club rdfs:label ?clubLabelEn FILTER(LANG(?clubLabelEn)="en") }}
  OPTIONAL {{ ?club rdfs:label ?clubLabelDe FILTER(LANG(?clubLabelDe)="de") }}
  OPTIONAL {{ ?club skos:altLabel ?clubAlt FILTER(LANG(?clubAlt)="en" || LANG(?clubAlt)="de") }}

  OPTIONAL {{ ?city rdfs:label ?cityLabelEn FILTER(LANG(?cityLabelEn)="en") }}
  OPTIONAL {{ ?city rdfs:label ?cityLabelDe FILTER(LANG(?cityLabelDe)="de") }}
  OPTIONAL {{ ?city skos:altLabel ?cityAlt FILTER(LANG(?cityAlt)="en" || LANG(?cityAlt)="de") }}
}}
GROUP BY ?club ?city
"""
        data = self.wd.sparql(query, request_id=request_id)

        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            raise UpstreamDataError("Could not load Bundesliga club index from Wikidata.")

        city_to_clubs: Dict[str, List[ClubInfo]] = {}
        clubkey_to_clubs: Dict[str, List[ClubInfo]] = {}

        def parse_concat(val: str) -> List[str]:
            if not val:
                return []
            parts = [p.strip() for p in val.split("|") if p.strip()]
            # de-duplicate while preserving order
            seen = set()
            out = []
            for p in parts:
                key = normalize_text(p)
                if key and key not in seen:
                    out.append(p)
                    seen.add(key)
            return out

        for row in bindings:
            club_uri = row["club"]["value"]
            club_qid = qid_from_uri(club_uri)

            # Labels are optional; fall back to QID
            club_en = row.get("clubLabelEn", {}).get("value", "")
            club_de = row.get("clubLabelDe", {}).get("value", "")
            city_uri = row.get("city", {}).get("value")
            if not city_uri:
                # Can't map a city -> skip, but log for visibility.
                self.logger.debug("[%s] Club %s missing city (P159/P131/P276). Skipping.", request_id, club_qid)
                continue
            city_qid = qid_from_uri(city_uri)
            city_en = row.get("cityLabelEn", {}).get("value", "")
            city_de = row.get("cityLabelDe", {}).get("value", "")

            club_alt = parse_concat(row.get("clubAltLabels", {}).get("value", ""))
            city_alt = parse_concat(row.get("cityAltLabels", {}).get("value", ""))

            club = ClubInfo(
                club_qid=club_qid,
                club_label_en=club_en,
                club_label_de=club_de,
                city_label_en=city_en,
                city_label_de=city_de,
                club_alt_labels=club_alt,
                city_alt_labels=city_alt,
            )

            # Index city keys
            for city_name in [city_en, city_de] + city_alt:
                key = normalize_text(city_name)
                if key:
                    city_to_clubs.setdefault(key, []).append(club)

            # Index club keys (aliases, but better to keep them separate to avoid city false positives)
            for club_name in [club_en, club_de] + club_alt:
                for key in self._generate_club_keys(club_name):
                    clubkey_to_clubs.setdefault(key, []).append(club)

            # A few special-case aliases that are very common in colloquial questions:
            # - "Pauli" for FC St. Pauli
            # - "HSV" for Hamburger SV
            label_blob = normalize_text(" ".join([club_en, club_de] + club_alt))
            if "pauli" in label_blob:
                clubkey_to_clubs.setdefault("pauli", []).append(club)
                clubkey_to_clubs.setdefault("st pauli", []).append(club)
            if "hamburger sv" in label_blob:
                clubkey_to_clubs.setdefault("hsv", []).append(club)

            self.logger.debug(
                "[%s] Indexed club %s (%s) in city %s (%s).",
                request_id, club_qid, club.display_name, city_qid, club.city_display
            )

        # De-duplicate lists
        def dedupe(mapping: Dict[str, List[ClubInfo]]) -> Dict[str, List[ClubInfo]]:
            """
            Removes duplicate club entries from a city-to-club mapping.

            The function ensures that each club appears only once per city key,
            which simplifies downstream resolution logic and avoids ambiguous
            or repeated results during retrieval.
            """
            out = {}
            for k, clubs in mapping.items():
                seen = set()
                uniq = []
                for c in clubs:
                    if c.club_qid not in seen:
                        uniq.append(c)
                        seen.add(c.club_qid)
                out[k] = uniq
            return out

        self.city_to_clubs = dedupe(city_to_clubs)
        self.clubkey_to_clubs = dedupe(clubkey_to_clubs)

        # Pre-sort keys to prefer longest matches (e.g., "st pauli" over "pauli")
        self._sorted_city_keys = sorted(self.city_to_clubs.keys(), key=len, reverse=True)
        self._sorted_club_keys = sorted(self.clubkey_to_clubs.keys(), key=len, reverse=True)

        self._built_at = time.time()
        self.logger.info(
            "[%s] Bundesliga index built: %d city keys, %d club keys.",
            request_id, len(self.city_to_clubs), len(self.clubkey_to_clubs)
        )

    @staticmethod
    def _generate_club_keys(name: str) -> List[str]:
        """
        Generate stable "match keys" for a club label/alias, filtering out extremely generic tokens.

        Example: "FC Bayern Munich" -> ["bayern munich", "bayern", "munich"] (filtered)
        It is kept this conservative to avoid mapping "fc" or "sv" as entities.
        """
        n = normalize_text(name)
        if not n:
            return []
        # Remove common club generic tokens
        boilerplate = {"fc", "sv", "vfb", "vfl", "tsg", "1", "ii", "iv", "04", "05", "09", "1899", "1846"}
        tokens = [t for t in n.split() if t not in boilerplate]
        if not tokens:
            return []
        keys = set()
        keys.add(" ".join(tokens))
        # also add single-token keys for distinctive tokens
        for t in tokens:
            if len(t) >= 5 or t in {"hsv", "pauli"}:
                keys.add(t)
        return sorted(keys, key=len, reverse=True)

    def list_supported_cities(self, max_items: int = 40) -> List[str]:
        """
        Return a readable list of supported city names (best-effort, may include duplicates).
        """
        # Prefer the shortest/most common representation per key
        keys = sorted(self.city_to_clubs.keys())
        # Filter overly generic keys
        keys = [k for k in keys if len(k) >= 3]
        return keys[:max_items]

    def resolve_club(self, question: str, request_id: str) -> Tuple[ClubInfo, str]:
        """
        Resolve the user's question to a Bundesliga club.
        Returns (ClubInfo, matched_key).

        Strategy:
        1) Match against city keys first (because the scenario is "coach of a city").
        2) If no city match, match against club keys (e.g., "pauli").
        3) If multiple clubs match the same key, disambiguate using club-key matches in question.
        """
        q_norm = normalize_text(question)
        self.logger.debug("[%s] Normalized question: %s", request_id, q_norm)

        # Special-case explicitly: "pauli" should map to St. Pauli if present.
        if re.search(r"\bpauli\b", q_norm):
            clubs = self.clubkey_to_clubs.get("pauli", [])
            if clubs:
                self.logger.info("[%s] Resolved via special alias 'pauli' -> %s", request_id, clubs[0].display_name)
                return clubs[0], "pauli"

        city_match = self._find_best_key_match(q_norm, self._sorted_city_keys)
        if city_match:
            key = city_match
            clubs = self.city_to_clubs.get(key, [])
            if not clubs:
                raise EntityResolutionError(f"Matched city key '{key}' but found no clubs (unexpected).")

            if len(clubs) == 1:
                self.logger.info("[%s] Resolved city '%s' -> %s", request_id, key, clubs[0].display_name)
                return clubs[0], key

            # Disambiguation: look for club key matches within question
            club = self._disambiguate_by_club_keys(q_norm, clubs)
            if club:
                self.logger.info(
                    "[%s] Resolved ambiguous city '%s' -> %s (disambiguated).",
                    request_id, key, club.display_name
                )
                return club, key

            # As a reasonable default, prefer the club whose name contains the city
            for c in clubs:
                if key in normalize_text(c.display_name):
                    return c, key

            raise EntityResolutionError(
                f"City '{key}' maps to multiple Bundesliga clubs: " + ", ".join(c.display_name for c in clubs)
            )

        club_match = self._find_best_key_match(q_norm, self._sorted_club_keys)
        if club_match:
            key = club_match
            clubs = self.clubkey_to_clubs.get(key, [])
            if len(clubs) == 1:
                self.logger.info("[%s] Resolved club key '%s' -> %s", request_id, key, clubs[0].display_name)
                return clubs[0], key
            if len(clubs) > 1:
                club = self._disambiguate_by_club_keys(q_norm, clubs)
                if club:
                    return club, key
                raise EntityResolutionError(
                    f"Term '{key}' maps to multiple Bundesliga clubs: " + ", ".join(c.display_name for c in clubs)
                )

        raise EntityResolutionError("Could not resolve a Bundesliga city/club from the question.\n")

    @staticmethod
    def _find_best_key_match(question_norm: str, keys_sorted_desc_len: List[str]) -> Optional[str]:
        """
        Find the longest matching key in the normalized question.

        We allow an optional trailing possessive 's' (e.g., "heidenheims").
        """
        for key in keys_sorted_desc_len:
            if not key:
                continue
            # Word boundary match, allow trailing possessive s/'s
            pattern = r"\b" + re.escape(key) + r"(?:s|'s)?\b"
            if re.search(pattern, question_norm):
                return key
        return None

    def _disambiguate_by_club_keys(self, question_norm: str, candidates: List[ClubInfo]) -> Optional[ClubInfo]:
        """
        If multiple clubs share a city, try to disambiguate using club tokens.
        """
        for club in candidates:
            # look for distinctive club key matches
            keys = set()
            for name in [club.club_label_en, club.club_label_de] + club.club_alt_labels:
                for k in self._generate_club_keys(name):
                    keys.add(k)
            for k in sorted(keys, key=len, reverse=True):
                if len(k) < 4 and k not in {"hsv", "pauli"}:
                    continue
                if re.search(r"\b" + re.escape(k) + r"\b", question_norm):
                    return club
        return None


# -----------------------
# Retrieval pipeline
# -----------------------
class RetrievalPipeline:
    """
    Orchestrates the retrieval pipeline for a single user question.

    This class is responsible for resolving the user's question to a Bundesliga
    club, retrieving the current head coach from Wikidata, and enriching the
    result with descriptive context from Wikipedia.
    """
    def __init__(self, index: BundesligaIndex, wd: WikidataClient, wp: WikipediaClient, logger: logging.Logger):
        self.index = index
        self.wd = wd
        self.wp = wp
        self.logger = logger

    def get_current_coach(self, club_qid: str, request_id: str) -> CoachInfo:
        """
        Retrieve the current head coach QID from Wikidata via SPARQL,
        then enrich via Wikidata API (labels/descriptions/sitelinks).
        """
        # SPARQL query to retrieve the current head coach QID from Wikidata
        # prefixes are shorthand aliases for long URIs
        query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?coach WHERE {{
  wd:{club_qid} wdt:{HEAD_COACH_PID} ?coach .
}}
LIMIT 5
"""
        data = self.wd.sparql(query, request_id=request_id)
        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            raise UpstreamDataError(f"No head coach (P286) found on Wikidata for club {club_qid}.")

        coach_qid = qid_from_uri(bindings[0]["coach"]["value"])

        entity_data = self.wd.wbgetentities([coach_qid], request_id=request_id)
        ent = (entity_data.get("entities") or {}).get(coach_qid) or {}
        labels = ent.get("labels") or {}
        descs = ent.get("descriptions") or {}
        sitelinks = ent.get("sitelinks") or {}

        label_en = (labels.get("en") or {}).get("value", "")
        label_de = (labels.get("de") or {}).get("value", "")
        desc_en = (descs.get("en") or {}).get("value", "")
        desc_de = (descs.get("de") or {}).get("value", "")

        # Prefer English Wikipedia for summary; fall back to German
        wiki_lang = None
        wiki_title = None
        if "enwiki" in sitelinks:
            wiki_lang = "en"
            wiki_title = sitelinks["enwiki"].get("title")
        elif "dewiki" in sitelinks:
            wiki_lang = "de"
            wiki_title = sitelinks["dewiki"].get("title")

        coach = CoachInfo(
            coach_qid=coach_qid,
            label_en=label_en,
            label_de=label_de,
            description_en=desc_en,
            description_de=desc_de,
            wikipedia_lang=wiki_lang,
            wikipedia_title=wiki_title,
        )
        self.logger.info("[%s] Coach for %s -> %s (%s)", request_id, club_qid, coach.display_name, coach_qid)
        return coach

    def get_coach_intro(self, coach: CoachInfo, request_id: str) -> str:
        """
        Retrieve the intro/lead from Wikipedia summary endpoint.
        Falls back to Wikidata description if Wikipedia is missing.
        """
        if coach.wikipedia_title and coach.wikipedia_lang:
            summary = self.wp.get_page_summary(coach.wikipedia_title, request_id=request_id, lang=coach.wikipedia_lang)
            if summary:
                return summary

        # Fallback: attempt English then German by using the label as title
        # This handles cases where sitelinks are missing.
        for lang in ["en", "de"]:
            title_guess = coach.label_en if lang == "en" else coach.label_de
            if not title_guess:
                continue
            summary = self.wp.get_page_summary(title_guess, request_id=request_id, lang=lang)
            if summary:
                return summary

        # Last resort: Wikidata description
        desc = coach.display_description
        if desc:
            return desc
        return "No Wikipedia intro available for this coach."

    @staticmethod
    def build_llm_prompt(user_question: str, club: Optional[ClubInfo], coach: Optional[CoachInfo], coach_intro: str, error: Optional[str] = None) -> str:
        """
        Construct the final prompt with system instructions, user question, and retrieved context.
        """
        system_prompt = (
                        "You are a factual assistant that answers ONLY questions of the form "
                        "'Who is coaching <city/club>?' for clubs in Germany's 1. Bundesliga.\n"
                        "Use ONLY the CONTEXT provided in the CONTEXT section below. Do NOT add "
                        "any facts that are not present in that context. If the context lacks "
                        "the requested information, state that you cannot determine the coach "
                        "from the available data and ask the user a concise clarifying question.\n"
                        "When answering, return the coach's full name and one or two concise, "
                        "verifiable facts from the Wikipedia intro (for example: nationality, "
                        "current role start date, notable prior club). Include a single-line "
                        "provenance string identifying the source(s) (e.g. 'Source: enwiki / "
                        "ArticleTitle; Wikidata Q####').\n"
                        "Match the language of the user's question when possible. Keep the "
                        "response short and factual."
                    )


        context_lines = []
        if error:
            context_lines.append(f"Retrieval status: ERROR - {error}")
        else:
            context_lines.append("Retrieval status: OK")

        if club:
            context_lines.append(f"Resolved Bundesliga club: {club.display_name} (Wikidata: {club.club_qid})")
            context_lines.append(f"Club city: {club.city_display}")
        else:
            context_lines.append("Resolved Bundesliga club: (not resolved)")

        if coach:
            context_lines.append(f"Current head coach: {coach.display_name} (Wikidata: {coach.coach_qid})")
            if coach.wikipedia_lang and coach.wikipedia_title:
                context_lines.append(
                    f"Coach Wikipedia page: {coach.wikipedia_lang}wiki / {coach.wikipedia_title}"
                )
        else:
            context_lines.append("Current head coach: (not found)")

        if coach_intro:
            context_lines.append("Coach intro (Wikipedia lead / fallback):")
            context_lines.append(coach_intro)

        prompt = (
            "### SYSTEM\n"
            f"{system_prompt}\n\n"
            "### USER\n"
            f"{user_question.strip()}\n\n"
            "### CONTEXT\n"
            + "\n".join(f"- {line}" for line in context_lines)
            + "\n"
        )
        return prompt

    def run(self, user_question: str) -> str:
        """
        Full pipeline for a single user question. Returns the final prompt string.
        """
        request_id = str(uuid.uuid4())[:8]
        self.logger.info("[%s] New question: %s", request_id, user_question)

        try:
            club, matched_key = self.index.resolve_club(user_question, request_id=request_id)
            self.logger.debug("[%s] Matched key: %s -> club %s", request_id, matched_key, club.display_name)

            coach = self.get_current_coach(club.club_qid, request_id=request_id)
            intro = self.get_coach_intro(coach, request_id=request_id)
            intro = safe_truncate(intro, max_chars=1100)

            return self.build_llm_prompt(user_question, club, coach, intro)

        except EntityResolutionError as e:
            # Provide helpful context: list supported cities
            supported = ", ".join(self.index.list_supported_cities())
            err = f"{e}. Supported cities include: {supported}"
            self.logger.warning("[%s] Entity resolution error: %s", request_id, err)
            return self.build_llm_prompt(user_question, None, None, "", error=err)

        except UpstreamDataError as e:
            self.logger.warning("[%s] Upstream data error: %s", request_id, str(e))
            return self.build_llm_prompt(user_question, None, None, "", error=str(e))

        except Exception as e:
            # Catch-all to keep console UX clean while still logging details
            self.logger.exception("[%s] Unexpected error", request_id)
            return self.build_llm_prompt(
                user_question,
                None,
                None,
                "",
                error=f"Unexpected error: {type(e).__name__}: {e}",
            )


# ---------------
# Logging setup
# ---------------
def setup_logging(debug: bool, log_file: str) -> logging.Logger:
    logger = logging.getLogger("bundesliga_rag")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if debug else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # Make requests/urllib3 logs quieter unless debugging
    logging.getLogger("urllib3").setLevel(logging.WARNING if not debug else logging.INFO)
    return logger


# -------------------------
# Main (console interface)
# -------------------------
def main() -> int:
    """
    Console entry point for the context retrieval script.

    Reads a user question from standard input, orchestrates entity resolution,
    data retrieval from Wikidata and Wikipedia, and outputs the final LLM-ready
    prompt to the console.
    """
    parser = argparse.ArgumentParser(description="Bundesliga coach context retrieval (Wikidata + Wikipedia).")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--log-file", default="rag_retrieval.log", help="Log file path.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="HTTP User-Agent header.")
    args = parser.parse_args()

    logger = setup_logging(args.debug, args.log_file)
    http = HttpClient(user_agent=args.user_agent)
    wd = WikidataClient(http=http, logger=logger)
    wp = WikipediaClient(http=http, logger=logger)

    index = BundesligaIndex(wd=wd, logger=logger)
    init_request_id = str(uuid.uuid4())[:8]
    try:
        index.refresh(request_id=init_request_id)
    except Exception as e:
        logger.exception("[%s] Failed to build Bundesliga index", init_request_id)
        print("Failed to initialize club index. Please try again later.", file=sys.stderr)
        return 2

    pipeline = RetrievalPipeline(index=index, wd=wd, wp=wp, logger=logger)

    print("Bundesliga coach context retrieval. Type a question, or 'exit' to quit.\n")
    while True:
        try:
            user_question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_question:
            continue
        if user_question.lower() in {"exit", "quit"}:
            break

        prompt = pipeline.run(user_question)
        print("\n" + prompt + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())