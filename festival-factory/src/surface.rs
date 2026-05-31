//! Surface-form catalog + prompt contract (build spec Part 4 & 6).
//!
//! ~20 DISTINCT STRUCTURAL framings per relationship — structural/format diversity is the extractability
//! lever (FinePhrase/EntiGraph: format variety ≫ lexical paraphrase within one template). Each form is
//! positive and single-fact-faithful (NO contrastive co-occurrence — v3 proved that backfires). The
//! engine generates N lexical variations within each form, so a relationship gets ~20 forms × N lines.

use crate::config::FestivalConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceForm {
    // ── the ~20 POSITIVE structural forms (used for every relationship) ──
    Declarative,
    QaDirect,
    QaMultipleChoice,
    QaWhenWhere,
    ScheduleLine,
    Conversational,
    PersonaFirstTimer,
    PersonaVeteran,
    PersonaPlanner,
    ReverseStage,
    ReverseDay,
    NewsHeadline,
    SocialPost,
    Recommendation,
    Wayfinding,
    Encyclopedic,
    Itinerary,
    Tabular,
    Cloze,
    Announcement,
    Countdown,
    FaqEntry,
    // ── special forms (NOT in the positive core) ──
    Compositional, // multi-fact roster sentences (relation-correctness risk; rosters only)
    MultiHop,      // 2-hop chains (rosters / recommendation)
    Negative,      // density-matched negatives — v4b experiment ONLY, never co-located sparse
}

/// The positive structural forms applied to every per-fact relationship. ~22 distinct kinds.
// NB: QaMultipleChoice is DELIBERATELY EXCLUDED. It lists all candidate days ("Friday, Saturday, or
// Sunday? — Friday"), co-locating the artist with the WRONG days as options — the day-analog of the
// contrastive-stage poison (v4 audit: 70% of day-drift came from this one form). qa_direct gives the
// same day-retrieval shape without the wrong-day proximity. Keep it out of the training distribution.
pub const POSITIVE_FORMS: &[SurfaceForm] = &[
    SurfaceForm::Declarative, SurfaceForm::QaDirect,
    SurfaceForm::QaWhenWhere, SurfaceForm::ScheduleLine, SurfaceForm::Conversational,
    SurfaceForm::PersonaFirstTimer, SurfaceForm::PersonaVeteran, SurfaceForm::PersonaPlanner,
    SurfaceForm::ReverseStage, SurfaceForm::ReverseDay, SurfaceForm::NewsHeadline,
    SurfaceForm::SocialPost, SurfaceForm::Recommendation, SurfaceForm::Wayfinding,
    SurfaceForm::Encyclopedic, SurfaceForm::Itinerary, SurfaceForm::Tabular,
    SurfaceForm::Cloze, SurfaceForm::Announcement, SurfaceForm::Countdown, SurfaceForm::FaqEntry,
];

impl SurfaceForm {
    pub fn name(self) -> &'static str {
        use SurfaceForm::*;
        match self {
            Declarative => "declarative", QaDirect => "qa_direct", QaMultipleChoice => "qa_multiple_choice",
            QaWhenWhere => "qa_when_where", ScheduleLine => "schedule_line", Conversational => "conversational",
            PersonaFirstTimer => "persona_firsttimer", PersonaVeteran => "persona_veteran",
            PersonaPlanner => "persona_planner", ReverseStage => "reverse_stage", ReverseDay => "reverse_day",
            NewsHeadline => "news_headline", SocialPost => "social_post", Recommendation => "recommendation",
            Wayfinding => "wayfinding", Encyclopedic => "encyclopedic", Itinerary => "itinerary",
            Tabular => "tabular", Cloze => "cloze", Announcement => "announcement", Countdown => "countdown",
            FaqEntry => "faq_entry", Compositional => "compositional", MultiHop => "multi_hop", Negative => "negative",
        }
    }

    /// Retrieval-shaped forms blend INTO cpt (spec Part 3.4 — QA during pretraining aids extraction).
    pub fn split(self) -> &'static str {
        use SurfaceForm::*;
        match self {
            QaDirect | QaMultipleChoice | QaWhenWhere | FaqEntry | Cloze | MultiHop => "qa_in_cpt",
            _ => "cpt",
        }
    }

    /// The query token-order this form mirrors (order-matched augmentation, spec Part 3.3).
    pub fn query_order(self) -> &'static str {
        use SurfaceForm::*;
        match self {
            ReverseStage => "leads_with_stage",
            ReverseDay | Itinerary => "leads_with_day",
            _ => "leads_with_artist",
        }
    }

    /// Per-form instruction (the variable USER turn). Each names a concrete structure so the model
    /// produces a genuinely different framing, not the same template reworded.
    pub fn instruction(self) -> &'static str {
        use SurfaceForm::*;
        match self {
            Declarative => "Plain declarative statements of the fact.",
            QaDirect => "Question→answer pairs: 'Q: Which stage does <artist> play? A: <stage>.' (and the day variant).",
            QaMultipleChoice => "Multiple-choice Q&A that names the options, e.g. 'Is <artist> playing Friday, Saturday, or Sunday? — <day>.'",
            QaWhenWhere => "Combined when-and-where Q&A: 'Q: When and where does <artist> play? A: <day> at the <stage> stage.'",
            ScheduleLine => "Terse schedule/agenda lines, e.g. '<stage>, <day>: <artist>.'",
            Conversational => "Casual two-person festival chat where a veteran answers with the specific fact.",
            PersonaFirstTimer => "A first-timer asking nervously where/when to find <artist>, answered with the fact.",
            PersonaVeteran => "A seasoned raver stating the fact with insider tone.",
            PersonaPlanner => "A planner slotting <artist> into a plan: 'For <day>, I've got <artist> at <stage>.'",
            ReverseStage => "Lead with the STAGE: '<stage> hosts <artist> on <day>.'",
            ReverseDay => "Lead with the DAY: 'On <day>, the <stage> stage features <artist>.'",
            NewsHeadline => "News-headline style: '<artist> to play <stage> at <festival>, <day>.'",
            SocialPost => "Short social-media posts hyping the set (an emoji is fine), stating stage+day.",
            Recommendation => "Set recommendations: 'Catch <artist> at <stage> on <day>.'",
            Wayfinding => "Wayfinding directions: 'Head to <stage> on <day> to see <artist>.'",
            Encyclopedic => "Neutral encyclopedic sentences (Wikipedia tone) stating the scheduling fact.",
            Itinerary => "Itinerary entries under a day heading: '<day> — <artist> @ <stage>.'",
            Tabular => "Pipe-delimited rows: 'Artist: <artist> | Stage: <stage> | Day: <day>.'",
            Cloze => "Fill-in-the-blank with the answer given: '<artist> plays the ___ stage at EDC 2026. (<stage>)'",
            Announcement => "Lineup-announcement phrasing: 'Now confirmed — <artist>: <stage>, <day>.'",
            Countdown => "Anticipation/countdown: 'Don't miss <artist> at <stage> this <day>.'",
            FaqEntry => "FAQ entries: 'Where/when is <artist>? — <stage>, <day>.'",
            Compositional => "ONE rich multi-fact sentence bundling the several GIVEN facts naturally; invent nothing beyond them.",
            MultiHop => "2-hop question→answer chaining two given facts (e.g. genre→stage→artist).",
            Negative => "DENSITY-MATCHED negatives correcting one wrong value ('<artist> plays <day> — NOT <wrong day>'). Only ever against the TRUE fact; assert nothing else.",
        }
    }
}

/// Invariant prompt contract (Part 6). Festival naming/format vocab injected from config (factory).
pub fn system_contract(cfg: &FestivalConfig) -> String {
    let mut naming = String::new();
    for st in cfg.stages.values() {
        let mut names = vec![st.camel.clone(), st.spaced.clone()];
        names.extend(st.aliases.clone());
        naming.push_str(&format!("    - {} = {}\n", st.spaced, names.join(" / ")));
    }
    let mut days = String::new();
    for (d, info) in &cfg.days {
        days.push_str(&format!("    - {} = {}\n", d, info.date));
    }
    format!(
        "You generate synthetic training data for continued pre-training of an on-device festival \
assistant. You are given VERIFIED facts about {name} ({dates}, at {venue}). Produce DIVERSE surface \
forms of EXACTLY the given facts.\n\n\
NON-NEGOTIABLE RULES:\n\
1. FAITHFULNESS — assert ONLY facts present in the GIVEN FACTS. Invent no artist, stage, day, time, \
genre, or brand not stated. Never name a stage or day the artist does NOT play (no wrong co-occurrence).\n\
2. ALL DIRECTIONS — when asked to lead with a stage or day, OPEN the sentence with that entity.\n\
3. NAMING — use canonical and colloquial names interchangeably:\n{naming}\
4. FORMAT/UNIT — vary day-name vs date and 12h vs 24h where applicable:\n{days}\
5. OUTPUT — one self-contained line per variation. No numbering, no preamble. Each line stands alone.\n",
        name = cfg.name, dates = cfg.dates, venue = cfg.venue, naming = naming, days = days,
    )
}
