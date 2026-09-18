from video_dataset.config import load_config
from video_dataset.deduplication.exact import exact_duplicates
from video_dataset.deduplication.near import near_duplicates
from video_dataset.deduplication.qa import dedupe_questions
from video_dataset.deduplication.records import dedupe_timed_texts
from video_dataset.schemas.qa import Evidence, QARecord, QAType
from video_dataset.validation.checks import check_timespan, confidence_status
from video_dataset.validation.grounding import answer_grounding_score
from video_dataset.validation.quality import description_quality_score


def test_timestamp_checks():
    assert check_timespan(0.0, 5.0, 10.0) == (True, None)
    assert check_timespan(6.0, 5.0, 10.0)[0] is False
    assert check_timespan(0.0, 11.0, 10.0)[0] is False
    assert check_timespan(-1.0, 1.0, 10.0)[0] is False
    assert check_timespan(2.0, 2.0, 10.0, allow_instant=False)[0] is False
    assert confidence_status(None, 0.75, 0.5) == "review"
    assert confidence_status(0.8, 0.75, 0.5) == "accepted"
    assert confidence_status(0.6, 0.75, 0.5) == "review"
    assert confidence_status(0.2, 0.75, 0.5) == "rejected"


def test_grounding_and_description_quality():
    assert answer_grounding_score("The red car enters the tunnel.", ["The red car enters a tunnel."]) >= 0.9
    assert answer_grounding_score("A dog barks loudly.", ["The red car enters a tunnel."]) < 0.3
    assert description_quality_score(None) is None
    good = description_quality_score("A red car drives along a wet road at night, lit by yellow street lamps on the left; the camera pans right.")
    bad = description_quality_score("A cinematic beautiful amazing video.")
    assert good is not None and bad is not None and good > bad


def test_exact_and_near_duplicates():
    items = [("a", "The car enters the tunnel."), ("b", "the car enters the tunnel"), ("c", "A dog runs across the beach at sunset."), ("d", "A dog runs across the beach at sunset!")]
    assert exact_duplicates(items) == {"b": "a", "d": "c"}
    near = near_duplicates([("x", "a red car drives along a winding mountain road surrounded by dense forest"), ("y", "a red car drives along a winding mountain road surrounded by dense forest today"), ("z", "two people sit at a kitchen table drinking coffee")], threshold=0.7)
    assert near == {"y": "x"}


def test_timed_text_dedup_keeps_same_text_at_different_times():
    items = [("e1", "The camera pans left.", 0.0, 3.0), ("e2", "The camera pans left.", 0.1, 3.1), ("e3", "The camera pans left.", 40.0, 43.0)]
    dups = dedupe_timed_texts(items, 0.9, 0.5)
    assert dups == {"e2": "e1"}


def _qa(qid, q, a, start, end, events, qtype=QAType.TEMPORAL_ORDERING):
    return QARecord(question_id=qid, video_id="v", type=qtype, question=q, answer=a, evidence=Evidence(start_time=start, end_time=end, event_ids=events))


def test_question_dedup_respects_evidence():
    cfg = load_config(None, {}).deduplication
    qs = [
        _qa("q1", "Which happens first: the car enters the tunnel, or the camera pans left?", "The car enters the tunnel happens first.", 4.0, 8.0, ["e1", "e2"]),
        _qa("q2", "Which happens first: the car enters the tunnel, or the camera pans left?", "The car enters the tunnel happens first.", 4.0, 8.0, ["e1", "e2"]),
        _qa("q3", "Which happens first: the car enters the tunnel, or the camera pans left?", "The camera pans left happens first.", 40.0, 48.0, ["e7", "e8"]),
        _qa("q4", "What is happening at 5.0 seconds?", "The car enters the tunnel.", 5.0, 5.0, ["e1"], QAType.TIMESTAMP),
        _qa("q5", "What is happening at 45.0 seconds?", "The car leaves the tunnel.", 45.0, 45.0, ["e9"], QAType.TIMESTAMP),
    ]
    dups = dedupe_questions(qs, cfg)
    assert dups == {"q2": "q1"}  # q3 has the same wording but different evidence -> kept


def test_numeric_grounding_for_duration_answers():
    from video_dataset.validation.grounding import numeric_grounding_score

    assert numeric_grounding_score("About 1 second (from 42.8s to 43.8s).", (42.8, 43.8)) == 1.0
    assert numeric_grounding_score("About 2.5 minutes (from 0.5s to 150.0s).", (0.5, 150.0)) == 1.0
    assert numeric_grounding_score("Around 1:44 (104 seconds) (starts at 103.8s).", (103.8, 110.0)) == 1.0
    assert numeric_grounding_score("About 30 seconds.", (10.0, 12.0)) == 0.0
